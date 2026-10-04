#!/usr/bin/env python3
"""
You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
    pp 1788-1800. 2019.

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. MedIA: Medical Image Analysis. (57).
    pp 226-236, 2019

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# Core library imports
import argparse
from typing import Sequence
from pathlib import Path

# Third-party imports
import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import neurite as ne

# Local imports
import voxelmorph as vxm


class VxmIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for infinite VoxelMorph registration data.
    """

    def __init__(self, device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        device : str
            Device to place tensors on.
        """
        self.teramedical_root = Path('/autofs/cluster/dalcalab1/data/teramedical/processed')
        self.device = device
        self.oasis_path = self.teramedical_root / 'OASIS/neurite/proc-v1.0'
        self._get_vol_paths()

    def __iter__(self):
        """
        Generate infinite stream of random volume pairs.

        Yields
        ------
        dict
            A dictionary containing the source and target volumes.
        """
        while True:
            idx1, idx2 = np.random.randint(0, len(self.folder_abspaths), size=2)

            # Get paths
            source_path = self.folder_abspaths[idx1]
            target_path = self.folder_abspaths[idx2]

            # Get niftis
            source_nii = nib.load(f'{source_path}/vol_norm_aligned.nii.gz')
            target_nii = nib.load(f'{target_path}/vol_norm_aligned.nii.gz')

            source = torch.from_numpy(source_nii.get_fdata()).float().unsqueeze(0)
            target = torch.from_numpy(target_nii.get_fdata()).float().unsqueeze(0)

            yield {'source': source, 'target': target}

    def _get_vol_paths(self) -> None:
        """
        Get the absolute paths of the volume folders.
        """
        self.folder_abspaths = []

        for i in range(1, 450):
            folder = self.oasis_path / f'OASIS_OAS1_{i:04}_MR1'

            if folder.exists():
                self.folder_abspaths.append(folder)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    steps_per_epoch: int,
    device: str = 'cuda'
) -> float:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The VoxelMorph model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    image_loss_fn : nn.Module
        The image loss function to use.
    grad_loss_fn : nn.Module
        The gradient loss function to use.
    loss_weights : Sequence[float]
        The weights for the image and gradient losses.
    steps_per_epoch : int
    """

    model.train()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        batch = next(dataloader)
        optimizer.zero_grad()

        # Move to device in training loop (not dataloader/dataset!)
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        # Get the displacement and the warped source image from the model
        displacement, warped_source = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement'
        )

        img_loss = image_loss_fn(target, warped_source)
        grad_loss = grad_loss_fn(displacement)

        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps_per_epoch


def main():
    parser = argparse.ArgumentParser(description='Train 3D VoxelMorph on OASIS data')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100_000, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01)
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)

    # Setup losses and optimizer
    image_loss_fn = ne.nn.modules.MSE()
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    train_dataset = VxmIterableDataset(device=device)
    train_loader = iter(
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    best_loss = float('inf')
    for epoch in tqdm(range(args.epochs), desc='Epochs'):

        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            steps_per_epoch=args.steps_per_epoch,
            device=device
        )

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}')

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:

            checkpoint_path = output_dir / f'checkpoint_epoch{epoch + 1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / 'best.pt'
            torch.save(model.state_dict(), best_path)

    # Save final model
    final_path = output_dir / 'final.pt'
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved to {final_path}')


if __name__ == '__main__':
    main()
