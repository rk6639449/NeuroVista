#!/usr/bin/env python3
"""
You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

This copy is configured for the ReMIND brain-shift dataset (remind-nifti/). ReMINDDataset
streams pairs in which the intraoperative postcontrast T1 (Intraop/*postcontrast*.nii[.gz])
is the fixed/target image and the preoperative T1 resampled onto that grid
(Preop/resampled_to_fixed.nii[.gz]) is the moving/source image.

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
from collections import Counter
from typing import NamedTuple, Sequence
from pathlib import Path

# Third-party imports
import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm import tqdm
import neurite as ne

# Local imports
import voxelmorph as vxm
import glob
from pathlib import Path
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

class ReMINDDataset(IterableDataset):
    """PyTorch IterableDataset for paired Preop (moving) and Intraop (fixed) brain scans from the ReMIND dataset."""

    def __init__(
        self,
        data_dir: str | Path,
        normalize: bool = True,
        target_shape: tuple[int, int, int] = (160, 192, 160),
        pairs: list[dict[str, Path]] | None = None,   # NEW
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.target_shape = target_shape

        # If pairs are passed in (e.g. from a split), use them directly.
        # Otherwise fall back to scanning data_dir as before.
        self.pairs = pairs if pairs is not None else self._find_pairs()

        if not self.pairs:
            raise RuntimeError(f'No valid Preop/Intraop pairs found in {self.data_dir}')

        print(f'Dataset initialized with {len(self.pairs)} subject pairs.')

    def _find_pairs(self) -> list[dict[str, Path]]:
        pairs = []
        subject_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('ReMIND')])

        for subj in subject_dirs:
            intraop_dir = subj / 'Intraop'
            preop_dir = subj / 'Preop'

            intraop_files = list(intraop_dir.glob('*.nii.gz')) if intraop_dir.exists() else []
            preop_files = list(preop_dir.glob('*.nii.gz')) if preop_dir.exists() else []

            if intraop_files and preop_files:
                pairs.append({
                    'fixed_path': intraop_files[0],
                    'moving_path': preop_files[0],
                    'subject_id': subj.name
                })

        return pairs

    def _pad_or_crop(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pads or crops 3D tensor shape (1, H, W, D) to self.target_shape."""
        curr_shape = tensor.shape[1:]
        pad_dims = []

        for curr, target in zip(reversed(curr_shape), reversed(self.target_shape)):
            diff = target - curr
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_dims.extend([pad_before, pad_after])
            else:
                pad_dims.extend([0, 0])

        if any(p > 0 for p in pad_dims):
            tensor = F.pad(tensor.unsqueeze(0), pad_dims, mode='constant', value=0).squeeze(0)

        # Crop if larger
        h, w, d = tensor.shape[1:]
        th, tw, td = self.target_shape
        sh = (h - th) // 2
        sw = (w - tw) // 2
        sd = (d - td) // 2

        return tensor[:, sh:sh + th, sw:sw + tw, sd:sd + td]

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        nii = nib.load(str(path))
        data = nii.get_fdata().astype(np.float32)

        if self.normalize:
            min_val, max_val = data.min(), data.max()
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)

        tensor = torch.from_numpy(data).unsqueeze(0)  # (1, H, W, D)
        return self._pad_or_crop(tensor)

    def __iter__(self):
        worker_info = get_worker_info()

        # Partition pairs across workers to prevent duplicate processing
        if worker_info is None:
            worker_pairs = self.pairs
        else:
            # Seed worker rng independently
            np.random.seed(worker_info.seed % (2**32))
            per_worker = int(np.ceil(len(self.pairs) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.pairs))
            worker_pairs = self.pairs[iter_start:iter_end]

        while True:
            idx = np.random.randint(0, len(worker_pairs))
            pair = worker_pairs[idx]

            moving = self._load_and_preprocess(pair['moving_path'])
            fixed = self._load_and_preprocess(pair['fixed_path'])

            yield {'source': moving, 'target': fixed, 'subject': pair['subject_id']}
def split_pairs(data_dir: str | Path, test_frac: float = 0.2, seed: int = 42):
    """
    Scans data_dir once and returns (train_pairs, test_pairs) split by subject,
    so no subject appears in both sets.
    """
    data_dir = Path(data_dir)
    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('ReMIND')])

    pairs = []
    for subj in subject_dirs:
        intraop_dir = subj / 'Intraop'
        preop_dir = subj / 'Preop'

        intraop_files = list(intraop_dir.glob('*.nii.gz')) if intraop_dir.exists() else []
        preop_files = list(preop_dir.glob('*.nii.gz')) if preop_dir.exists() else []

        if intraop_files and preop_files:
            pairs.append({
                'fixed_path': intraop_files[0],
                'moving_path': preop_files[0],
                'subject_id': subj.name
            })

    if not pairs:
        raise RuntimeError(f'No valid Preop/Intraop pairs found in {data_dir}')

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))

    n_test = max(1, int(round(len(pairs) * test_frac)))
    test_idx = set(indices[:n_test])

    train_pairs = [p for i, p in enumerate(pairs) if i not in test_idx]
    test_pairs = [p for i, p in enumerate(pairs) if i in test_idx]

    print(f'Split {len(pairs)} subjects -> {len(train_pairs)} train / {len(test_pairs)} test')
    return train_pairs, test_pairs

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


def parse_shape(value: str) -> Sequence[int] | str | None:
    """
    Parse a --shape argument: 'auto', 'none', or a comma-separated D,H,W triple.
    """
    value = value.strip().lower()
    if value == 'auto':
        return 'auto'
    if value in ('none', 'off'):
        return None
    return tuple(int(v) for v in value.split(','))


def main():
    parser = argparse.ArgumentParser(description='Train 3D VoxelMorph on ReMIND data')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01)
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    parser.add_argument(
        '--data-root',
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help='Path to the remind-nifti dataset root',
    )
    parser.add_argument(
        '--shape',
        type=str,
        default='auto',
        help="Target volume shape: 'auto' (most common), 'none', or D,H,W like 256,256,176",
    )
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

    # Create dataset and dataloader
    dataset_path = 'dataset_2'
    train_pairs, test_pairs = split_pairs(dataset_path, test_frac=0.2, seed=42)

    train_dataset = ReMINDDataset(data_dir=dataset_path, normalize=True, pairs=train_pairs)
    test_dataset = ReMINDDataset(data_dir=dataset_path, normalize=True, pairs=test_pairs)

    train_loader = iter(
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    )
    test_loader = iter(
        DataLoader(
            test_dataset,
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
