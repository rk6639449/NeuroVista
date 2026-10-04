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


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / 'remind-nifti'

# VxmPairwise's default 5-level UNet downsamples 5 times, so every spatial dimension must
# be a multiple of 2 ** 5 = 32 or the decoder's skip connections will not line up.
SHAPE_MULTIPLE = 32


class _ReMINDPair(NamedTuple):
    """A single subject's (fixed, moving) volume paths on a shared voxel grid."""

    subject: str
    fixed: Path   # Intraop postcontrast T1 -- defines the target grid
    moving: Path  # Preop T1 already resampled onto the fixed grid
    shape: tuple  # common (D, H, W) shape of both volumes


class ReMINDDataset(IterableDataset):
    """
    Infinite PyTorch IterableDataset over ReMIND preop/intraop registration pairs.

    For each ``ReMIND-*`` subject folder under ``root`` this dataset yields:

    - ``target`` (fixed image): the intraoperative postcontrast T1 volume,
      ``<subject>/Intraop/*postcontrast*.nii[.gz]``.
    - ``source`` (moving image): the preoperative T1 already resampled onto the fixed grid
      by ``Preprocessing/Resampling.py``, ``<subject>/Preop/resampled_to_fixed.nii[.gz]``.

    Volumes are min-max normalized to [0, 1], given a leading channel dimension, and --
    unless ``target_shape`` is None -- center cropped/padded to ``target_shape``. The very
    same crop/pad is applied to both volumes of a pair (they share one grid), so relative
    alignment is preserved while every batch stays collatable despite the handful of
    subjects with a non-standard field of view.

    Subjects are sampled uniformly at random forever, so the stream never exhausts and
    ``train_epoch`` can consume a fixed number of steps per epoch.
    """

    FIXED_PATTERN = '*postcontrast*.nii*'
    MOVING_CANDIDATES = ('resampled_to_fixed.nii.gz', 'resampled_to_fixed.nii')

    def __init__(
        self,
        root: str = str(DEFAULT_DATA_ROOT),
        target_shape: Sequence[int] | str | None = 'auto',
        normalize: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        root : str
            Path to the remind-nifti dataset root.
        target_shape : sequence of int, 'auto', or None
            Volume shape every sample is center cropped/padded to. 'auto' picks the most
            common shape in the dataset and pads each dimension up to a multiple of
            ``SHAPE_MULTIPLE`` (32) so the default UNet can consume every volume; None
            keeps native shapes (then all volumes in a batch must already share a shape).
        normalize : bool
            If True, min-max normalize each volume to [0, 1].
        seed : int or None
            Optional base RNG seed for reproducible sampling (offset per worker).
        """
        self.root = Path(root)
        self.normalize = normalize
        self.seed = seed
        self.pairs, skipped = self._discover_pairs()

        if not self.pairs:
            raise FileNotFoundError(f'No complete ReMIND pairs found under {self.root}')
        if skipped:
            print(f'ReMINDDataset: skipped {len(skipped)} subject(s): {", ".join(skipped)}')

        if target_shape == 'auto':
            target_shape = Counter(pair.shape for pair in self.pairs).most_common(1)[0][0]
            # Pad up (never crop anatomy away) to the next UNet-compatible size.
            target_shape = tuple(-(-s // SHAPE_MULTIPLE) * SHAPE_MULTIPLE for s in target_shape)
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        print(
            f'ReMINDDataset: {len(self.pairs)} pairs from {self.root} '
            f'(target_shape={self.target_shape or "native"})'
        )

    def __iter__(self):
        """
        Generate an infinite stream of randomly sampled registration pairs.

        Yields
        ------
        dict
            ``source`` (moving, Preop) and ``target`` (fixed, Intraop) volumes as
            (1, D, H, W) float tensors, plus the ``subject`` id for traceability.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        pairs = self.pairs[worker_id::num_workers]
        if not pairs:
            raise RuntimeError(f'worker {worker_id}/{num_workers} received no ReMIND pairs')

        rng = np.random.default_rng(None if self.seed is None else self.seed + worker_id)
        while True:
            pair = pairs[rng.integers(len(pairs))]
            yield {
                'source': self._load(pair.moving),
                'target': self._load(pair.fixed),
                'subject': pair.subject,
            }

    def _discover_pairs(self) -> tuple[list[_ReMINDPair], list[str]]:
        """
        Locate complete fixed/moving pairs, validating that each pair shares a shape.

        Returns
        -------
        (pairs, skipped) : tuple
            The discovered pairs and human-readable reasons for skipped subjects.
        """
        pairs = []
        skipped = []

        if not self.root.is_dir():
            raise FileNotFoundError(f'ReMIND dataset root not found: {self.root}')

        for subject_dir in sorted(self.root.glob('ReMIND-*')):
            if not subject_dir.is_dir():
                continue

            fixed = self._first_match(subject_dir / 'Intraop', self.FIXED_PATTERN)
            moving = None
            for candidate in self.MOVING_CANDIDATES:
                moving = self._first_match(subject_dir / 'Preop', candidate)
                if moving is not None:
                    break

            if fixed is None or moving is None:
                missing = 'Intraop postcontrast' if fixed is None else 'Preop resampled_to_fixed'
                skipped.append(f'{subject_dir.name} (missing {missing})')
                continue

            fixed_shape = nib.load(str(fixed)).shape
            moving_shape = nib.load(str(moving)).shape
            if fixed_shape != moving_shape:
                skipped.append(f'{subject_dir.name} (shape {fixed_shape} != {moving_shape})')
                continue

            pairs.append(_ReMINDPair(subject_dir.name, fixed, moving, tuple(fixed_shape)))

        return pairs, skipped

    @staticmethod
    def _first_match(folder: Path, pattern: str) -> Path | None:
        """Return the first sorted file in ``folder`` matching ``pattern``, or None."""
        if not folder.is_dir():
            return None
        matches = sorted(folder.glob(pattern))
        return matches[0] if matches else None

    def _load(self, path: Path) -> torch.Tensor:
        """Load one NIfTI volume as a normalized (1, D, H, W) float tensor."""
        volume = nib.load(str(path)).get_fdata(dtype=np.float32)

        if self.normalize:
            vmin = float(volume.min())
            vmax = float(volume.max())
            if vmax > vmin:
                volume = (volume - vmin) / (vmax - vmin)

        if self.target_shape is not None:
            volume = self._fit_shape(volume)

        return torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)

    def _fit_shape(self, volume: np.ndarray) -> np.ndarray:
        """Center crop (or zero-pad) a volume to ``target_shape``."""
        if volume.ndim != len(self.target_shape):
            raise ValueError(f'volume is {volume.ndim}D but target_shape is {self.target_shape}')

        slices = []
        padding = []
        for current, desired in zip(volume.shape, self.target_shape):
            if current >= desired:
                start = (current - desired) // 2
                slices.append(slice(start, start + desired))
                padding.append((0, 0))
            else:
                slices.append(slice(None))
                padding.append((0, desired - current))

        volume = volume[tuple(slices)]
        return np.pad(volume, padding) if any(pad != (0, 0) for pad in padding) else volume


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
    parser.add_argument('--epochs', type=int, default=100_000, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01)
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
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

    # Create dataloader
    train_dataset = ReMINDDataset(root=args.data_root, target_shape=parse_shape(args.shape))
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
