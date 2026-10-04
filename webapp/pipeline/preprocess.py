"""Image loading, geometric resampling, normalization and model-compatible padding."""
from __future__ import annotations

from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# The 5-level UNet used in training requires every spatial dim to be a multiple of 32.
SHAPE_MULTIPLE = 32


@dataclass
class Volume:
    """A 3D volume with its world transform."""

    data: np.ndarray                      # (D, H, W) float32
    affine: np.ndarray                    # (4, 4) voxel -> RAS+ mm
    zooms: tuple = field(default=(1.0, 1.0, 1.0))
    path: str = ''

    @classmethod
    def load(cls, path) -> 'Volume':
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj).astype(np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f'Expected a 3D NIfTI volume, got shape {data.shape} in {path}')
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        return cls(data, np.asarray(img.affine, dtype=np.float64), zooms, str(path))

    @property
    def shape(self) -> tuple:
        return tuple(self.data.shape)

    def save(self, path) -> None:
        nib.save(nib.Nifti1Image(self.data, self.affine), str(path))


def normalize(data: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Robust min-max normalization to [0, 1] using percentiles."""
    lo, hi = np.percentile(data, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resample_to_grid(moving: Volume, fixed_shape: tuple, fixed_affine: np.ndarray,
                     order: int = 1) -> np.ndarray:
    """Geometrically resample ``moving`` onto the fixed image's voxel grid."""
    src = nib.Nifti1Image(moving.data, moving.affine)
    ref = nib.Nifti1Image(np.zeros(fixed_shape, np.float32), fixed_affine)
    out = resample_from_to(src, (ref.shape, ref.affine), order=order)
    return out.get_fdata(dtype=np.float32)


def pad_amounts(shape: tuple, multiple: int = SHAPE_MULTIPLE) -> list:
    """Per-dimension (before, after) zero-pads needed to reach ``multiple``."""
    pads = []
    for s in shape:
        target = -(-s // multiple) * multiple
        total = target - s
        pads.append((total // 2, total - total // 2))
    return pads


def pad_to(volume: np.ndarray, pads: list) -> np.ndarray:
    """Zero-pad an array with the given per-dimension pads."""
    if all(p == (0, 0) for p in pads):
        return volume
    return np.pad(volume, pads, mode='constant')


def crop_to(volume: np.ndarray, pads: list) -> np.ndarray:
    """Remove padding added by :func:`pad_to`."""
    if all(p == (0, 0) for p in pads):
        return volume
    slices = tuple(slice(before, None if after == 0 else -after) for before, after in pads)
    return volume[slices]


def world_to_voxel(affine: np.ndarray, xyz) -> np.ndarray:
    """Map RAS+ mm world coordinates to continuous voxel indices."""
    vec = np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    return (np.linalg.inv(affine) @ vec)[:3]


def voxel_to_world(affine: np.ndarray, ijk) -> np.ndarray:
    """Map continuous voxel indices to RAS+ mm world coordinates."""
    vec = np.append(np.asarray(ijk, dtype=np.float64), 1.0)
    return (affine @ vec)[:3]
