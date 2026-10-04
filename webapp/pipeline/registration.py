"""VoxelMorph inference: load trained weights and register moving -> fixed."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import voxelmorph as vxm
from voxelmorph.nn import functional as vxm_f

from .preprocess import SHAPE_MULTIPLE, crop_to, pad_amounts, pad_to

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / 'voxelmorph' / 'final.pt'
NB_FEATURES = [16, 16, 16, 16, 16]


class Registrar:
    """Wraps the trained VxmPairwise checkpoint for inference."""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str | None = None) -> None:
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f'Trained VoxelMorph weights not found at {weights}. '
                'Train the model or pass --weights.'
            )
        self.weights = weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = vxm.nn.models.VxmPairwise(
            ndim=3,
            source_channels=1,
            target_channels=1,
            nb_features=NB_FEATURES,
            integration_steps=0,
        )
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def register(self, source: np.ndarray, target: np.ndarray):
        """
        Register ``source`` onto ``target`` (both normalized, same padded shape).

        Returns
        -------
        disp : (3, D, H, W) float32 displacement field in voxel units (fixed grid)
        warped : (D, H, W) float32 source warped into target space
        runtime_s : wall-clock seconds
        """
        src = torch.from_numpy(source)[None, None].to(self.device)
        tgt = torch.from_numpy(target)[None, None].to(self.device)
        start = time.time()
        disp, warped = self.model(
            src, tgt, return_warped_source=True, return_field_type='displacement'
        )
        if self.device == 'cuda':
            torch.cuda.synchronize()
        runtime = time.time() - start
        return (
            disp[0].cpu().numpy().astype(np.float32),
            warped[0, 0].cpu().numpy().astype(np.float32),
            runtime,
        )

    def run_padded(self, source: np.ndarray, target: np.ndarray):
        """
        Register at an UNet-compatible shape (multiples of 32), downscaling large
        volumes so the network stays within its memory budget, then restore the
        field and warped image to the original grid.

        Returns ``(disp, warped, runtime_s)`` at the original (unpadded) shape.
        """
        full = tuple(target.shape)
        factor, low = reg_resolution(full)
        if factor > 1.0:
            src, tgt = _zoom_to(source, low), _zoom_to(target, low)
        else:
            low, src, tgt = full, source, target

        pads = pad_amounts(low)
        disp, warped, runtime = self.register(pad_to(src, pads), pad_to(tgt, pads))
        # disp has a leading channel axis (3, D, H, W): pad/crop only the spatial dims.
        disp = crop_to(disp, [(0, 0), *pads])
        warped = crop_to(warped, pads)

        if factor > 1.0:
            ratios = tuple(s / l for s, l in zip(full, low))
            disp = scale_field_back(disp, full, low, ratios)
            warped = _zoom_to(warped, full, order=1).astype(np.float32)
        return disp, warped, runtime


def warp_volume(volume: np.ndarray, disp: np.ndarray, method: str = 'linear') -> np.ndarray:
    """Warp a volume through a displacement field (voxel units, fixed grid)."""
    vol_t = torch.from_numpy(volume.astype(np.float32))[None, None]
    disp_t = torch.from_numpy(disp.astype(np.float32))[None]
    out = vxm_f.spatial_transform(vol_t, disp_t, method=method)
    return out[0, 0].numpy()


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation QC score between two volumes."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Adaptive resolution (memory safety for 512^2 scans)
# --------------------------------------------------------------------------
REG_BUDGET = 28_000_000  # max voxels fed to the UNet


def _zoom_to(arr: np.ndarray, target, order: int = 1) -> np.ndarray:
    """Resample to an exact target shape (scipy zoom + off-by-one cleanup)."""
    from scipy import ndimage

    target = tuple(int(t) for t in target)
    if tuple(arr.shape) == target:
        return arr
    factors = [t / s for t, s in zip(target, arr.shape)]
    out = ndimage.zoom(np.ascontiguousarray(arr), factors, order=order, mode='nearest')
    if out.shape != target:
        if all(o >= t for o, t in zip(out.shape, target)):
            out = out[tuple(slice(0, t) for t in target)]
        else:
            out = np.pad(out, [(0, max(0, t - o))
                               for t, o in zip(target, out.shape)])
    return out


def reg_resolution(shape: tuple) -> tuple:
    """
    Return ``(factor, low_shape)``: the downscale factor used for registration.

    ``factor == 1.0`` means the volume is registered at native resolution.
    """
    factor = 1.0
    while factor < 8.0:
        low = [max(16, int(s / factor)) for s in shape]
        padded = [int(SHAPE_MULTIPLE * np.ceil(dim / SHAPE_MULTIPLE)) for dim in low]
        if int(np.prod(padded)) <= REG_BUDGET:
            return factor, tuple(low)
        factor *= 1.08
    return factor, tuple(max(16, int(s / factor)) for s in shape)


def scale_field_back(disp: np.ndarray, full_shape: tuple, low_shape: tuple,
                     ratios) -> np.ndarray:
    """
    Convert a low-resolution displacement field to the full grid.

    Displacements are voxel units of their own grid (``disp_to_coords`` adds the
    raw values to a voxel meshgrid), so a component measured in low-res voxels
    equals ``value * (full_i / low_i)`` full-res voxels. Spatial upsampling is
    linear interpolation of the smooth field.
    """
    return np.stack([
        _zoom_to(disp[c], full_shape, order=1) * ratios[c]
        for c in range(disp.shape[0])
    ]).astype(np.float32)
