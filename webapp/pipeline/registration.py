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
        Pad both volumes to an UNet-compatible shape, register, then crop back.

        Returns ``(disp, warped, runtime_s)`` at the original (unpadded) shape.
        """
        pads = pad_amounts(target.shape)
        disp, warped, runtime = self.register(
            pad_to(source, pads), pad_to(target, pads)
        )
        # disp has a leading channel axis (3, D, H, W): pad/crop only the spatial dims.
        return crop_to(disp, [(0, 0), *pads]), crop_to(warped, pads), runtime


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
