"""Tumour relocation through the deformation field + surgical shift metrics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from voxelmorph.py.utils import jacobian_determinant

from .preprocess import voxel_to_world, world_to_voxel

_DIR_LABELS = ((1.0, 'right', 'left'), (-1.0, 'right', 'left'),
               (2.0, 'anterior', 'posterior'), (-2.0, 'anterior', 'posterior'),
               (3.0, 'superior', 'inferior'), (-3.0, 'superior', 'inferior'))


def centroid_voxel(mask: np.ndarray) -> np.ndarray:
    """Centroid of a binary mask in voxel coordinates (continuous)."""
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        raise ValueError('Empty tumour mask - cannot compute a centroid.')
    return np.array([c.mean() for c in idx], dtype=np.float64)


def sample_disp(disp: np.ndarray, vox: np.ndarray) -> np.ndarray:
    """Trilinearly sample a (3, D, H, W) displacement field at a voxel point."""
    coords = np.asarray(vox, dtype=np.float64).reshape(3, 1)
    return np.array([
        map_coordinates(disp[c], coords, order=1, mode='nearest')[0]
        for c in range(3)
    ])


def invert_point(disp: np.ndarray, moving_vox: np.ndarray) -> np.ndarray:
    """
    Map a point from moving space to fixed space.

    The field satisfies ``moving = fixed + disp(fixed)``; solve for ``fixed``
    by fixed-point iteration (converges for the smooth, sub-voxel-scale fields
    VoxelMorph produces).
    """
    p = np.asarray(moving_vox, dtype=np.float64).copy()
    for _ in range(30):
        p_new = p - sample_disp(disp, p)
        if np.max(np.abs(p_new - p)) < 1e-4:
            return p_new
        p = p_new
    return p


def direction_phrase(vec: np.ndarray) -> str:
    """Human-readable dominant direction of an RAS+ mm vector."""
    order = np.argsort(np.abs(vec))[::-1]
    primary = int(order[0]) + 1 if vec[order[0]] > 0 else -(int(order[0]) + 1)
    for sign, pos, neg in _DIR_LABELS:
        if primary == sign:
            return pos if vec[order[0]] >= 0 else neg
    return 'minimal'


def hd95(a: np.ndarray, b: np.ndarray, zooms) -> float | None:
    """
    Symmetric 95th-percentile Hausdorff distance (mm) between two masks.

    Landmark-free TRE proxy: how far the relocated tumour surface sits from the
    intraoperative residual surface.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not a.any() or not b.any():
        return None
    surf_a = a & ~binary_erosion(a)
    surf_b = b & ~binary_erosion(b)
    if not surf_a.any():
        surf_a = a
    if not surf_b.any():
        surf_b = b
    sampling = tuple(float(z) for z in zooms)
    d_to_b = distance_transform_edt(~b, sampling=sampling)
    d_to_a = distance_transform_edt(~a, sampling=sampling)
    return float(max(np.percentile(d_to_b[surf_a], 95),
                     np.percentile(d_to_a[surf_b], 95)))


def relocate(mask_moving: np.ndarray, disp: np.ndarray, moving_affine: np.ndarray,
             fixed_affine: np.ndarray, moving_zooms: tuple, mask_affine_aligned: np.ndarray,
             mask_shifted: np.ndarray) -> dict:
    """
    Compute where the tumour is *now* (after brain shift) and by how much it moved.

    Three positions are reported:

    - ``preop``        : centroid in the original preoperative scanner coordinates
    - ``affine_aligned``: centroid after geometric resampling only (before the
      learned deformation) - the "initial plan" reference in intraoperative space
    - ``shifted``      : centroid after the learned deformation field - "now"
    """
    pre_vox = centroid_voxel(mask_moving)
    pre_mm = voxel_to_world(moving_affine, pre_vox)

    aff_vox = centroid_voxel(mask_affine_aligned)
    aff_mm = voxel_to_world(fixed_affine, aff_vox)

    shf_vox = centroid_voxel(mask_shifted)
    shf_mm = voxel_to_world(fixed_affine, shf_vox)

    shift_mm = shf_mm - aff_mm
    disp_at_tumour_vox = sample_disp(disp, aff_vox)
    disp_at_tumour_mm = disp_at_tumour_vox * np.asarray(moving_zooms)

    # Local deformation quality around the tumour (Jacobian determinant).
    # jacobian_determinant expects channels-last (*vol_shape, nb_dims).
    inner = np.s_[4:-4, 4:-4, 4:-4]
    try:
        jac = jacobian_determinant(np.moveaxis(disp, 0, -1))
        jac_region = jac[inner]
        jac_stats = {
            'min': float(jac_region.min()),
            'mean': float(jac_region.mean()),
            'pct_nonpositive': float((jac_region <= 0).mean() * 100.0),
        }
    except Exception as exc:
        jac_stats = {'error': str(exc)}

    mag = np.linalg.norm(disp, axis=0)
    return {
        'preop': {'voxel': pre_vox.round(2).tolist(), 'mm': pre_mm.round(2).tolist()},
        'affine_aligned': {'voxel': aff_vox.round(2).tolist(), 'mm': aff_mm.round(2).tolist()},
        'shifted': {'voxel': shf_vox.round(2).tolist(), 'mm': shf_mm.round(2).tolist()},
        'shift_mm': shift_mm.round(2).tolist(),
        'shift_magnitude_mm': float(np.linalg.norm(shift_mm)),
        'shift_direction': direction_phrase(shift_mm),
        'displacement_at_tumour_vox': disp_at_tumour_vox.round(3).tolist(),
        'displacement_at_tumour_mm': disp_at_tumour_mm.round(2).tolist(),
        'field_magnitude_mm': {
            'mean': float(mag.mean() * np.mean(moving_zooms)),
            'max': float(mag.max() * np.mean(moving_zooms)),
            'p95': float(np.percentile(mag, 95) * np.mean(moving_zooms)),
        },
        'jacobian': jac_stats,
    }
