"""Fast server-side slice rendering (PIL) for the in-browser viewer.

Grayscale views carry the image; overlay views are transparent RGBA layers the
client composites on top of any base layer. All renderers return PNG bytes.
"""
from __future__ import annotations

import io

import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

from .preprocess import world_to_voxel

COLORS = {
    'green': (60, 220, 120),
    'red': (255, 84, 84),
    'yellow': (255, 214, 90),
    'cyan': (80, 210, 255),
    'violet': (196, 140, 255),
}
_PLANE_AXIS = {'axial': 2, 'coronal': 1, 'sagittal': 0}


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()


def window(data: np.ndarray) -> tuple:
    """Robust display window -> (uint8 array, lo, hi). Already-windowed pass through."""
    if data.dtype == np.uint8:
        return data, 0.0, 255.0
    lo, hi = np.percentile(data, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(np.min(data)), float(np.max(data)) + 1e-6
    scaled = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return np.uint8(scaled * 255), float(lo), float(hi)


def _slice(data: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """One 2D display slice: rows = second non-plane axis (top = high index)."""
    return np.take(data, idx, axis=_PLANE_AXIS[plane]).T[::-1].copy()


def _axes(plane: str) -> tuple:
    """(col_axis, row_axis) for a plane."""
    ax = _PLANE_AXIS[plane]
    others = [a for a in range(3) if a != ax]
    return others[0], others[1]


def orientation(affine: np.ndarray, plane: str) -> dict:
    """(top, bottom, left, right) RAS letters for the displayed plane."""
    col_axis, row_axis = _axes(plane)
    names = {0: ('L', 'R'), 1: ('P', 'A'), 2: ('I', 'S')}
    col_sign = np.sign(affine[col_axis, col_axis]) or 1.0
    row_sign = np.sign(affine[row_axis, row_axis]) or 1.0
    neg, pos = names[col_axis]
    left, right = (neg, pos) if col_sign > 0 else (pos, neg)
    neg, pos = names[row_axis]
    bottom, top = (neg, pos) if row_sign > 0 else (pos, neg)
    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}


def shape_for(data: np.ndarray, plane: str) -> tuple:
    """(width, height) of the rendered slice for a volume."""
    col_axis, row_axis = _axes(plane)
    return data.shape[col_axis], data.shape[row_axis]


def render_gray(data: np.ndarray, plane: str, idx: int) -> bytes:
    """Grayscale slice PNG (RGB)."""
    u8, _, _ = window(data)
    sl = _slice(u8, plane, idx)
    return _png(Image.fromarray(sl).convert('RGB'))


def render_overlay(mask: np.ndarray, plane: str, idx: int, color: str = 'green',
                   markers: list | None = None, affine: np.ndarray | None = None) -> bytes:
    """Transparent RGBA layer: mask contour + optional centroid markers + labels."""
    sl = _slice(mask.astype(np.uint8), plane, idx) > 0
    rgba = np.zeros(sl.shape + (4,), dtype=np.uint8)
    rgb = COLORS.get(color, COLORS['green'])
    rgba[sl] = (*rgb, 45)
    if sl.any():
        rgba[find_boundaries(sl, mode='outer')] = (*rgb, 255)
    img = Image.fromarray(rgba, mode='RGBA')

    if markers and affine is not None:
        draw = ImageDraw.Draw(img)
        ax = _PLANE_AXIS[plane]
        col_axis, row_axis = _axes(plane)
        height = sl.shape[0]
        for mk in markers:
            vox = world_to_voxel(affine, mk['mm'])
            if abs(vox[ax] - idx) > 0.75:
                continue
            x = float(vox[col_axis])
            y = float(height - 1 - vox[row_axis])
            mrgb = COLORS.get(mk.get('color', 'yellow'), COLORS['yellow'])
            r = 7
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(*mrgb, 255), width=3)
            draw.line([x - r - 5, y, x + r + 5, y], fill=(*mrgb, 255), width=1)
            draw.line([x, y - r - 5, x, y + r + 5], fill=(*mrgb, 255), width=1)
            if mk.get('label'):
                draw.text((x + r + 6, y - r - 6), mk['label'], fill=(*mrgb, 255))
    return _png(img)


def render_flow(disp: np.ndarray, base: np.ndarray, plane: str, idx: int,
                alpha: float = 0.6) -> bytes:
    """Displacement-magnitude heatmap (inferno) blended over the fixed image."""
    mag = np.linalg.norm(disp, axis=0)
    heat = (cm.inferno(mag / (float(mag.max()) + 1e-8))[..., :3] * 255).astype(np.uint8)
    base_u8, _, _ = window(base)
    sl_base = _slice(base_u8, plane, idx).astype(np.float32)[..., None]
    # RGB slice: transpose the two spatial axes only, keep the channel axis last.
    sl_heat = np.take(heat, idx, axis=_PLANE_AXIS[plane]).transpose(1, 0, 2)[::-1]
    comp = np.uint8(sl_base * (1.0 - alpha) + sl_heat.astype(np.float32) * alpha)
    return _png(Image.fromarray(comp))


def render_checker(a: np.ndarray, b: np.ndarray, plane: str, idx: int,
                   tile: int = 32) -> bytes:
    """Checkerboard blend of two volumes (classic registration QC view)."""
    ua, _, _ = window(a)
    ub, _, _ = window(b)
    sa, sb = _slice(ua, plane, idx), _slice(ub, plane, idx)
    h, w = sa.shape
    yy, xx = np.mgrid[0:h, 0:w]
    take_a = ((yy // tile + xx // tile) % 2 == 0)
    comp = np.where(take_a, sa, sb)
    return _png(Image.fromarray(np.uint8(comp)).convert('RGB'))


def tumor_mesh(mask: np.ndarray, affine: np.ndarray, level: float = 0.5) -> dict | None:
    """Marching-cubes tumour surface in RAS+ mm for the 3D view (JSON-friendly)."""
    from skimage import measure

    if int(mask.sum()) < 20:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(np.float32), level=level, step_size=2)
    except Exception:
        return None
    homo = np.c_[verts, np.ones(len(verts))]
    verts_mm = (np.asarray(affine) @ homo.T).T[:, :3]
    return {'vertices': verts_mm.round(2).tolist(),
            'faces': faces.astype(int).tolist()}
