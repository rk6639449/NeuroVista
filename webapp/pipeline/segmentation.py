"""Tumour segmentation stage.

Pluggable engines, tried in order:

1. ``manual``  - mask uploaded by the clinician (always wins when provided)
2. ``model``   - trained segmentation model at ``segmentation_model.pt`` (PENDING: the
                 model will be supplied later; the loader is ready and activates
                 automatically as soon as the file exists)
3. ``pseudo``  - intensity-based demo fallback so the rest of the pipeline can be
                 exercised end-to-end before the real model arrives
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

MODEL_PATH = Path(__file__).resolve().parents[1] / 'segmentation_model.pt'


def load_model():
    """Load the segmentation model if present. Returns a callable or None."""
    if not MODEL_PATH.is_file():
        return None
    import torch

    try:
        model = torch.jit.load(str(MODEL_PATH), map_location='cpu')
    except Exception:
        model = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
    model.eval()
    return model


def _predict(model, volume: np.ndarray) -> np.ndarray:
    """Run the loaded model on a normalized volume -> binary mask."""
    import torch

    x = torch.from_numpy(volume.astype(np.float32))[None, None]
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    logits = out[0, 0].numpy()
    return logits > 0.0 if logits.min() < 0 < logits.max() else logits > 0.5


def _pseudo_mask(volume: np.ndarray) -> np.ndarray:
    """Demo-only: bright-region heuristic (largest connected component)."""
    brain = volume > 0
    if not brain.any():
        return np.zeros(volume.shape, dtype=bool)
    threshold = np.percentile(volume[brain], 97)
    candidates = volume >= threshold
    labels, count = ndimage.label(candidates)
    if count == 0:
        return candidates
    sizes = ndimage.sum(candidates, labels, range(1, count + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    largest = ndimage.binary_closing(largest, iterations=2)
    largest = ndimage.binary_fill_holes(largest)
    return largest


# --------------------------------------------------------------------------
# Ground-truth masks shipped with the repository (masks_test/)
# --------------------------------------------------------------------------
import nibabel as nib  # noqa: E402  (kept next to the mask helpers)

MASKS_ROOT = Path(__file__).resolve().parents[2] / 'masks_test'
MIN_MASK_VOXELS = 500  # ignore degenerate segmentations (e.g. 141-voxel stubs)


def series_of(filename: str) -> str:
    """``3D_AX_T1_postcontrast_1.nii.gz`` -> ``3D_AX_T1_postcontrast``."""
    stem = Path(filename).name.replace('.nii.gz', '').replace('.nii', '')
    head, _, tail = stem.rpartition('_')
    return head if tail.isdigit() else stem


def find_dataset_mask(patient: str, moving_series: str) -> tuple:
    """
    Pick the best preop tumour mask for a patient from ``masks_test/``.

    Preference: sequence-matched mask (unless degenerate), then the largest
    mask (world-coordinate resampling makes any reference sequence usable).

    Returns ``(path, voxel_count)`` or ``(None, 0)``.
    """
    pre_dir = MASKS_ROOT / patient / 'preop'
    if not pre_dir.is_dir():
        return None, 0
    best_path, best_key = None, None
    for cand in sorted(pre_dir.glob('*_preop_tumor_*.nii*')):
        data = np.asanyarray(nib.load(str(cand)).dataobj) > 0
        voxels = int(data.sum())
        if voxels < MIN_MASK_VOXELS:
            continue
        ref = cand.name.split('ref-')[-1].replace('.nii.gz', '').replace('.nii', '')
        key = (1 if ref == moving_series else 0, voxels)
        if best_key is None or key > best_key:
            best_path, best_key = cand, key
    return best_path, (best_key[1] if best_key else 0)


def load_dataset_mask(path, moving_shape, moving_affine,
                      ref_image=None) -> np.ndarray:
    """Place a dataset mask on the moving image's grid (see ``mask_to_grid``)."""
    return mask_to_grid(path, moving_shape, moving_affine, ref_image)


def _enhancement_rank(image: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of masked voxels brighter than the median brain voxel."""
    if not mask.any():
        return 0.0
    image = np.asarray(image, dtype=np.float32)
    brain = image[image > np.percentile(image[image > 0], 25)]
    if brain.size == 0 or not np.isfinite(brain).any():
        return 0.0
    return float((image[mask] > np.median(brain)).mean())


def mask_to_grid(path, shape, affine, ref_image=None) -> np.ndarray:
    """
    Place a ``masks_test`` mask on the target image grid (order 0).

    The masks in ``masks_test`` disagree with the image affines about the
    anterior-posterior direction (images report ``(R,A,I)``, masks report
    ``(R,P,I)``), and the inconsistency differs per patient: for some masks only
    the stored affine is wrong (voxels were rasterized on the reference grid),
    for others the voxel array itself is A-P flipped while the stored world
    position is right.  Resampling with the stored affine therefore marks the
    wrong anatomy for a subset of masks.

    Resolution: build BOTH candidate placements — the stored-affine world
    mapping and the flip-corrected one (``stored @ YFLIP``) — and keep the one
    that covers brighter tissue on the reference image (tumour masks are drawn
    on contrast-enhanced scans, so the true placement scores higher).  Ties
    (within 0.05) keep the stored-affine placement, which is the world position
    documented in ``all_masks_index.csv``.  Without a reference image the
    stored-affine placement is returned unchanged (backwards compatible).
    """
    from nibabel.processing import resample_from_to

    src = nib.load(str(path))
    target = (tuple(shape), np.asarray(affine))
    stored = resample_from_to(src, target, order=0).get_fdata() > 0.5
    if ref_image is None:
        return stored

    flip = np.eye(4)
    flip[1, 1] = -1.0
    flip[1, 3] = src.shape[1] - 1
    corrected = nib.Nifti1Image(np.asanyarray(src.dataobj), src.affine @ flip)
    alt = resample_from_to(corrected, target, order=0).get_fdata() > 0.5

    if not stored.any() or not alt.any():
        return stored if stored.any() else alt
    if alt.sum() != stored.sum():
        # Order-0 world resampling must preserve the count; a mismatch means one
        # candidate fell partly off the grid — keep the fuller one.
        return stored if stored.sum() > alt.sum() else alt

    r_stored = _enhancement_rank(ref_image, stored)
    r_alt = _enhancement_rank(ref_image, alt)
    return alt if r_alt > r_stored + 0.05 else stored


def find_gt_mask(patient: str, fixed_series: str):
    """Best intraop tumour-residual mask (ground truth for 'where it is now')."""
    ino_dir = MASKS_ROOT / patient / 'intraop'
    if not ino_dir.is_dir():
        return None
    best, best_key = None, None
    for cand in sorted(ino_dir.glob('*_tumor_residual*.nii*')):
        voxels = int((np.asanyarray(nib.load(str(cand)).dataobj) > 0).sum())
        if voxels < MIN_MASK_VOXELS:
            continue
        ref = cand.name.split('ref-')[-1].replace('.nii.gz', '').replace('.nii', '')
        key = (1 if ref == fixed_series else 0, voxels)
        if best_key is None or key > best_key:
            best, best_key = cand, key
    return best


def segment(volume: np.ndarray, manual_mask: np.ndarray | None = None,
            dataset_mask: np.ndarray | None = None) -> dict:
    """
    Produce a binary tumour mask in the *moving* (preoperative) space.

    Engine priority: clinician upload > repository ground-truth mask >
    trained model > demo heuristic.

    Returns ``{'mask': bool ndarray, 'engine': str, 'detail': str}``.
    """
    if manual_mask is not None:
        mask = manual_mask.astype(bool)
        return {'mask': mask, 'engine': 'manual',
                'detail': 'Clinician-provided tumour mask (preoperative space).'}

    if dataset_mask is not None:
        mask = dataset_mask.astype(bool)
        return {'mask': mask, 'engine': 'dataset',
                'detail': 'Ground-truth tumour segmentation from masks_test/ '
                          '(world-resampled onto the preoperative grid).'}

    model = load_model()
    if model is not None:
        try:
            return {'mask': _predict(model, volume).astype(bool), 'engine': 'model',
                    'detail': f'Segmentation model loaded from {MODEL_PATH.name}.'}
        except Exception as exc:  # fall back so the demo never dead-ends
            pseudo = _pseudo_mask(volume)
            return {'mask': pseudo, 'engine': 'pseudo',
                    'detail': f'Model failed ({exc}); fell back to demo segmentation.'}

    mask = _pseudo_mask(volume)
    return {'mask': mask, 'engine': 'pseudo',
            'detail': 'No segmentation model at segmentation_model.pt yet - using the '
                      'intensity-based DEMO mask. Replace with the trained model later.'}
