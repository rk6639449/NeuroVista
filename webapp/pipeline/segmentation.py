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


def segment(volume: np.ndarray, manual_mask: np.ndarray | None = None) -> dict:
    """
    Produce a binary tumour mask in the *moving* (preoperative) space.

    Returns ``{'mask': bool ndarray, 'engine': str, 'detail': str}``.
    """
    if manual_mask is not None:
        mask = manual_mask.astype(bool)
        return {'mask': mask, 'engine': 'manual',
                'detail': 'Clinician-provided tumour mask (preoperative space).'}

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
