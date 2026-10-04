# NeuroVista

## About

TODO: Add a short description of this project.

## Getting Started

TODO: Add setup and usage instructions.

## Vendored Dependencies

### VoxelMorph

This project vendors a snapshot of [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
(Unsupervised Learning for Image Registration) in the `voxelmorph/` directory.

- **Source:** https://github.com/voxelmorph/voxelmorph
- **Upstream commit:** `c4155e1baf04bc8f1774775b23a6027b58b37e00` (2026-09-11)
- **License:** Apache-2.0 (see `voxelmorph/LICENSE.md`)

The copy is unlinked from upstream and may be modified freely. To import it as a
Python package from the source checkout:

```bash
pip install -e voxelmorph
```

## Web application — surgical planning suite

`webapp/` is the full planning application:

1. **Upload** the intraoperative scan (fixed) and preoperative scan (moving), plus an
   optional preoperative tumour mask (`.nii` / `.nii.gz`).
2. **Pipeline** — tumour segmentation (pluggable), VoxelMorph registration with the
   trained `voxelmorph/final.pt` checkpoint, deformation-field extraction, and tumour
   relocation (preop → affine-aligned → post-shift positions, shift in mm, Jacobian QC).
3. **Visualization** — in-browser multi-plane slice viewer with tumour contours,
   centroid markers, deformation heatmap and checkerboard QC, metrics cards, a 3D
   rendering of the preoperative vs. current tumour, and a one-click **3D Slicer
   bundle** (NIfTI volumes, masks, 3-component displacement field, metrics, plan).
4. **Planning (Cline SDK)** — enter the initial preoperative plan; every coordinate is
   mapped through the measured deformation field and an updated navigation plan is
   generated (deterministic engine always works offline; set `ANTHROPIC_API_KEY` to
   enable Claude-authored narratives and free-form chat).

### Run

```bash
cd webapp
python -m uvicorn server:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Dependencies: `pip install fastapi uvicorn python-multipart` (all already installed in
this environment), plus the `voxelmorph` package from the previous section.

### Sample cases (one click)

The **Step 1** screen lists ready-to-run samples (Sample_1 … Sample_4) — real ReMIND
patients that have expert tumour masks in `masks_test/`. Clicking a card loads the scans,
attaches the ground-truth segmentation (`dataset` engine), runs segment → register →
relocate, and lands on the **Tumour relocation showcase**: a drag-to-compare wipe view
(initial plan vs. after-shift), a Δ-vector arrow, pulsing centroid, animated KPIs
(shift mm, volume, Dice vs. the intraop residual mask, NCC, Jacobian), the 3D surfaces
and the 3D Slicer bundle.

### Segmentation model (to be provided)

Place the trained model at **`webapp/segmentation_model.pt`** — the pipeline detects it
automatically (TorchScript `torch.jit.save`, or `torch.save` of a module that maps a
normalized `(1, 1, D, H, W)` volume to logits). Until then the UI clearly labels the
**demo engine** (intensity heuristic) or accepts a clinician-provided mask.
