"""Bundle the full case for review inside 3D Slicer (NIfTI + metrics + plan)."""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

README = """NeuroVista case bundle - how to review in 3D Slicer
=====================================================
1. Open 3D Slicer (3D Slicer 5.x, free: https://slicer.org).
2. Drag & drop these volumes onto the main window:
     01_fixed_intraop.nii.gz                      -> fixed (intraoperative) reference
     02_moving_preop_resampled_to_fixed.nii.gz    -> preoperative scan, affine-aligned
     03_moving_warped_to_fixed.nii.gz             -> preoperative scan AFTER registration
                                                    (i.e. anatomy corrected for brain shift)
3. Tumour masks (load as LabelMap or via Segment Editor -> Import):
     05_tumour_preop_mask.nii.gz                  -> tumour at initial planning time
     06_tumour_now_mask.nii.gz                    -> tumour location AFTER brain shift
4. 04_displacement_field_voxels.nii.gz is the 3-component deformation field
   (voxel units, X/Y/Z = axis 0/1/2 of the fixed grid).
5. metrics.json / updated_plan.txt contain the measured shift and the updated plan.
"""


def _nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))


def build_bundle(dest: Path, *, fixed, moving_resampled, warped, disp,
                 mask_pre, mask_now, metrics: dict, plan_text: str) -> Path:
    """Write a zip archive containing the whole case. Returns the zip path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='nv_bundle_', dir=str(dest.parent)))

    try:
        _nifti(workdir / '01_fixed_intraop.nii.gz', fixed.data, fixed.affine)
        _nifti(workdir / '02_moving_preop_resampled_to_fixed.nii.gz',
               moving_resampled.data, moving_resampled.affine)
        _nifti(workdir / '03_moving_warped_to_fixed.nii.gz', warped.data, warped.affine)
        _nifti(workdir / '04_displacement_field_voxels.nii.gz',
               np.moveaxis(disp, 0, -1).astype(np.float32), fixed.affine)
        _nifti(workdir / '05_tumour_preop_mask.nii.gz',
               mask_pre.data.astype(np.uint8), mask_pre.affine)
        _nifti(workdir / '06_tumour_now_mask.nii.gz',
               mask_now.data.astype(np.uint8), mask_now.affine)
        (workdir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (workdir / 'updated_plan.txt').write_text(plan_text or 'Plan not generated yet.')
        (workdir / 'README.txt').write_text(README)

        with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(workdir.iterdir()):
                zf.write(file, file.name)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return dest
