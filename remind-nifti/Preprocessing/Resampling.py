#!/usr/bin/env python3
"""
VoxelMorph Preprocessing Pipeline for the ReMIND Brain-Shift Dataset
======================================================================

WHY RESAMPLING IS REQUIRED BEFORE VOXELMORPH
----------------------------------------------
VoxelMorph is a learning-based deformable registration framework that expects
its moving and fixed input volumes to already share the same array shape
(D, H, W) and voxel grid (i.e. the same affine / voxel spacing). Internally,
VoxelMorph predicts a dense deformation field defined on a single fixed
sampling grid and warps the moving image onto that grid using a spatial
transformer. If the moving and fixed volumes have different shapes or voxel
spacings, the network has no well-defined coordinate system to predict a
per-voxel displacement field over, and simple stacking/concatenation of the
two volumes (which VoxelMorph's UNet backbone requires) becomes impossible.
Therefore, before any VoxelMorph training or inference, all image pairs must
be placed on a common voxel grid. This is a purely geometric preprocessing
step, independent of the (much harder) learned deformable registration that
VoxelMorph itself will later perform.

DIFFERENCE BETWEEN RESAMPLING AND DEFORMABLE REGISTRATION
------------------------------------------------------------
- Resampling (this script): Uses the *existing* affine transforms already
  encoded in each NIfTI header to re-grid one image's voxel data onto
  another image's grid. It only accounts for global affine differences
  (translation, rotation, scaling, shearing) that are already known from
  the scanner/acquisition geometry. It does NOT try to compensate for
  anatomical differences between the two images (e.g. brain shift during
  surgery). No optimization, no similarity metric, no deformation field is
  computed. It is a one-shot, deterministic interpolation operation.

- Deformable registration (VoxelMorph's job, NOT done here): Estimates a
  dense, non-linear deformation field that warps the moving image so that
  its anatomy spatially aligns with the fixed image's anatomy, compensating
  for real tissue deformation (e.g. intraoperative brain shift). This
  requires an optimization process (classical) or a trained neural network
  (VoxelMorph) and a similarity loss (e.g. NCC, MSE, MI).

In short: resampling fixes the *grid*; deformable registration fixes the
*anatomy*. This script only performs the former.

WHY THE FIXED IMAGE DEFINES THE TARGET GRID
-----------------------------------------------
By convention (and by this pipeline's requirement), the Intraop image is
the "fixed" image and the Preop image is the "moving" image. The fixed
image's shape, voxel spacing and affine define the reference coordinate
system that the network will later learn to warp the moving image into.
Resampling the moving (Preop) image onto the fixed (Intraop) image's grid
ensures that both volumes occupy identical array shapes and physical voxel
coordinates, which is the exact input format VoxelMorph expects.

USAGE
-----
    python voxelmorph_preprocessing.py [--dataset-root dataset/Dataset]
                                        [--output-csv resampling_summary.csv]
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NIFTI_PATTERNS = ("*.nii.gz", "*.nii")
PREOP_DIRNAME = "Preop"
INTRAOP_DIRNAME = "Intraop"
RESAMPLED_FILENAME = "resampled_to_fixed.nii.gz"
DEFAULT_CSV_NAME = "resampling_summary.csv"


# --------------------------------------------------------------------------- #
# Step 1: Locate patient files
# --------------------------------------------------------------------------- #
def find_patient_files(patient_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate exactly one Preop (moving) and one Intraop (fixed) NIfTI file
    inside a single patient's directory.

    Parameters
    ----------
    patient_dir : Path
        Path to a single patient folder, e.g. dataset/Dataset/ReMIND-001

    Returns
    -------
    (moving_path, fixed_path) : Tuple[Optional[Path], Optional[Path]]
        Paths to the Preop and Intraop NIfTI files. Either may be None if
        not found, which the caller must handle as an error case.
    """
    preop_dir = patient_dir / PREOP_DIRNAME
    intraop_dir = patient_dir / INTRAOP_DIRNAME

    moving_path = _find_first_nifti(preop_dir)
    fixed_path = _find_first_nifti(intraop_dir)

    return moving_path, fixed_path


def _find_first_nifti(folder: Path) -> Optional[Path]:
    """
    Return the first NIfTI file (*.nii.gz or *.nii) found in `folder`,
    excluding any previously-generated resampled output files, or None if
    the folder does not exist or contains no NIfTI files.
    """
    if not folder.exists() or not folder.is_dir():
        return None

    candidates: List[Path] = []
    for pattern in NIFTI_PATTERNS:
        candidates.extend(sorted(folder.glob(pattern)))

    # Exclude any resampled output from a previous run so re-running the
    # script doesn't accidentally pick up its own generated file.
    candidates = [c for c in candidates if c.name != RESAMPLED_FILENAME]

    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- #
# Step 2: Extract metadata from loaded NIfTI images
# --------------------------------------------------------------------------- #
def extract_metadata(img: nib.Nifti1Image) -> Dict[str, object]:
    """
    Extract shape, voxel spacing and affine determinant from a loaded
    NIfTI image.

    Parameters
    ----------
    img : nib.Nifti1Image
        A loaded nibabel image object.

    Returns
    -------
    dict with keys: shape, spacing, affine_det
    """
    shape = img.shape[:3]  # (D, H, W) -- ignore any extra 4th dim if present
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    affine_det = float(np.linalg.det(img.affine))

    return {
        "shape": shape,
        "spacing": spacing,
        "affine_det": affine_det,
    }


# --------------------------------------------------------------------------- #
# Step 3: Resample moving image onto fixed image grid (geometric only)
# --------------------------------------------------------------------------- #
def resample_moving_to_fixed(
    moving_img: nib.Nifti1Image, fixed_img: nib.Nifti1Image
) -> nib.Nifti1Image:
    """
    Resample the moving (Preop) image onto the fixed (Intraop) image's
    voxel grid using pure geometric resampling (linear interpolation).

    This uses nibabel.processing.resample_from_to(), which:
      - Takes the target shape and affine from `fixed_img`.
      - Maps each target voxel back into moving-image physical space using
        the two images' affine matrices (moving_affine^-1 @ fixed_affine).
      - Interpolates the moving image's intensity values at those
        locations (order=1 -> trilinear interpolation).

    IMPORTANT: This performs NO registration and estimates NO deformation
    field. It only accounts for the affine (linear) transforms already
    stored in the NIfTI headers. Orientation is preserved because the
    affine matrices (which encode orientation, spacing, and position) are
    used directly.

    Parameters
    ----------
    moving_img : nib.Nifti1Image
        The Preop (moving) image to be resampled.
    fixed_img : nib.Nifti1Image
        The Intraop (fixed) image whose grid defines the resampling target.

    Returns
    -------
    nib.Nifti1Image
        The moving image resampled onto the fixed image's grid.
    """
    # resample_from_to expects (image_to_resample, (target_shape, target_affine))
    target_grid = (fixed_img.shape[:3], fixed_img.affine)

    resampled_img = resample_from_to(
        moving_img,
        target_grid,
        order=1,  # linear interpolation
    )

    return resampled_img


# --------------------------------------------------------------------------- #
# Step 4: Process a single patient end-to-end
# --------------------------------------------------------------------------- #
def process_patient(patient_dir: Path) -> Dict[str, object]:
    """
    Run the full preprocessing pipeline for a single patient:
      1. Locate Preop (moving) and Intraop (fixed) NIfTI files.
      2. Load both images with nibabel.
      3. Extract metadata (shape, spacing, affine determinant) for both.
      4. Resample the moving image onto the fixed image's grid.
      5. Save the resampled image.
      6. Extract metadata for the resampled result and compare to fixed.

    Any failure at any stage is caught and reported; this function never
    raises -- it always returns a result dict with a "status" field of
    either "success" or "failed", so that the caller can continue
    processing remaining patients.

    Parameters
    ----------
    patient_dir : Path
        Path to a single patient folder, e.g. dataset/Dataset/ReMIND-001

    Returns
    -------
    dict
        A row of results suitable for inclusion in the summary CSV, plus
        a "status" and optional "error" key.
    """
    patient_id = patient_dir.name
    print(f"Processing {patient_id} ...")

    row: Dict[str, object] = {
        "PatientID": patient_id,
        "MovingPath": None,
        "FixedPath": None,
        "MovingShape": None,
        "FixedShape": None,
        "MovingSpacing": None,
        "FixedSpacing": None,
        "ResampledShape": None,
        "ResampledSpacing": None,
        "ShapeMatch": None,
        "MovingAffineDeterminant": None,
        "FixedAffineDeterminant": None,
        "ResampledFile": None,
        "status": "failed",
        "error": None,
    }

    try:
        # --- 1. Locate files -------------------------------------------------
        moving_path, fixed_path = find_patient_files(patient_dir)

        if moving_path is None:
            raise FileNotFoundError(
                f"No Preop NIfTI file found for {patient_id} "
                f"(missing folder or missing file)."
            )
        if fixed_path is None:
            raise FileNotFoundError(
                f"No Intraop NIfTI file found for {patient_id} "
                f"(missing folder or missing file)."
            )

        row["MovingPath"] = str(moving_path)
        row["FixedPath"] = str(fixed_path)

        # --- 2. Load images (nibabel loads lazily; force a read to catch
        #        corrupt files early via .get_fdata() / .dataobj access) ----
        try:
            moving_img = nib.load(str(moving_path))
            moving_img.affine  # touch to force header validation
        except Exception as exc:
            raise IOError(f"Corrupt or unreadable Preop NIfTI file: {exc}") from exc

        try:
            fixed_img = nib.load(str(fixed_path))
            fixed_img.affine
        except Exception as exc:
            raise IOError(f"Corrupt or unreadable Intraop NIfTI file: {exc}") from exc

        # --- 3. Extract metadata ---------------------------------------------
        moving_meta = extract_metadata(moving_img)
        fixed_meta = extract_metadata(fixed_img)

        row["MovingShape"] = moving_meta["shape"]
        row["FixedShape"] = fixed_meta["shape"]
        row["MovingSpacing"] = moving_meta["spacing"]
        row["FixedSpacing"] = fixed_meta["spacing"]
        row["MovingAffineDeterminant"] = moving_meta["affine_det"]
        row["FixedAffineDeterminant"] = fixed_meta["affine_det"]

        # --- 4. Resample moving -> fixed grid --------------------------------
        resampled_img = resample_moving_to_fixed(moving_img, fixed_img)

        # --- 5. Save resampled image ------------------------------------------
        preop_dir = patient_dir / PREOP_DIRNAME
        preop_dir.mkdir(parents=True, exist_ok=True)  # should already exist
        resampled_path = preop_dir / RESAMPLED_FILENAME
        nib.save(resampled_img, str(resampled_path))
        row["ResampledFile"] = str(resampled_path)

        # --- 6. Post-resampling metadata & sanity checks -----------------------
        resampled_meta = extract_metadata(resampled_img)
        row["ResampledShape"] = resampled_meta["shape"]
        row["ResampledSpacing"] = resampled_meta["spacing"]

        shape_match = tuple(resampled_meta["shape"]) == tuple(fixed_meta["shape"])
        row["ShapeMatch"] = bool(shape_match)

        row["status"] = "success"

    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        print(f"  [ERROR] {patient_id}: {exc}")
        # Uncomment for full traceback during debugging:
        # traceback.print_exc()
    
    run_voxelmorph_registration(moving_path, resampled_path,)

    return row


# --------------------------------------------------------------------------- #
# Step 5: Discover patients and orchestrate the full pipeline
# --------------------------------------------------------------------------- #
def discover_patient_dirs(dataset_root: Path) -> List[Path]:
    """
    Recursively find all patient folders directly under `dataset_root`.
    A patient folder is any immediate subdirectory (e.g. ReMIND-001).
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoxelMorph preprocessing pipeline for the ReMIND dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset/Dataset/remind-nifti",
        help="Path to the dataset root containing per-patient folders "
        "(default: dataset/Dataset)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default='Resampled_results',
        help=f"Path to the output summary CSV (default: {DEFAULT_CSV_NAME})",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    try:
        patient_dirs = discover_patient_dirs(dataset_root)
    except FileNotFoundError as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)

    total_found = len(patient_dirs)
    results: List[Dict[str, object]] = []

    for patient_dir in patient_dirs:
        result = process_patient(patient_dir)
        results.append(result)

import subprocess


def run_voxelmorph_registration(
    moving_path,
    fixed_path,
    model_path,
    moved_output_path,
    warp_output_path=None,
):
    """
    Run VoxelMorph registration for a single moving/fixed pair.

    Parameters
    ----------
    moving_path : str or Path
        PreOp MRI
    fixed_path : str or Path
        IntraOp MRI
    model_path : str or Path
        Trained VoxelMorph model
    moved_output_path : str or Path
        Output warped image
    warp_output_path : str or Path, optional
        Output deformation field
    """

    cmd = [
        "python",
        "scripts/tf/register.py",
        "--moving", str(moving_path),
        "--fixed", str(fixed_path),
        "--moved", str(moved_output_path),
        "--model", str(model_path),
    ]

    if warp_output_path is not None:
        cmd.extend([
            "--save-warp",
            str(warp_output_path)
        ])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"VoxelMorph failed:\n{result.stderr}"
        )

    return result.stdout


if __name__ == "__main__":
    main()