import os
import csv
import nibabel as nib


def extract_mri_sizes(root_folder, output_file="mri_sizes.csv"):
    results = []

    # Go through everything inside remind-nifti
    for patient_folder in os.listdir(root_folder):

        patient_path = os.path.join(root_folder, patient_folder)

        # Only process folders named ReMIND-XXX
        if not os.path.isdir(patient_path):
            continue

        if not patient_folder.startswith("ReMIND-"):
            continue

        # Look for intraop and preop
        for phase in ["intraop", "preop"]:

            phase_path = os.path.join(patient_path, phase)

            if not os.path.isdir(phase_path):
                continue

            # Find all NIfTI files
            for root, dirs, files in os.walk(phase_path):

                for filename in files:

                    if not (filename.endswith(".nii") or
                            filename.endswith(".nii.gz")):
                        continue

                    filepath = os.path.join(root, filename)

                    try:
                        img = nib.load(filepath)

                        # Image dimensions
                        shape = img.shape

                        # Physical voxel dimensions in mm
                        voxel_size = img.header.get_zooms()

                        results.append({
                            "Patient": patient_folder,
                            "Phase": phase,
                            "Filename": filename,
                            "Size_X": shape[0],
                            "Size_Y": shape[1],
                            "Size_Z": shape[2] if len(shape) > 2 else "",
                            "Voxel_X_mm": voxel_size[0],
                            "Voxel_Y_mm": voxel_size[1],
                            "Voxel_Z_mm": voxel_size[2] if len(voxel_size) > 2 else "",
                            "Full_Path": filepath
                        })

                        print(
                            f"{patient_folder} | {phase} | "
                            f"{filename} | Shape: {shape}"
                        )

                    except Exception as e:
                        print(f"ERROR: {filepath}")
                        print(e)

    # Save results
    with open(output_file, "w", newline="") as f:

        fieldnames = [
            "Patient",
            "Phase",
            "Filename",
            "Size_X",
            "Size_Y",
            "Size_Z",
            "Voxel_X_mm",
            "Voxel_Y_mm",
            "Voxel_Z_mm",
            "Full_Path"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print("\n--------------------------------")
    print(f"Done!")
    print(f"Total MRI files found: {len(results)}")
    print(f"Saved to: {output_file}")
    print("--------------------------------")


# ==========================================================
# RUN
# ==========================================================

root_folder = "/Users/harshitabharti/Desktop/remind-nifti"

extract_mri_sizes(
    root_folder,
    "/Users/harshitabharti/Desktop/remind-nifti/mri_sizes.csv"
)