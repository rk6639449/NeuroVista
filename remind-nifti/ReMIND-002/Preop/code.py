import os
import nibabel as nib

folder = "/Users/harshitabharti/Desktop/remind-nifti/ReMIND-002/Preop"

for file in os.listdir(folder):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        filepath = os.path.join(folder, file)

        img = nib.load(filepath)

        print(f"File: {file}")
        print(f"Image size: {img.shape}")
        print(f"Voxel size: {img.header.get_zooms()}")
        print()