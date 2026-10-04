import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

import voxelmorph as vxm  # nopep8


def preprocess_volume(
    nii_path: str,
    normalize: bool = True,
    target_shape: tuple[int, int, int] = (160, 192, 160)
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int, int], tuple[float, float], list[int]]:
    """Loads a NIfTI volume, applies intensity normalization, pads/crops to match training dimensions,

    and returns preprocessed tensor and spatial metadata.
    """
    nii = nib.load(nii_path)
    affine = nii.affine
    orig_shape = nii.shape[:3]
    data = nii.get_fdata().astype(np.float32)

    # 1. Track Intensity Range & Normalize
    min_val, max_val = float(data.min()), float(data.max())
    if normalize and max_val > min_val:
        data = (data - min_val) / (max_val - min_val)

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)

    # 2. Calculate Symmetric Padding or Cropping
    curr_shape = tensor.shape[2:]
    pad_dims = []
    pad_offsets = []

    for curr, target in zip(reversed(curr_shape), reversed(target_shape)):
        diff = target - curr
        if diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_dims.extend([pad_before, pad_after])
            pad_offsets.append(pad_before)
        else:
            pad_dims.extend([0, 0])
            pad_offsets.append(0)

    if any(p > 0 for p in pad_dims):
        tensor = F.pad(tensor, pad_dims, mode='constant', value=0)

    # Crop along center if larger than target shape
    _, _, h, w, d = tensor.shape
    th, tw, td = target_shape
    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    sd = max((d - td) // 2, 0)

    tensor = tensor[:, :, sh:sh + th, sw:sw + tw, sd:sd + td]
    
    # Store spatial offsets for postprocessing (reverse order back to H, W, D)
    offsets = [pad_offsets[2], pad_offsets[1], pad_offsets[0], sh, sw, sd]
    
    return tensor, affine, orig_shape, (min_val, max_val), offsets


def postprocess_volume(
    tensor: torch.Tensor,
    orig_shape: tuple[int, int, int],
    offsets: list[int],
    intensity_range: tuple[float, float] = None,
    target_shape: tuple[int, int, int] = (160, 192, 160)
) -> np.ndarray:
    """Restores the output image or warp field tensor to its original input shape and intensity scale."""
    data = tensor.detach().cpu().squeeze().numpy()

    is_warp = data.ndim == 4 and data.shape[0] == 3
    if is_warp:
        data = np.moveaxis(data, 0, -1)

    pad_h, pad_w, pad_d, sh, sw, sd = offsets

    # 1. Re-scale intensity back to original range if image
    if not is_warp and intensity_range is not None:
        min_val, max_val = intensity_range
        data = data * (max_val - min_val) + min_val

    # 2. Re-pad if cropped during preprocessing
    curr_h, curr_w, curr_d = data.shape[:3]
    if sh > 0 or sw > 0 or sd > 0:
        pad_width = [(sh, sh), (sw, sw), (sd, sd)]
        if is_warp:
            pad_width.append((0, 0))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)

    # 3. Un-pad to original volume dimensions
    h_end = pad_h + orig_shape[0]
    w_end = pad_w + orig_shape[1]
    d_end = pad_d + orig_shape[2]

    if is_warp:
        data = data[pad_h:h_end, pad_w:w_end, pad_d:d_end, :]
    else:
        data = data[pad_h:h_end, pad_w:w_end, pad_d:d_end]

    return data


def main():
    parser = argparse.ArgumentParser(description='VoxelMorph inference for ReMIND dataset')
    parser.add_argument('--moving', required=True, help='moving image (Preop) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (Intraop) filename')
    parser.add_argument('--moved', required=True, help='warped image output filename')
    parser.add_argument('--model', required=True, help='trained pytorch model (.pt) path')
    parser.add_argument('--warp', help='output warp deformation field filename')
    parser.add_argument('-g', '--gpu', help='GPU ID. If omitted, CPU is used')
    args = parser.parse_args()

    # Device Handling
    if args.gpu and (args.gpu != '-1'):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print(f'Using device: {device}')

    # 1. Load Preprocessed Input Images & Spatial Metadata
    moving_tensor, moving_affine, orig_shape_m, range_m, offsets_m = preprocess_volume(args.moving)
    fixed_tensor, fixed_affine, orig_shape_f, range_f, offsets_f = preprocess_volume(args.fixed)

    input_moving = moving_tensor.to(device).float()
    input_fixed = fixed_tensor.to(device).float()

    # 2. Instantiate and Load the Trained VxmPairwise Model
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)

    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Predict Displacement Field and Warped Image
    with torch.no_grad():
        displacement, warped_source = model(
            input_moving,
            input_fixed,
            return_warped_source=True,
            return_field_type='displacement'
        )

    # 4. Save Warped Moving Image (with moving_affine header)
    if args.moved:
        moved_np = postprocess_volume(
            warped_source, 
            orig_shape_m, 
            offsets_m, 
            intensity_range=range_m
        )
        vxm.py.utils.save_volfile(moved_np, args.moved, moving_affine)
        print(f'Warped image saved to: {args.moved}')

    # 5. Save Deformation Field (with fixed_affine header)
    if args.warp:
        warp_np = postprocess_volume(
            displacement, 
            orig_shape_f, 
            offsets_f
        )
        vxm.py.utils.save_volfile(warp_np, args.warp, fixed_affine)
        print(f'Deformation field saved to: {args.warp}')


if __name__ == '__main__':
    main()