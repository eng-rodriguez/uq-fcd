"""
File I/O operations for medical imaging data.

This module handles loading, saving, and metadata extraction for NIfTI and SITK formats.
"""
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Union, Tuple


def load_nifti_volume(nii_path: Union[str, Path]):
    """Load a NIfTI file and return both the data array and the image object"""
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    return data, img


def save_nifti_volume(volume: np.ndarray, reference_img: nib.Nifti1Image, output_path: Union[str, Path]):
    """Save a volume as NIfTI using a reference image for affine/header info"""
    new_img = nib.Nifti2Image(volume, reference_img.affine, reference_img.header)
    nib.save(new_img, str(output_path))
    print(f"Saved volume to {output_path}")


def load_sitk_volume(path: Union[str, Path]):
    """Load a volume using SimpleITK and return both array and object"""
    sitk_img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(sitk_img)
    return data, sitk_img


def save_sitk_volume(volume: np.ndarray, reference_img: sitk.Image, output_path: Union[str, Path]):
    """Save volume using SimpleITK with reference metadata"""
    new_img = sitk.GetImageFromArray(volume)
    new_img.CopyInformation(reference_img)
    sitk.WriteImage(new_img, str(output_path))
    print(f"Saved volume to {output_path}")


def print_nifti_info(nii_path: Union[str, Path]):
    """Print comprehensive metadata for a NIfTI file using nibabel"""
    img = nib.load(str(nii_path))
    header = img.header
    
    print(f"File: {nii_path}")
    print(f"Data shape: {img.shape}")
    print(f"Data type: {img.get_data_dtype()}")
    print(f"Voxel dimensions (mm): {header.get_zooms()}")
    print(f"Affine matrix:\n{img.affine}")
    print(f"Orientation: {nib.aff2axcodes(img.affine)}")
    
    # Additional header info if available
    if hasattr(header, 'get_xyzt_units'):
        spatial_unit, temporal_unit = header.get_xyzt_units()
        print(f"Spatial units: {spatial_unit}")
        if temporal_unit != 'unknown':
            print(f"Temporal units: {temporal_unit}")


def print_sitk_image_info(img: sitk.Image):
    """Print basic metadata for SimpleITK image"""
    pixel_type = img.GetPixelIDTypeAsString()
    origin = img.GetOrigin()
    size = img.GetSize()
    spacing = img.GetSpacing()
    direction = img.GetDirection()

    info = {
        "Pixel Type": pixel_type,
        "Dimensions (x, y, z)": size,
        "Spacing (mm)": spacing,
        "Origin": origin, 
        "Direction": direction
    }

    for k, v in info.items():
        print(f"{k}: {v}")


def append_nifti_suffix(filename: Union[str, Path], suffix: str):
    """Append a suffix to a NIfTI filename, preserving extensions"""
    path = Path(filename)
    suffix = suffix.strip("_")

    if path.name.endswith(".nii.gz"):
        stem = path.name[:-7]
        new_name = f"{stem}_{suffix}.nii.gz"
    elif path.suffix == ".nii":
        stem = path.stem
        new_name = f"{stem}_{suffix}.nii"
    else:
        raise ValueError(f"Unsupported NIfTI extension for '{filename}'")
    
    return str(path.with_name(new_name))
