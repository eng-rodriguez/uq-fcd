"""
Basic visualization fucntions for 3D medical imaging volumes.

This module provides interactive and static viewers for exploring
volumetric data along different axes and orientations.
"""
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from typing import Optional, Tuple, List

from .preprocessing import rescale_intensity_linear


def view_volume_slice(volume: np.ndarray, axis: int = 0, cmap: str = "gray"):
    """Interactive viewer for a 3D volume using ipywidgets"""
    vol = np.asarray(volume)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    
    if axis not in (0, 1, 2):
        raise ValueError(f"Axis must be 0, 1, or 2, got {axis}")
    
    vol = np.moveaxis(vol, axis, 0)
    n_slices = vol.shape[0]

    def _show(SLICE: int):
        plt.figure(figsize=(7,7))
        plt.imshow(vol[SLICE, :, :], cmap=cmap)
        plt.title(f"Slice {SLICE}/{n_slices - 1} (axis={axis})")
        plt.axis("off")
        plt.show()

    interact(_show, SLICE=(0, n_slices - 1))


def view_orthogonal_slices(volume: np.ndarray, slice_idx: Optional[Tuple[int, int, int]] = None, cmap: str = "gray", figsize: Tuple[int, int] = (15, 5)):
    """Display three orthogonal slices (axial, coronal, sagittal) of a 3D volume"""
    vol = np.asarray(volume)
    
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    
    if slice_idx is None:
        slice_idx = tuple(s // 2 for s in vol.shape)
    
    ax_idx, cor_idx, sag_idx = slice_idx
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Axial (z-axis)
    axes[0].imshow(vol[ax_idx, :, :], cmap=cmap, origin='lower')
    axes[0].set_title(f'Axial (slice {ax_idx}/{vol.shape[0]-1})')
    axes[0].axis('off')
    
    # Coronal (y-axis)
    axes[1].imshow(vol[:, cor_idx, :], cmap=cmap, origin='lower')
    axes[1].set_title(f'Coronal (slice {cor_idx}/{vol.shape[1]-1})')
    axes[1].axis('off')
    
    # Sagittal (x-axis)
    axes[2].imshow(vol[:, :, sag_idx], cmap=cmap, origin='lower')
    axes[2].set_title(f'Sagittal (slice {sag_idx}/{vol.shape[2]-1})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def interactive_orthogonal_viewer(volume: np.ndarray, cmap: str = "gray"):
    """Interactive orthogonal viewer with sliders for all three axes"""
    vol = np.asarray(volume)
    
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    
    def _show(Axial: int, Coronal: int, Sagittal: int):
        view_orthogonal_slices(vol, (Axial, Coronal, Sagittal), cmap=cmap)
    
    interact(
        _show,
        Axial=(0, vol.shape[0] - 1, 1),
        Coronal=(0, vol.shape[1] - 1, 1),
        Sagittal=(0, vol.shape[2] - 1, 1)
    )


def create_montage(volume: np.ndarray, axis: int = 0, n_slices: Optional[int] = None, cmap: str = "gray", ncols: int = 5, figsize: Tuple[int, int] = (15, 12)):
    """Create a montage (grid) of slices from a volume"""
    vol = np.moveaxis(volume, axis, 0)
    total_slices = vol.shape[0]
    
    if n_slices is None:
        indices = range(total_slices)
    else:
        indices = np.linspace(0, total_slices - 1, n_slices, dtype=int)
    
    n_display = len(indices)
    nrows = (n_display + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(vol[idx], cmap=cmap)
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_display, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
