"""
Comparison and difference visualization functions.

This module provides tools for comparing multiple volumes side-by-side
and computing difference maps.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from ipywidgets import interact


def compare_volume_slices(volume_before: np.ndarray, volume_after: np.ndarray, axis: int = 0, cmap: str = "gray", title_before: str = "Before", title_after: str = "After"):
    """Interactive side-by-side comparison of two 3D volumes"""
    vol_a = np.asarray(volume_before)
    vol_b = np.asarray(volume_after)

    if vol_a.shape != vol_b.shape:
        raise ValueError(f"Volumes must have same shape, got {vol_a.shape} vs {vol_b.shape}")
    
    if vol_a.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got shape {vol_a.shape}")
    
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2 got {axis}")
    
    vol_a = np.moveaxis(vol_a, axis, 0)
    vol_b = np.moveaxis(vol_b, axis, 0)
    n_slices = vol_a.shape[0]

    def _show(SLICE: int):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        im1 = ax1.imshow(vol_a[SLICE, :, :], cmap=cmap)
        ax1.set_title(f"{title_before} - Slice {SLICE}/{n_slices - 1}", fontsize=14)
        ax1.axis("off")
        
        im2 = ax2.imshow(vol_b[SLICE, :, :], cmap=cmap)
        ax2.set_title(f"{title_after} - Slice {SLICE}/{n_slices - 1}", fontsize=14)
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    interact(_show, SLICE=(0, n_slices - 1))


def compare_multiple_volumes(volumes: List[np.ndarray], titles: List[str], axis: int = 0, cmap: str = "gray", ncols: int = 3, figsize: Tuple[int, int] = (15, 5)):
    """Interactive comparison of multiple volumes in a grid layout"""
    if len(volumes) != len(titles):
        raise ValueError("Number of volumes must match number of titles")
    
    # Verify all volumes have same shape
    base_shape = volumes[0].shape
    for i, vol in enumerate(volumes):
        if vol.shape != base_shape:
            raise ValueError(f"Volume {i} has shape {vol.shape}, expected {base_shape}")
    
    # Move axis to front for all volumes
    volumes = [np.moveaxis(vol, axis, 0) for vol in volumes]
    n_slices = volumes[0].shape[0]
    n_volumes = len(volumes)
    nrows = (n_volumes + ncols - 1) // ncols
    
    def _show(SLICE: int):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for i, (vol, title) in enumerate(zip(volumes, titles)):
            im = axes[i].imshow(vol[SLICE, :, :], cmap=cmap)
            axes[i].set_title(f"{title}\nSlice {SLICE}/{n_slices-1}")
            axes[i].axis("off")
        
        # Hide unused subplots
        for i in range(n_volumes, len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.show()
    
    interact(_show, SLICE=(0, n_slices - 1))
