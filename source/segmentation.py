"""

Segmentation visualization functions.

This module provides tools for visualizing bianry masks and multi-label
segmentations overlaid on anatomical volumes.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from typing import Optional, Tuple
from matplotlib.colors import ListedColormap

from .preprocessing import rescale_intensity_linear


def view_volume_with_mask_contours(volume: np.ndarray, mask: np.ndarray, axis: int = 0, thickness: int = 2, contour_color: Tuple[int, int, int] = (0, 255, 0)):
    """Interactive viewer that overlays mask contours on a 3D volume"""
    v = np.asarray(volume)
    m = np.asarray(mask)

    if v.shape != m.shape:
        raise ValueError(f"volume and mask must have same shape, got {v.shape} vs. {m.shape}")
    
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume and mask, got shape {v.shape}")
    
    if axis not in (0, 1, 2):
        raise ValueError(f"Axis must be 0, 1, or 2, got {axis}")
    
    # Normalize volume for visualization
    v = rescale_intensity_linear(v, 0.0, 255.0).astype(np.uint8)
    m = (m > 0).astype(np.uint8)
    
    v = np.moveaxis(v, axis, 0)
    m = np.moveaxis(m, axis, 0)
    n_slices = v.shape[0]

    def _show(SLICE: int):
        slice_img = v[SLICE]
        rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB)
        
        contours, _ = cv2.findContours(m[SLICE], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        overlay = rgb.copy()
        cv2.drawContours(overlay, contours, -1, contour_color, thickness)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title(f"Slice {SLICE}/{n_slices - 1} with mask contours")
        plt.axis("off")
        plt.show()

    interact(_show, SLICE=(0, n_slices - 1))


def view_volume_with_mask_overlay(volume: np.ndarray, mask: np.ndarray, axis: int = 0, alpha: float = 0.4, mask_cmap: str = "Reds"):
    """Interactive viewer with semi-transparent mask overlay"""
    v = np.asarray(volume)
    m = np.asarray(mask)
    
    if v.shape != m.shape:
        raise ValueError(f"volume and mask must have same shape")
    
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume")
    
    v = np.moveaxis(v, axis, 0)
    m = np.moveaxis(m, axis, 0)
    n_slices = v.shape[0]
    
    def _show(SLICE: int):
        plt.figure(figsize=(8, 8))
        plt.imshow(v[SLICE, :, :], cmap='gray')
        plt.imshow(m[SLICE, :, :], cmap=mask_cmap, alpha=alpha)
        plt.title(f"Slice {SLICE}/{n_slices - 1} with mask overlay")
        plt.axis("off")
        plt.show()
    
    interact(_show, SLICE=(0, n_slices - 1))


def view_multi_label_segmentation(volume: np.ndarray, segmentation: np.ndarray, axis: int = 0, alpha: float = 0.5, label_names: Optional[dict] = None):
    """Interactive viewer for multi-label segmentation overlays"""
    v = np.asarray(volume)
    seg = np.asarray(segmentation)
    
    if v.shape != seg.shape:
        raise ValueError("volume and segmentation must have same shape")
    
    v = np.moveaxis(v, axis, 0)
    seg = np.moveaxis(seg, axis, 0)
    n_slices = v.shape[0]
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Create custom colormap
    n_labels = len(unique_labels)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_labels))
    
    def _show(SLICE: int):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show volume
        ax.imshow(v[SLICE, :, :], cmap='gray')
        
        # Overlay segmentation
        seg_slice = seg[SLICE, :, :]
        masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
        
        im = ax.imshow(masked_seg, cmap=ListedColormap(colors), 
                      alpha=alpha, vmin=unique_labels.min(), 
                      vmax=unique_labels.max())
        
        ax.set_title(f"Slice {SLICE}/{n_slices - 1}")
        ax.axis("off")
        
        # Add legend if label names provided
        if label_names:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[i], label=label_names.get(label, f'Label {label}'))
                for i, label in enumerate(unique_labels)
            ]
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.show()
    
    interact(_show, SLICE=(0, n_slices - 1))


def compare_segmentations(volume: np.ndarray, seg1: np.ndarray, seg2: np.ndarray, axis: int = 0, labels: Tuple[str, str] = ("Seg 1", "Seg 2")):
    """Comapre two segmentations side-by-side with overlap visualization"""
    v = np.asarray(volume)
    s1 = np.asarray(seg1)
    s2 = np.asarray(seg2)
    
    if not (v.shape == s1.shape == s2.shape):
        raise ValueError("All inputs must have the same shape")
    
    v = np.moveaxis(v, axis, 0)
    s1 = np.moveaxis(s1, axis, 0)
    s2 = np.moveaxis(s2, axis, 0)
    n_slices = v.shape[0]
    
    def _show(SLICE: int):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # First segmentation
        axes[0].imshow(v[SLICE], cmap='gray')
        axes[0].imshow(s1[SLICE], cmap='Reds', alpha=0.4)
        axes[0].set_title(labels[0])
        axes[0].axis('off')
        
        # Second segmentation
        axes[1].imshow(v[SLICE], cmap='gray')
        axes[1].imshow(s2[SLICE], cmap='Blues', alpha=0.4)
        axes[1].set_title(labels[1])
        axes[1].axis('off')
        
        # Overlap visualization
        # Green = agreement, Red = only in seg1, Blue = only in seg2
        overlap = np.zeros((*v[SLICE].shape, 3), dtype=np.uint8)
        both = (s1[SLICE] > 0) & (s2[SLICE] > 0)
        only_1 = (s1[SLICE] > 0) & (s2[SLICE] == 0)
        only_2 = (s1[SLICE] == 0) & (s2[SLICE] > 0)
        
        overlap[both] = [0, 255, 0]      # Green for overlap
        overlap[only_1] = [255, 0, 0]    # Red for seg1 only
        overlap[only_2] = [0, 0, 255]    # Blue for seg2 only
        
        axes[2].imshow(v[SLICE], cmap='gray')
        axes[2].imshow(overlap, alpha=0.5)
        axes[2].set_title('Overlap (Green=Both, Red=Seg1, Blue=Seg2)')
        axes[2].axis('off')
        
        plt.suptitle(f"Slice {SLICE}/{n_slices - 1}")
        plt.tight_layout()
        plt.show()
    
    interact(_show, SLICE=(0, n_slices - 1))


def compute_dice_coefficient(seg1: np.ndarray, seg2: np.ndarray):
    """Compute Dice coefficient between two binary segmentations"""
    s1 = (seg1 > 0).astype(bool)
    s2 = (seg2 > 0).astype(bool)
    
    if s1.shape != s2.shape:
        raise ValueError("Segmentations must have same shape")
    
    intersection = np.sum(s1 & s2)
    sum_volumes = np.sum(s1) + np.sum(s2)
    
    if sum_volumes == 0:
        return 1.0  # Both empty
    
    return 2.0 * intersection / sum_volumes


def compute_iou(seg1: np.ndarray, seg2: np.ndarray):
    """Compute Intersection over Union (Jaccard index) between two segmentations"""
    s1 = (seg1 > 0).astype(bool)
    s2 = (seg2 > 0).astype(bool)
    
    if s1.shape != s2.shape:
        raise ValueError("Segmentations must have same shape")
    
    intersection = np.sum(s1 & s2)
    union = np.sum(s1 | s2)
    
    if union == 0:
        return 1.0  # Both empty
    
    return intersection / union

