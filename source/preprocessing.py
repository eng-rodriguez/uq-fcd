"""
Preprocessing and intensity manipulation functions for medical imaging.

This module contains functions for normalizing, rescaling, and adjusting
image intensities for better visualization and analysis.
"""
import numpy as np
from typing import Optional


def rescale_intensity_linear(volume: np.ndarray, new_min: float = 0.0, new_max: float = 1.0, old_min: Optional[float] = None, old_max: Optional[float] = None):
    """Linearly rescale a Numpy array to a new intensity range"""
    arr = np.asarray(volume, dtype=np.float32)

    if old_min is None:
        old_min = float(np.min(arr))
    
    if old_max is None:
        old_max = float(np.max(arr))

    if old_max == old_min:
        return np.full_like(arr, fill_value=new_min, dtype=np.float32)
    
    scale = (new_max - new_min) / (old_max - old_min)
    out = new_min + scale * (arr - old_min)

    return out.astype(np.float32)


def window_level_adjustment(volume: np.ndarray, window: float, level: float):
    """Apply window/level (width/center) adjustment commonly used in medical imaging"""
    arr = np.asarray(volume, dtype=np.float32)
    min_val = level - window / 2
    max_val = level + window / 2
    
    arr_clipped = np.clip(arr, min_val, max_val)
    arr_normalized = (arr_clipped - min_val) / (max_val - min_val)
    
    return arr_normalized


def normalize_to_uint8(volume: np.ndarray):
    """Normalize volume to uint9 range [0, 255]"""
    return rescale_intensity_linear(volume, 0, 255).astype(np.uint8)


def percentile_normalization(volume: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
    """Normalize volume using percentile clipping to handle outliers"""
    arr = np.asarray(volume, dtype=np.float32)
    p_low = np.percentile(arr, lower_percentile)
    p_high = np.percentile(arr, upper_percentile)
    
    return rescale_intensity_linear(arr, 0.0, 1.0, old_min=p_low, old_max=p_high)


def z_score_normalization(volume: np.ndarray, clip_range: Optional[float]= None):
    """Normalize volume using z-score (zero mean, unit vaiance)"""
    arr = np.asarray(volume, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        return np.zeros_like(arr)
    
    normalized = (arr - mean) / std
    
    if clip_range is not None:
        normalized = np.clip(normalized, -clip_range, clip_range)
    
    return normalized


def histogram_matching(source: np.ndarray, reference: np.ndarray):
    """Match histogram of source volume to reference volume"""
    # Get the histograms
    source_values = source.flatten()
    reference_values = reference.flatten()
    
    # Compute CDFs
    source_sorted = np.sort(source_values)
    reference_sorted = np.sort(reference_values)
    
    # Interpolate to match histograms
    matched = np.interp(source_values, source_sorted, reference_sorted)
    
    return matched.reshape(source.shape).astype(source.dtype)
