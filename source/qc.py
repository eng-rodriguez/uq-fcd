"""
Quality Control (QC) and analysis functions.

This module provides tools for assessing image quality, detecting artifacts,
and analyzing intensity distributions.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def plot_intensity_histogram(volume: np.ndarray, bins: int = 100, exclude_zeros: bool = True, log_scale: bool = False, figsize: Tuple[int, int] = (10, 5)):
    """Plot intensity histogram of a volume with statistic overlay"""
    vol = np.asarray(volume).flatten()
    
    if exclude_zeros:
        vol = vol[vol != 0]
    
    plt.figure(figsize=figsize)
    plt.hist(vol, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title(f'Intensity Histogram {"(excluding zeros)" if exclude_zeros else ""}')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    # Add statistics text
    stats_text = (f'Mean: {np.mean(vol):.2f}\n'
                 f'Std: {np.std(vol):.2f}\n'
                 f'Min: {np.min(vol):.2f}\n'
                 f'Max: {np.max(vol):.2f}')
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_slice_intensity_profiles(volume: np.ndarray, axis: int = 0, aggregation: str = "mean", figsize: Tuple[int, int] = (12, 5)):
    """Plot itensity profiles across slices"""
    vol = np.moveaxis(volume, axis, 0)
    n_slices = vol.shape[0]
    
    agg_funcs = {
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'min': np.min,
        'std': np.std
    }
    
    if aggregation not in agg_funcs:
        raise ValueError(f"aggregation must be one of {list(agg_funcs.keys())}")
    
    func = agg_funcs[aggregation]
    values = [func(vol[i]) for i in range(n_slices)]
    
    plt.figure(figsize=figsize)
    plt.plot(range(n_slices), values, marker='o', linestyle='-', markersize=3)
    plt.xlabel(f'Slice Index (axis={axis})')
    plt.ylabel(f'{aggregation.capitalize()} Intensity')
    plt.title(f'Intensity {aggregation.capitalize()} per Slice')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def check_image_orientation(volume: np.ndarray, expected_shape: Tuple[int, int, int] = None) -> dict:
    """Check if image has expected orientation and dimensions"""
    actual_shape = volume.shape
    
    result = {
        'actual_shape': actual_shape,
        'ndim': volume.ndim,
        'is_3d': volume.ndim == 3,
    }
    
    if expected_shape is not None:
        result['expected_shape'] = expected_shape
        result['matches_expected'] = actual_shape == expected_shape
        result['shape_difference'] = tuple(a - e for a, e in zip(actual_shape, expected_shape))
    
    return result


def detect_motion_artifacts(volume: np.ndarray, axis: int = 0, threshold_std: float = 3.0) -> dict:
    """Detect potential motion artifacts by analyzing slice-wise intensity variation"""
    vol = np.moveaxis(volume, axis, 0)
    n_slices = vol.shape[0]
    
    # Compute mean intensity per slice
    means = np.array([np.mean(vol[i]) for i in range(n_slices)])
    
    # Compute differences between consecutive slices
    diffs = np.diff(means)
    
    # Find outliers
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    outliers = np.abs(diffs - mean_diff) > (threshold_std * std_diff)
    suspicious_slices = np.where(outliers)[0].tolist()
    
    return {
        'suspicious_slices': suspicious_slices,
        'n_suspicious': len(suspicious_slices),
        'mean_intensity_change': float(mean_diff),
        'std_intensity_change': float(std_diff),
        'max_intensity_jump': float(np.max(np.abs(diffs))),
    }


def compute_snr(volume: np.ndarray, signal_mask: np.ndarray, noise_mask: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio (SNR)"""
    signal = volume[signal_mask > 0]
    noise = volume[noise_mask > 0]
    
    if len(signal) == 0 or len(noise) == 0:
        raise ValueError("Signal or noise mask is empty")
    
    snr = np.mean(signal) / np.std(noise)

    return float(snr)


def compute_cnr(volume: np.ndarray, tissue1_mask: np.ndarray, tissue2_mask: np.ndarray, noise_mask: np.ndarray):
    """Compute Contrast-to-Noise Ratio (CNR) between two tissue types"""
    tissue1 = volume[tissue1_mask > 0]
    tissue2 = volume[tissue2_mask > 0]
    noise = volume[noise_mask > 0]
    
    if len(tissue1) == 0 or len(tissue2) == 0 or len(noise) == 0:
        raise ValueError("One or more masks are empty")
    
    contrast = np.abs(np.mean(tissue1) - np.mean(tissue2))
    cnr = contrast / np.std(noise)

    return float(cnr)
