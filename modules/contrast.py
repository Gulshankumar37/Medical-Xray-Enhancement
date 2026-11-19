"""Linear and nonlinear gray-level adjustments."""

import cv2
import numpy as np

from .preprocessing import check_grayscale


def _apply_curve(image: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Look up each pixel in a table with one entry per possible input level."""
    lookup = np.rint(curve.ravel()).astype(image.dtype)
    if image.dtype == np.uint8:
        return cv2.LUT(image, lookup)
    return lookup[image]


def contrast_stretching(
    image: np.ndarray, low_percentile: float = 1, high_percentile: float = 99
) -> np.ndarray:
    """Map the selected percentiles to black and white, clipping values outside them."""
    check_grayscale(image)
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Choose two different percentiles, with the lower value first.")
    low, high = np.percentile(image, [low_percentile, high_percentile])
    if high == low:
        return image.copy()
    peak = int(np.iinfo(image.dtype).max)
    # Clip before scaling. Even a one-level window in a 16-bit scan stays accurate.
    levels = np.clip(np.arange(peak + 1, dtype=np.float64), low, high)
    curve = cv2.normalize(levels, None, 0, peak, cv2.NORM_MINMAX)
    return _apply_curve(image, curve)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply output = peak * (input / peak) ** gamma. Values below one brighten."""
    check_grayscale(image)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be a positive number.")
    peak = int(np.iinfo(image.dtype).max)
    ramp = np.linspace(0, 1, peak + 1, dtype=np.float64)
    curve = cv2.pow(ramp, gamma) * peak
    return _apply_curve(image, curve)
