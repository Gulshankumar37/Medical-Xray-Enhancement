"""Edge enhancement with signed intermediate values and saturating output."""

import cv2
import numpy as np

from .preprocessing import check_grayscale


def _check_strength(strength: float) -> None:
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Sharpening strength must be zero or greater.")


def laplacian_sharpening(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Subtract a signed four-neighbor Laplacian to accentuate edges."""
    check_grayscale(image)
    _check_strength(strength)
    edges = cv2.Laplacian(image, cv2.CV_32F, ksize=1)
    depth = cv2.CV_16U if image.dtype == np.uint16 else cv2.CV_8U
    return cv2.addWeighted(image, 1.0, edges, -strength, 0, dtype=depth)


def unsharp_masking(image: np.ndarray, strength: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Add a scaled copy of the detail removed by Gaussian smoothing."""
    check_grayscale(image)
    _check_strength(strength)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Blur sigma must be greater than zero.")
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
