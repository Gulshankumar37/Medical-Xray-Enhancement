"""Noise-reduction filters for grayscale images."""

import cv2
import numpy as np

from .preprocessing import check_grayscale


def _check_kernel(size: int, maximum: int = 15) -> None:
    if not isinstance(size, (int, np.integer)) or not 3 <= size <= maximum or size % 2 == 0:
        raise ValueError(f"Choose an odd kernel size between 3 and {maximum}.")


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Smooth pixels using a Gaussian-weighted neighborhood."""
    check_grayscale(image)
    _check_kernel(kernel_size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be greater than zero.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Replace each pixel with its neighborhood median."""
    check_grayscale(image)
    # OpenCV supports 3x3 and 5x5 median kernels for 16-bit images.
    _check_kernel(kernel_size, maximum=5 if image.dtype == np.uint16 else 15)
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
    image: np.ndarray, diameter: int = 7, sigma_color: float = 35, sigma_space: float = 5
) -> np.ndarray:
    """Smooth nearby pixels of similar brightness. Intensity sigma uses a 0–255 scale."""
    check_grayscale(image)
    _check_kernel(diameter)
    if not all(np.isfinite(value) and value > 0 for value in (sigma_color, sigma_space)):
        raise ValueError("Both bilateral sigma values must be positive.")
    if image.dtype == np.uint8:
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    # bilateralFilter supports floating point for high-depth data. Keep native intensity units.
    smooth = cv2.bilateralFilter(image.astype(np.float32), diameter, sigma_color * 257, sigma_space)
    return np.rint(np.clip(smooth, 0, 65535)).astype(np.uint16)
