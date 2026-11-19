"""Histogram-based contrast adjustment and intensity counts."""

import cv2
import numpy as np

from .preprocessing import check_grayscale


def intensity_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Count every pixel, including black and the highest representable gray level."""
    check_grayscale(image)
    levels = int(np.iinfo(image.dtype).max) + 1
    counts = cv2.calcHist([image], [0], None, [bins], [0, levels])
    return counts.ravel()


def global_equalization(image: np.ndarray) -> np.ndarray:
    """Spread the occupied gray levels using their cumulative histogram."""
    check_grayscale(image)
    if image.dtype == np.uint8:
        return cv2.equalizeHist(image)

    # equalizeHist accepts only 8-bit input. Build the same mapping at 16-bit precision.
    counts = intensity_histogram(image, bins=65536)
    cumulative = counts.cumsum(dtype=np.float64)
    first = cumulative[np.flatnonzero(counts)[0]]
    if first == image.size:
        return image.copy()
    mapped = (cumulative - first) * (65535.0 / (image.size - first))
    lookup = np.rint(np.clip(mapped, 0, 65535)).astype(np.uint16)
    return lookup[image]


def clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Equalize local tiles while limiting amplification within each tile."""
    check_grayscale(image)
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("The clip limit must be greater than zero.")
    if not isinstance(grid_size, (int, np.integer)) or not 1 <= grid_size <= 16:
        raise ValueError("Use between 1 and 16 tiles per axis.")
    height, width = image.shape
    grid = (min(grid_size, width), min(grid_size, height))
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(image)
