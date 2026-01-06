"""
Internal Array Utilities for Feature Extraction
================================================

This module provides shared array manipulation utilities used by texture and morphology
feature calculation modules. These are internal functions not intended for external use.

Note: The underscore prefix (_utils) indicates this is a private module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BBoxInfo:
    """Bounding box information for cropped arrays."""

    slices: tuple[slice, slice, slice]
    origin_offset: tuple[int, int, int]  # (z, y, x) offset from original origin


def compute_nonzero_bbox(mask: np.ndarray) -> Optional[tuple[slice, slice, slice]]:
    """Compute the tight bounding box of non-zero voxels in a 3D mask.

    Args:
        mask: 3D array where non-zero indicates ROI.

    Returns:
        A tuple of slices (z, y, x) covering the non-zero region, or None if the mask is empty.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={mask.shape!r}")

    m = mask != 0
    z_any = np.any(m, axis=(1, 2))
    if not bool(np.any(z_any)):
        return None
    y_any = np.any(m, axis=(0, 2))
    x_any = np.any(m, axis=(0, 1))

    z0 = int(np.argmax(z_any))
    z1 = int(len(z_any) - 1 - np.argmax(z_any[::-1]))
    y0 = int(np.argmax(y_any))
    y1 = int(len(y_any) - 1 - np.argmax(y_any[::-1]))
    x0 = int(np.argmax(x_any))
    x1 = int(len(x_any) - 1 - np.argmax(x_any[::-1]))

    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


def crop_arrays_to_bbox(
    *arrays: np.ndarray,
    mask: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], Optional[BBoxInfo]]:
    """Crop multiple arrays to the bounding box of the mask.

    Args:
        *arrays: Arrays to crop (must have same shape as mask).
        mask: Reference mask for computing bounding box.

    Returns:
        Tuple of (cropped_arrays, bbox_info).
        bbox_info is None if mask is empty.
    """
    bbox = compute_nonzero_bbox(mask)
    if bbox is None:
        return arrays, None

    cropped = tuple(arr[bbox] for arr in arrays)
    origin_offset = (bbox[0].start, bbox[1].start, bbox[2].start)
    info = BBoxInfo(slices=bbox, origin_offset=origin_offset)

    return cropped, info
