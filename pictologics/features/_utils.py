"""
Internal Array Utilities for Feature Extraction
================================================

This module provides shared array manipulation utilities used by texture and morphology
feature calculation modules. These are internal functions not intended for external use.

Note: The underscore prefix (_utils) indicates this is a private module.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numba import jit, prange
from numpy import typing as npt


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _bbox_scan_numba(
    mask: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Single parallel pass over the mask collecting per-slice nonzero extents.

    Avoids the `mask != 0` boolean temporary and the three separate axis
    reductions of the pure-numpy approach. The whole-row `!= 0` OR-reduction is
    branch-free and SIMD-vectorizable, and runs at memory bandwidth (measured
    ~1.3x over a blocked early-exit scan at CT row widths on float64, and equal
    to a dedicated uint8 max-reduction); the scalar locate loops only touch
    non-empty rows. `!= 0` is correct for every mask dtype: negative values
    count, and NaN counts as nonzero, matching `mask != 0`.
    """
    depth, height, width = mask.shape
    z_any = np.zeros(depth, dtype=np.uint8)
    y_min = np.full(depth, height, dtype=np.int64)
    y_max = np.full(depth, -1, dtype=np.int64)
    x_min = np.full(depth, width, dtype=np.int64)
    x_max = np.full(depth, -1, dtype=np.int64)

    for z in prange(depth):
        for y in range(height):
            # Vectorized any-nonzero test for the whole row.
            hit = False
            for x in range(width):
                hit |= mask[z, y, x] != 0
            if not hit:
                continue
            # First nonzero from the left (row is known non-empty).
            first = 0
            for x in range(width):
                if mask[z, y, x] != 0:
                    first = x
                    break
            # Last nonzero from the right (never left of `first`).
            last = first
            for x in range(width - 1, first, -1):
                if mask[z, y, x] != 0:
                    last = x
                    break

            z_any[z] = 1
            if y < y_min[z]:
                y_min[z] = y
            if y > y_max[z]:
                y_max[z] = y
            if first < x_min[z]:
                x_min[z] = first
            if last > x_max[z]:
                x_max[z] = last

    return z_any, y_min, y_max, x_min, x_max


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _roi_min_max_numba(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Per-slice min/max of `data` over `mask > 0` voxels in a single fused pass."""
    depth, height, width = data.shape
    found = np.zeros(depth, dtype=np.uint8)
    mins = np.full(depth, np.inf, dtype=np.float64)
    maxs = np.full(depth, -np.inf, dtype=np.float64)

    for z in prange(depth):
        lo = np.inf
        hi = -np.inf
        hit = False
        for y in range(height):
            for x in range(width):
                if mask[z, y, x] > 0:
                    hit = True
                    v = data[z, y, x]
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
        if hit:
            found[z] = 1
            mins[z] = lo
            maxs[z] = hi

    return found, mins, maxs


@jit(nopython=True, cache=True)  # type: ignore
def _roi_min_max_serial_numba(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Serial variant of `_roi_min_max_numba` for small volumes, where the
    parallel threading-dispatch cost exceeds the scan itself."""
    depth, height, width = data.shape
    found = np.zeros(depth, dtype=np.uint8)
    mins = np.full(depth, np.inf, dtype=np.float64)
    maxs = np.full(depth, -np.inf, dtype=np.float64)

    for z in range(depth):
        lo = np.inf
        hi = -np.inf
        hit = False
        for y in range(height):
            for x in range(width):
                if mask[z, y, x] > 0:
                    hit = True
                    v = data[z, y, x]
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
        if hit:
            found[z] = 1
            mins[z] = lo
            maxs[z] = hi

    return found, mins, maxs


def roi_min_max(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[Any],
) -> Optional[tuple[float, float]]:
    """Min and max of `data` over ROI voxels (`mask > 0`).

    Equivalent to `(data[mask > 0].min(), data[mask > 0].max())` but in a single
    fused pass, without the boolean-mask and gathered-copy temporaries.

    Args:
        data: 3D array of values.
        mask: 3D array where values > 0 indicate ROI membership. Same shape as data.

    Returns:
        (min, max) over the ROI, or None if the mask has no positive voxels.
    """
    if data.ndim != 3 or data.shape != mask.shape:
        raise ValueError(
            f"Expected two 3D arrays of equal shape, got {data.shape!r} vs {mask.shape!r}"
        )
    # Measured crossover: below ~2^19 voxels the parallel launch overhead dominates.
    if data.size < 1 << 19:
        found, mins, maxs = _roi_min_max_serial_numba(data, mask)
    else:
        found, mins, maxs = _roi_min_max_numba(data, mask)
    idx = np.flatnonzero(found)
    if idx.size == 0:
        return None
    return float(mins[idx].min()), float(maxs[idx].max())


def compute_nonzero_bbox(
    mask: npt.NDArray[Any],
) -> Optional[tuple[slice, slice, slice]]:
    """Compute the tight bounding box of non-zero voxels in a 3D mask.

    Args:
        mask: 3D array where non-zero indicates ROI.

    Returns:
        A tuple of slices (z, y, x) covering the non-zero region, or None if the mask is empty.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={mask.shape!r}")

    # For small masks the numba parallel-launch overhead exceeds the scan itself;
    # keep the pure-numpy reductions there.
    if mask.size < 1 << 20:
        m = mask != 0
        z_any_np = np.any(m, axis=(1, 2))
        if not bool(np.any(z_any_np)):
            return None
        y_any = np.any(m, axis=(0, 2))
        x_any = np.any(m, axis=(0, 1))

        z0 = int(np.argmax(z_any_np))
        z1 = int(len(z_any_np) - 1 - np.argmax(z_any_np[::-1]))
        y0 = int(np.argmax(y_any))
        y1 = int(len(y_any) - 1 - np.argmax(y_any[::-1]))
        x0 = int(np.argmax(x_any))
        x1 = int(len(x_any) - 1 - np.argmax(x_any[::-1]))
        return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)

    z_any, y_min, y_max, x_min, x_max = _bbox_scan_numba(mask)
    nz = np.flatnonzero(z_any)
    if nz.size == 0:
        return None

    z0, z1 = int(nz[0]), int(nz[-1])
    y0 = int(y_min[nz].min())
    y1 = int(y_max[nz].max())
    x0 = int(x_min[nz].min())
    x1 = int(x_max[nz].max())

    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


def merge_bboxes(
    a: Optional[tuple[slice, slice, slice]],
    b: Optional[tuple[slice, slice, slice]],
) -> Optional[tuple[slice, slice, slice]]:
    """Merge two nonzero bounding boxes per axis.

    The bbox of a union of two nonzero sets is the per-axis merge of their bboxes,
    so no union array needs to be materialised. None (empty mask) is the identity.
    """
    if a is None:
        return b
    if b is None:
        return a
    return (
        slice(min(a[0].start, b[0].start), max(a[0].stop, b[0].stop)),
        slice(min(a[1].start, b[1].start), max(a[1].stop, b[1].stop)),
        slice(min(a[2].start, b[2].start), max(a[2].stop, b[2].stop)),
    )
