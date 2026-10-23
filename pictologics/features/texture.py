"""
Texture Feature Extraction Module
=================================

This module provides a comprehensive suite of functions for calculating 3D texture features
from medical images. It implements the Image Biomarker Standardisation Initiative (IBSI)
compliant algorithms for various texture matrices.

Key Concepts:
-------------
Texture analysis quantifies the spatial arrangement of grey levels in an image.
It assumes that the texture (e.g., "smooth", "coarse", "regular") is contained in the
spatial relationship between the grey levels of the voxels.

Implemented Matrices:
---------------------
1.  **GLCM (Grey Level Co-occurrence Matrix)**:
    Counts how often pairs of grey levels occur at a specific distance and direction.
    *Captures*: Contrast, homogeneity, correlation.

2.  **GLRLM (Grey Level Run Length Matrix)**:
    Counts the lengths of consecutive runs of the same grey level.
    *Captures*: Coarseness, directionality.

3.  **GLSZM (Grey Level Size Zone Matrix)**:
    Counts the size of zones (connected components) of the same grey level.
    *Captures*: Regional homogeneity, size distribution of texture elements.

4.  **GLDZM (Grey Level Distance Zone Matrix)**:
    Counts zones based on their distance from the ROI border.
    *Captures*: Spatial distribution relative to the boundary.

5.  **NGTDM (Neighbourhood Grey Tone Difference Matrix)**:
    Quantifies the difference between a voxel and its neighbours.
    *Captures*: Human perception of texture (coarseness, contrast, busyness).

6.  **NGLDM (Neighbourhood Grey Level Dependence Matrix)**:
    Captures the dependence of grey levels on their neighbours.
    *Captures*: Dependence, spatial relationships.

Optimization:
-------------
This module uses `numba` for Just-In-Time (JIT) compilation to achieve high performance.
The core calculations are parallelized and optimized for memory usage.
- **Single-pass calculation**: Multiple matrices are computed in a single pass over the image
  to minimize memory access overhead.
- **Flattened DFS**: Zone-based features (GLSZM, GLDZM) use a memory-efficient Depth-First Search
  with flattened stack indices.

Usage:
------
`calculate_all_texture_features` computes the matrices once and gives the features of all six
families, or of the families in `families`. `calculate_all_texture_matrices` gives the raw
matrices, and the family functions (e.g., `calculate_glcm_features`) take them. All take the
IBSI 1 distances `glcm_distance`, `ngtdm_distance` and `ngldm_distance` (default 1) and the
NGLDM coarseness `ngldm_alpha` (default 0).

Example:
        Calculate texture features:

        ```python
        import numpy as np
        from pictologics.features.texture import (
            calculate_all_texture_matrices,
            calculate_glcm_features
        )

        # Create dummy data
        data = np.random.randint(1, 33, (50, 50, 50))
        mask = np.ones((50, 50, 50))

        # Calculate matrices
        matrices = calculate_all_texture_matrices(data, mask, n_bins=32)

        # Extract features
        glcm_feats = calculate_glcm_features(
            data,
            mask,
            n_bins=32,
            glcm_matrix=matrices['glcm']
        )
        print(glcm_feats['contrast_ACUI'])
        ```
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Iterable
from functools import lru_cache
from typing import Any, Optional, cast

import numba
import numpy as np
from numba import jit, prange
from numba.np.ufunc.parallel import get_thread_id
from numpy import typing as npt
from scipy.ndimage import distance_transform_cdt

from ._utils import PRANGE_ONLY, compute_nonzero_bbox, merge_bboxes, roi_min_max, serial_twin

# The texture feature families, in the order of their features
_TEXTURE_FAMILIES = ("glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm")


def _maybe_crop_to_bbox(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    distance_mask: Optional[npt.NDArray[np.floating[Any]]] = None,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    Optional[npt.NDArray[np.floating[Any]]],
]:
    """Crop data/masks to a tight bounding box around the ROI.

    Cropping is a major performance win for sparse ROIs because the texture kernels are
    mostly memory-bandwidth bound and otherwise touch the full image volume.

    The bounding box is computed from the union of `mask` and `distance_mask` (when provided)
    to preserve GLDZM distance-map correctness.
    """
    if data.shape != mask.shape:
        raise ValueError(
            f"data and mask must have the same shape, got {data.shape!r} vs {mask.shape!r}"
        )
    if distance_mask is not None and distance_mask.shape != mask.shape:
        raise ValueError(
            "distance_mask must have the same shape as mask, "
            f"got {distance_mask.shape!r} vs {mask.shape!r}"
        )

    bbox = compute_nonzero_bbox(mask)
    if distance_mask is not None:
        bbox = merge_bboxes(bbox, compute_nonzero_bbox(distance_mask))

    if bbox is None:
        return _level_form(data), mask, distance_mask

    data_c = _level_form(data[bbox])
    mask_c = mask[bbox]
    dist_c = distance_mask[bbox] if distance_mask is not None else None
    return data_c, mask_c, dist_c


def _level_form(data: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """The grey levels of a direct call as a type and order that the import compiles for the
    texture kernels: int32 (from discretise_image), int64 or float64, in row order. Another
    integer or bool type becomes int64, another float type float64: the same values, so the
    same features. A column-order array becomes a row-order copy. A strided crop stays as it
    is (the import compiles that form too)."""
    if data.dtype.kind in "biu" and data.dtype not in (np.int32, np.int64):
        return data.astype(np.int64, order="C")
    if data.dtype.kind == "f" and data.dtype != np.float64:
        return data.astype(np.float64, order="C")
    if data.flags.f_contiguous and not data.flags.c_contiguous:
        return np.ascontiguousarray(data)
    return data


def _roi_voxel_count(mask: npt.NDArray[np.floating[Any]]) -> int:
    """Count ROI voxels using nonzero mask membership semantics."""
    # np.count_nonzero has no fast path for float dtypes (it inspects each 8-byte
    # value), so on a float mask it is ~5x slower than counting a bool array.
    # `mask != 0` yields the identical count (NaN/Inf -> True, -0.0 -> False) via the
    # fast bool popcount path; integer/bool masks already hit that path directly.
    if mask.dtype.kind == "f":
        return int(np.count_nonzero(mask != 0))
    return int(np.count_nonzero(mask))


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _chamfer_distance_taxicab_numba(
    mask_bool: npt.NDArray[np.floating[Any]],
    planar0: bool = False,
    planar1: bool = False,
    planar2: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """
    Exact 2-pass raster chamfer distance transform (taxicab/cityblock metric, 6-connectivity).

    An axis of size 1 in the image (planarN) has no border step: a one-slice image gets the
    in-plane (4-connected) distance, not 1 everywhere.

    Forward and backward raster sweeps propagate distance-to-background using unit-weight
    face-neighbour steps. This is mathematically exact (not approximate) for the L1 metric
    with a connectivity-1 structuring element. Out-of-bounds neighbours are treated as
    background, matching the effect of the one-voxel zero-padding used by the
    `distance_transform_cdt` call this replaces.

    Returns a `(depth+2, height+2, width+2)` uint16 array; the caller slices
    `[1:-1, 1:-1, 1:-1]` to recover the unpadded distance map, matching the layout the
    scipy-based padded-then-sliced code previously produced.
    """
    depth, height, width = mask_bool.shape
    p_depth, p_height, p_width = depth + 2, height + 2, width + 2
    dist = np.zeros((p_depth, p_height, p_width), dtype=np.uint16)
    inf = np.uint16(min(depth + height + width + 1, 65535))

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if mask_bool[z, y, x]:
                    dist[z + 1, y + 1, x + 1] = inf

    # Forward pass: increasing z, y, x. Predecessor neighbours are already-visited
    # positions in raster order; the padding border (still 0) supplies the background
    # distance for voxels touching the array edge.
    for z in range(1, p_depth - 1):
        for y in range(1, p_height - 1):
            for x in range(1, p_width - 1):
                if dist[z, y, x] == 0:
                    continue
                best = dist[z, y, x]
                cand = dist[z - 1, y, x] + 1
                if cand < best and not planar0:
                    best = cand
                cand = dist[z, y - 1, x] + 1
                if cand < best and not planar1:
                    best = cand
                cand = dist[z, y, x - 1] + 1
                if cand < best and not planar2:
                    best = cand
                dist[z, y, x] = best

    # Backward pass: decreasing z, y, x. Successor neighbours.
    for z in range(p_depth - 2, 0, -1):
        for y in range(p_height - 2, 0, -1):
            for x in range(p_width - 2, 0, -1):
                if dist[z, y, x] == 0:
                    continue
                best = dist[z, y, x]
                cand = dist[z + 1, y, x] + 1
                if cand < best and not planar0:
                    best = cand
                cand = dist[z, y + 1, x] + 1
                if cand < best and not planar1:
                    best = cand
                cand = dist[z, y, x + 1] + 1
                if cand < best and not planar2:
                    best = cand
                dist[z, y, x] = best

    return dist  # type: ignore[return-value]


@jit(nopython=True, nogil=True, parallel=True, cache=True)  # type: ignore
def _distance_map_parallel_numba(
    mask_bool: npt.NDArray[np.bool_], planar0: bool, planar1: bool, planar2: bool
) -> npt.NDArray[np.uint16]:
    """`_chamfer_distance_taxicab_numba` in threads, with the same uint16 map. The taxicab
    distance is separable: one pass forward and one pass backward along each axis give
    the same integers as the two raster passes. The passes along x and y run in each
    slice (the slices in parallel), then the passes along z (the rows in parallel)."""
    depth, height, width = mask_bool.shape
    dist = np.zeros((depth + 2, height + 2, width + 2), dtype=np.uint16)
    inf = np.uint16(min(depth + height + width + 1, 65535))
    for z in prange(1, depth + 1):
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                if mask_bool[z - 1, y - 1, x - 1]:
                    dist[z, y, x] = inf
            if not planar2:
                for x in range(1, width + 1):
                    cand = dist[z, y, x - 1] + 1
                    if cand < dist[z, y, x]:
                        dist[z, y, x] = cand
                for x in range(width, 0, -1):
                    cand = dist[z, y, x + 1] + 1
                    if cand < dist[z, y, x]:
                        dist[z, y, x] = cand
        if not planar1:
            for y in range(1, height + 1):
                for x in range(1, width + 1):
                    cand = dist[z, y - 1, x] + 1
                    if cand < dist[z, y, x]:
                        dist[z, y, x] = cand
            for y in range(height, 0, -1):
                for x in range(1, width + 1):
                    cand = dist[z, y + 1, x] + 1
                    if cand < dist[z, y, x]:
                        dist[z, y, x] = cand
    if not planar0:
        for y in prange(1, height + 1):
            for z in range(1, depth + 1):
                for x in range(1, width + 1):
                    cand = dist[z - 1, y, x] + 1
                    if cand < dist[z, y, x]:
                        dist[z, y, x] = cand
            for z in range(depth, 0, -1):
                for x in range(1, width + 1):
                    cand = dist[z + 1, y, x] + 1
                    if cand < dist[z, y, x]:
                        dist[z, y, x] = cand
    return dist


# From this many voxels on (and two slices), the distance map runs in threads. Measured on
# 10 threads: 0.4 to 0.6 times the serial time from 2^18 voxels on; the two parallel
# regions cost about 0.2 ms, so below about 190,000 voxels the serial passes win.
_DISTANCE_PARALLEL_MIN = 1 << 18


def _distance_map(
    mask_bool: npt.NDArray[np.bool_], planar: tuple[bool, bool, bool]
) -> npt.NDArray[np.uint16]:
    """The padded GLDZM distance map of a row-order bool mask (see
    `_chamfer_distance_taxicab_numba`), in threads for a large mask."""
    if mask_bool.size >= _DISTANCE_PARALLEL_MIN and mask_bool.shape[0] > 1:
        return cast(npt.NDArray[np.uint16], _distance_map_parallel_numba(mask_bool, *planar))
    return cast(npt.NDArray[np.uint16], _chamfer_distance_taxicab_numba(mask_bool, *planar))


def _gldzm_distance_map(
    mask_bool: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """
    Compute the GLDZM distance-to-border map for a binary ROI mask.

    Equivalent to `scipy.ndimage.distance_transform_cdt(mask, metric="taxicab")` on `mask`
    zero-padded by one voxel on every side, sliced back to the original shape: background
    voxels get distance 0, foreground voxels get the taxicab (cityblock) distance to the
    nearest background voxel, and voxels outside the array are treated as background.

    3D masks (the only shape the texture pipeline produces) use the exact numba kernels of
    `_distance_map`. Any other dimensionality falls back to the original scipy-based
    implementation.
    """
    if mask_bool.ndim == 3:
        padded = _distance_map(
            np.ascontiguousarray(mask_bool, dtype=np.bool_), (False, False, False)
        )
        return cast(npt.NDArray[Any], padded[1:-1, 1:-1, 1:-1])

    mask_padded = np.pad(mask_bool, 1, mode="constant", constant_values=0)
    dist_map_padded = distance_transform_cdt(mask_padded, metric="taxicab").astype(np.uint16)
    unpad = tuple(slice(1, -1) for _ in range(mask_bool.ndim))
    return cast(npt.NDArray[Any], dist_map_padded[unpad])


# --- Combined Local Features Kernel ---

# Pre-calculate 26-neighbor offsets for NGTDM/NGLDM
# Generate all combinations of -1, 0, 1 using mgrid
_z, _y, _x = np.mgrid[-1:2, -1:2, -1:2]
_offsets = np.stack([_z.ravel(), _y.ravel(), _x.ravel()], axis=1)
# Remove the center pixel (0, 0, 0) and ensure int32 type
OFFSETS_26 = _offsets[np.any(_offsets != 0, axis=1)].astype(np.int32)

# Define 6-neighbor offsets (Manhattan distance 1)
_manhattan_dist = np.abs(OFFSETS_26[:, 0]) + np.abs(OFFSETS_26[:, 1]) + np.abs(OFFSETS_26[:, 2])
OFFSETS_6 = OFFSETS_26[_manhattan_dist == 1].astype(np.int32)

# Convert to tuple of tuples for literal_unroll (pure Python ints)
OFFSETS_26_TUPLE = tuple(tuple(map(int, row)) for row in OFFSETS_26)

# Filter for unique directions (first non-zero element > 0)
# This gives 13 directions for 3D (Chebyshev distance 1)
_c0 = OFFSETS_26[:, 0]
_c1 = OFFSETS_26[:, 1]
_c2 = OFFSETS_26[:, 2]
_mask = (_c0 > 0) | ((_c0 == 0) & (_c1 > 0)) | ((_c0 == 0) & (_c1 == 0) & (_c2 > 0))
DIRECTIONS_13 = OFFSETS_26[_mask]
# Convert to pure Python tuples/ints for literal_unroll compatibility
DIRECTIONS_13_TUPLE = tuple(tuple(map(int, row)) for row in DIRECTIONS_13)
DIRECTIONS_13_WITH_ID = tuple(enumerate(DIRECTIONS_13_TUPLE))


# --- Texture kernels ---
# The kernels read one zero-padded volume of grey levels: 0 marks a voxel outside the ROI
# (or with a level outside [1, n_bins]), and an ROI voxel holds its 1-based grey level.
# The padding stops every neighbour step and every run at the box edge, with no bounds
# check. uint16 holds every n_bins up to 65,535, so each kernel compiles once.
_MAX_TEXTURE_LEVELS = 65535


@jit(nopython=True, nogil=True, parallel=True, cache=True)  # type: ignore
def _texture_volume_numba(
    data: npt.NDArray[Any],
    roi: npt.NDArray[np.bool_],
    n_bins: int,
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
    span: npt.NDArray[np.float64],
) -> None:
    """Write the level of each ROI voxel with a level in [1, n_bins] into the interior of
    the zero array `vol`. counts[z, 0] gets the number of these voxels in slice z, and
    counts[z, 1] the number of ROI voxels. span[z] gets the smallest and the largest of
    these levels (inf and -inf for none), for the grey-level span of the GLCM."""
    depth, height, width = data.shape
    for z in prange(depth):
        valid = 0
        inside = 0
        lo = np.inf
        hi = -np.inf
        for y in range(height):
            for x in range(width):
                if roi[z, y, x]:
                    inside += 1
                    g = data[z, y, x]
                    if g >= 1 and g <= n_bins:
                        vol[z + 1, y + 1, x + 1] = g
                        valid += 1
                        lo = min(lo, np.float64(g))
                        hi = max(hi, np.float64(g))
        counts[z, 0] = valid
        counts[z, 1] = inside
        span[z, 0] = lo
        span[z, 1] = hi


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _texture_volume_serial_numba(
    data: npt.NDArray[Any],
    roi: npt.NDArray[np.bool_],
    n_bins: int,
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
    span: npt.NDArray[np.float64],
) -> None:
    """Serial variant of `_texture_volume_numba` for small volumes, where starting the
    threads costs more than the pass (about 85 us)."""
    depth, height, width = data.shape
    for z in range(depth):
        valid = 0
        inside = 0
        lo = np.inf
        hi = -np.inf
        for y in range(height):
            for x in range(width):
                if roi[z, y, x]:
                    inside += 1
                    g = data[z, y, x]
                    if g >= 1 and g <= n_bins:
                        vol[z + 1, y + 1, x + 1] = g
                        valid += 1
                        lo = min(lo, np.float64(g))
                        hi = max(hi, np.float64(g))
        counts[z, 0] = valid
        counts[z, 1] = inside
        span[z, 0] = lo
        span[z, 1] = hi


# From this many voxels on, the grey-level volume is built in threads
_PARALLEL_VOLUME_MIN = 1 << 16


def _texture_volume(
    data: npt.NDArray[Any],
    roi: npt.NDArray[np.bool_],
    n_bins: int,
    span: Optional[npt.NDArray[np.float64]] = None,
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.int64]]:
    """The padded grey-level volume of the texture kernels, and the counts of
    `_texture_volume_numba` (texture voxels and ROI voxels of each slice). `span`, a
    (depth, 2) float64 array, gets the smallest and the largest level of each slice."""
    if n_bins > _MAX_TEXTURE_LEVELS:
        raise ValueError(
            f"Texture features take at most {_MAX_TEXTURE_LEVELS:,} grey levels, not {n_bins:,}"
        )
    vol = np.zeros(tuple(s + 2 for s in data.shape), dtype=np.uint16)
    counts = np.empty((data.shape[0], 2), dtype=np.int64)
    if span is None:
        span = np.empty((data.shape[0], 2), dtype=np.float64)
    if data.size < _PARALLEL_VOLUME_MIN:
        _texture_volume_serial_numba(data, roi, n_bins, vol, counts, span)
    else:
        _texture_volume_numba(data, roi, n_bins, vol, counts, span)
    return vol, counts


def _z_blocks(per_slice: npt.NDArray[np.int64], n_blocks: int) -> npt.NDArray[np.int64]:
    """Bounds of at most `n_blocks` contiguous slice ranges with about the same number of
    texture voxels, so that each thread gets about the same work."""
    cum = np.concatenate(([0], np.cumsum(per_slice)))
    bounds = np.searchsorted(cum, cum[-1] * np.arange(n_blocks + 1) / n_blocks, side="left")
    bounds[0], bounds[-1] = 0, per_slice.size
    return np.unique(bounds).astype(np.int64)


@lru_cache(maxsize=None)
def _directions(planar: tuple[bool, bool, bool]) -> npt.NDArray[np.int64]:
    """The indices of the 13 directions that stay in the image. A direction that steps
    along an axis of size 1 (a one-slice image) leaves the image at once: in it, every
    voxel would be a run of length 1, so the GLCM and the GLRLM use the in-plane
    directions only, as for a 2D image. Cached and shared: do not write it (it stays
    writable, so the local kernel takes it with its compiled signature)."""
    steps_out = (DIRECTIONS_13 != 0) & np.array(planar)
    return np.flatnonzero(~steps_out.any(axis=1)).astype(np.int64)


@lru_cache(maxsize=None)
def _neighbour_offsets(distance: int) -> npt.NDArray[np.int32]:
    """The (dz, dy, dx) steps to the voxels at Chebyshev distance 1 to `distance` (the
    NGTDM and NGLDM neighbourhood), in the order of OFFSETS_26 (its 26 steps for 1).
    Cached and read-only."""
    steps = np.arange(-distance, distance + 1)
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = grid[np.any(grid != 0, axis=1)].astype(np.int32)
    offsets.flags.writeable = False
    return offsets


def _neighbourhood_size(distance: int) -> int:
    """The voxels of a neighbourhood with its center: (2 distance + 1)^3 (27 for 1)."""
    return (2 * distance + 1) ** 3


def _flat_offsets(shape: tuple[int, ...], offsets: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    """The (dz, dy, dx) offsets as steps in the flat index of an array of `shape`."""
    steps = offsets.astype(np.int64) @ np.array([shape[1] * shape[2], shape[2], 1])
    return cast(npt.NDArray[np.int64], steps.astype(np.int64))


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _pairs_and_runs_voxel(
    flat: npt.NDArray[np.uint16],
    v: int,
    g: int,
    tid: int,
    calc_glcm: bool,
    calc_glrlm: bool,
    pair_off: npt.NDArray[np.int64],
    dir_off: npt.NDArray[np.int64],
    dir_table: npt.NDArray[np.int64],
    merged: bool,
    glcm: npt.NDArray[np.uint32],
    glrlm: npt.NDArray[np.uint32],
) -> None:
    """Add the GLCM pairs and the GLRLM runs of voxel v (level g > 0) in each direction to
    the tables of thread tid: the path of a row segment shorter than _ROW_SEGMENT_MIN
    voxels (see _pairs_and_runs_row for the tables and the runs)."""
    i = g - 1
    last = flat.size - 1
    ones = np.uint32(0)
    twos = np.uint32(0)
    for d in range(dir_off.shape[0]):
        od = dir_off[d]
        t = dir_table[d]
        if calc_glcm:
            glcm[tid, 0, t, g, flat[v + pair_off[d]]] += 1
        if calc_glrlm:
            # & on the booleans: Python's "and" would make a branch
            starts = np.int64(flat[v - od]) != g
            ends = np.int64(flat[v + od]) != g
            ends_2 = np.int64(flat[min(v + 2 * od, last)]) != g
            one = np.uint32(starts & ends)
            two = np.uint32(starts & (not ends) & ends_2)
            if merged:
                ones += one
                twos += two
            else:
                glrlm[tid, t, i, 1] += one
                glrlm[tid, t, i, 2] += two
            if starts & (not ends) & (not ends_2):
                length = 3
                c = v + 3 * od
                while np.int64(flat[c]) == g:
                    length += 1
                    c += od
                glrlm[tid, t, i, length] += 1
    if calc_glrlm and merged:
        glrlm[tid, dir_table[0], i, 1] += ones
        glrlm[tid, dir_table[0], i, 2] += twos


# A row segment of fewer ROI voxels goes voxel by voxel: the row passes pay a start for
# each direction
_ROW_SEGMENT_MIN = 8


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _pairs_and_runs_row(
    flat: npt.NDArray[np.uint16],
    start: int,
    n: int,
    tid: int,
    calc_glcm: bool,
    calc_glrlm: bool,
    pair_off: npt.NDArray[np.int64],
    dir_off: npt.NDArray[np.int64],
    dir_table: npt.NDArray[np.int64],
    merged: bool,
    glcm: npt.NDArray[np.uint32],
    glrlm: npt.NDArray[np.uint32],
    ones_x: npt.NDArray[np.uint32],
    twos_x: npt.NDArray[np.uint32],
    longf: npt.NDArray[np.bool_],
    idx: npt.NDArray[np.int64],
) -> None:
    """Add the GLCM pairs and the GLRLM runs of the n ROI voxels of the row segment from
    flat index `start` (no voxel of the segment is 0) in each direction to the tables of
    thread tid. The GLCM tables are 1-based with a dummy row and column 0 (the pairs with
    a voxel outside the ROI land there). The pairs of the even and the odd x alternate
    between the two tables of the thread. A run starts where
    the voxel before has another level. Runs of length 1 and 2 count from three row
    slices (the voxel before, after and two after), in one loop without a branch; a walk
    counts the longer runs. With `merged` (one table for all directions), the runs of
    length 1 and 2 add once for each voxel."""
    size = flat.shape[0]
    last = size - 1
    row = flat[start : start + n]
    n_alt = glcm.shape[1]
    if calc_glrlm and merged:
        for x in range(n):
            ones_x[x] = 0
            twos_x[x] = 0
    for d in range(dir_off.shape[0]):
        od = dir_off[d]
        t = dir_table[d]
        if calc_glcm:
            po = pair_off[d]
            nxt = flat[start + po : start + po + n]
            if n_alt == 1:
                for x in range(n):
                    glcm[tid, 0, t, row[x], nxt[x]] += 1
            else:
                for x in range(n):
                    glcm[tid, x & 1, t, row[x], nxt[x]] += 1
        if calc_glrlm:
            prev = flat[start - od : start - od + n]
            nxt = flat[start + od : start + od + n]
            if start + 2 * od + n <= size:
                nxt2 = flat[start + 2 * od : start + 2 * od + n]
                if merged:
                    for x in range(n):
                        g = row[x]
                        s = prev[x] != g
                        e = nxt[x] != g
                        e2 = nxt2[x] != g
                        ones_x[x] += np.uint32(s & e)
                        twos_x[x] += np.uint32(s & (not e) & e2)
                        longf[x] = s & (not e) & (not e2)
                else:
                    for x in range(n):
                        g = row[x]
                        s = prev[x] != g
                        e = nxt[x] != g
                        e2 = nxt2[x] != g
                        longf[x] = s & (not e) & (not e2)
                        glrlm[tid, t, g - 1, 1] += np.uint32(s & e)
                        glrlm[tid, t, g - 1, 2] += np.uint32(s & (not e) & e2)
            else:  # the last rows of the volume: clamp the second step
                for x in range(n):
                    g = row[x]
                    v = start + x
                    s = flat[v - od] != g
                    e = flat[v + od] != g
                    e2 = flat[min(v + 2 * od, last)] != g
                    longf[x] = s & (not e) & (not e2)
                    if merged:
                        ones_x[x] += np.uint32(s & e)
                        twos_x[x] += np.uint32(s & (not e) & e2)
                    else:
                        glrlm[tid, t, g - 1, 1] += np.uint32(s & e)
                        glrlm[tid, t, g - 1, 2] += np.uint32(s & (not e) & e2)
            k = 0
            for x in range(n):
                idx[k] = x
                k += int(longf[x])
            for j in range(k):
                x = idx[j]
                g = row[x]
                length = 3
                c = start + x + 3 * od
                while flat[c] == g:
                    length += 1
                    c += od
                glrlm[tid, t, g - 1, length] += 1
    if calc_glrlm and merged:
        t0 = dir_table[0]
        for x in range(n):
            g = row[x]
            glrlm[tid, t0, g - 1, 1] += ones_x[x]
            glrlm[tid, t0, g - 1, 2] += twos_x[x]


@jit(nopython=True, inline="always", cache=True)  # type: ignore
def _neighbour_rows(
    flat: npt.NDArray[np.uint16],
    start: int,
    n: int,
    steps: npt.NDArray[np.int64],
    calc_ngtdm: bool,
    calc_ngldm: bool,
    alpha: int,
    n_sum: npt.NDArray[np.int32],
    n_count: npt.NDArray[np.int16],
    dependence: npt.NDArray[np.int16],
) -> None:
    """The NGTDM and NGLDM sums over the 26 neighbours of the n voxels of a row from flat
    index `start`: the level sum and the count of ROI voxels (n_sum, n_count), and the
    count of ROI voxels within `alpha` levels (dependence). One pass for each of the 9
    neighbour rows (`steps`) reads x - 1, x and x + 1 with contiguous loads."""
    for x in range(n):
        n_sum[x] = 0
        n_count[x] = 0
        dependence[x] = 0
    row = flat[start : start + n]
    for k in range(steps.shape[0]):
        o = start + steps[k] - 1
        r = flat[o : o + n + 2]
        centre = steps[k] == 0  # the row of the voxels: x itself is not a neighbour
        if calc_ngtdm:
            if centre:
                for x in range(n):
                    a = r[x]
                    c = r[x + 2]
                    n_sum[x] += np.int32(a) + np.int32(c)
                    n_count[x] += np.int16(a != 0) + np.int16(c != 0)
            else:
                for x in range(n):
                    a = r[x]
                    m = r[x + 1]
                    c = r[x + 2]
                    n_sum[x] += np.int32(a) + np.int32(m) + np.int32(c)
                    n_count[x] += np.int16(a != 0) + np.int16(m != 0) + np.int16(c != 0)
        if calc_ngldm:
            if alpha == 0:  # the levels equal to the level of x (> 0) are ROI voxels
                if centre:
                    for x in range(n):
                        g = row[x]
                        dependence[x] += np.int16(r[x] == g) + np.int16(r[x + 2] == g)
                else:
                    for x in range(n):
                        g = row[x]
                        dependence[x] += (
                            np.int16(r[x] == g) + np.int16(r[x + 1] == g) + np.int16(r[x + 2] == g)
                        )
            else:
                for x in range(n):
                    g = np.int64(row[x])
                    for j in range(3):
                        if centre and j == 1:
                            continue
                        nv = np.int64(r[x + j])
                        dependence[x] += np.int16((nv != 0) & (abs(nv - g) <= alpha))


@jit(nopython=True, nogil=True, parallel=True, cache=True)  # type: ignore
def _local_tables_numba(
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
    blocks: npt.NDArray[np.int64],
    pad: int,
    calc_glcm: bool,
    calc_glrlm: bool,
    calc_ngtdm: bool,
    calc_ngldm: bool,
    neighbours: npt.NDArray[np.int64],
    pair_off: npt.NDArray[np.int64],
    dir_off: npt.NDArray[np.int64],
    dir_table: npt.NDArray[np.int64],
    ngldm_alpha: int,
    glcm: npt.NDArray[np.uint32],
    glrlm: npt.NDArray[np.uint32],
    ngtdm_d: npt.NDArray[np.int64],
    ngtdm_n: npt.NDArray[np.int64],
    ngldm: npt.NDArray[np.uint32],
) -> None:
    """Add the GLCM, GLRLM, NGTDM and NGLDM counts of the volume `vol` (padded by `pad`
    zero voxels on each side) to the tables of the thread that does the work (first axis
    of each table).

    The slice blocks `blocks` run in parallel. Direction d adds to table `dir_table[d]`
    of the GLCM and the GLRLM: the GLCM pairs step `pair_off[d]` in the flat index (the
    GLCM distance), the GLRLM runs step `dir_off[d]` (see `_pairs_and_runs_row` and
    `_pairs_and_runs_voxel`). The NGTDM
    and the NGLDM read the voxels at the flat steps `neighbours`; for the 26 neighbours of
    distance 1, each row is cut to its first and last ROI voxel and row passes give the
    sums (`_neighbour_rows`). The NGTDM keeps whole numbers: for a voxel of level g whose
    n valid neighbours have the level sum S, it adds |g n - S| (n times |g - S / n|) to
    ngtdm_d[tid, g - 1, n]. Integer sums do not depend on the order of the voxels, so the
    counts do not depend on the number of threads or on the visit order. The tables are
    made and summed outside, so the kernel has no other parallel region.
    """
    height = vol.shape[1] - 2 * pad
    width = vol.shape[2] - 2 * pad
    s0 = vol.shape[1] * vol.shape[2]
    s1 = vol.shape[2]
    flat = vol.ravel()
    alpha = np.int64(ngldm_alpha)
    rows = (calc_ngtdm or calc_ngldm) and neighbours.shape[0] == 26
    local = calc_glcm or calc_glrlm
    steps = np.empty(9, dtype=np.int64)  # the 9 (dz, dy) rows of the neighbourhood
    for k in range(9):
        steps[k] = (k // 3 - 1) * s0 + (k % 3 - 1) * s1
    merged = dir_table.shape[0] > 0  # one table for all directions
    for d in range(1, dir_table.shape[0]):
        if dir_table[d] != dir_table[0]:
            merged = False
    size = width if rows else 0
    lsize = width if local else 0
    for b in prange(blocks.shape[0] - 1):
        tid = get_thread_id()
        n_sum = np.zeros(size, dtype=np.int32)
        n_count = np.zeros(size, dtype=np.int16)
        dependence = np.zeros(size, dtype=np.int16)
        ones_x = np.zeros(lsize, dtype=np.uint32)
        twos_x = np.zeros(lsize, dtype=np.uint32)
        longf = np.zeros(lsize, dtype=np.bool_)
        idx = np.zeros(lsize, dtype=np.int64)
        for z in range(blocks[b], blocks[b + 1]):
            if counts[z, 0] == 0:
                continue
            for y in range(height):
                base = (z + pad) * s0 + (y + pad) * s1 + pad
                # Cut the row to its first and last ROI voxel
                x0 = 0
                while x0 < width and flat[base + x0] == 0:
                    x0 += 1
                if x0 == width:
                    continue
                x1 = width
                while flat[base + x1 - 1] == 0:
                    x1 -= 1
                start = base + x0
                n = x1 - x0
                if rows:
                    _neighbour_rows(
                        flat, start, n, steps, calc_ngtdm, calc_ngldm, alpha,
                        n_sum, n_count, dependence,
                    )  # fmt: skip
                if local:  # the contiguous ROI segments of the row
                    xs = 0
                    while xs < n:
                        if flat[start + xs] == 0:
                            xs += 1
                            continue
                        xe = xs + 1
                        while xe < n and flat[start + xe] != 0:
                            xe += 1
                        if xe - xs >= _ROW_SEGMENT_MIN:
                            _pairs_and_runs_row(
                                flat, start + xs, xe - xs, tid, calc_glcm, calc_glrlm, pair_off,
                                dir_off, dir_table, merged, glcm, glrlm, ones_x, twos_x, longf, idx,
                            )  # fmt: skip
                        else:
                            for x in range(xs, xe):
                                v = start + x
                                _pairs_and_runs_voxel(
                                    flat, v, np.int64(flat[v]), tid, calc_glcm, calc_glrlm,
                                    pair_off, dir_off, dir_table, merged, glcm, glrlm,
                                )  # fmt: skip
                        xs = xe
                if not (calc_ngtdm or calc_ngldm):
                    continue
                for x in range(n):
                    v = start + x
                    g = np.int64(flat[v])
                    if g == 0:
                        continue
                    i = g - 1
                    if rows:
                        if calc_ngtdm and n_count[x] > 0:
                            count = np.int64(n_count[x])
                            ngtdm_d[tid, i, count] += abs(g * count - np.int64(n_sum[x]))
                            ngtdm_n[tid, i] += 1
                        if calc_ngldm:
                            ngldm[tid, i, dependence[x]] += 1
                    elif calc_ngtdm or calc_ngldm:
                        level_sum = np.int64(0)
                        count = np.int64(0)
                        equal = np.int64(1)
                        for k in range(neighbours.shape[0]):
                            nv = np.int64(flat[v + neighbours[k]])
                            m = np.int64(nv != 0)
                            level_sum += nv
                            count += m
                            equal += m * np.int64(abs(nv - g) <= alpha)
                        if calc_ngtdm and count > 0:
                            ngtdm_d[tid, i, count] += abs(g * count - level_sum)
                            ngtdm_n[tid, i] += 1
                        if calc_ngldm:
                            ngldm[tid, i, equal - 1] += 1


# The serial twin of the local kernel: one table, no parallel region (see
# _LOCAL_PARALLEL_MIN_SIZE)
_local_tables_numba_serial = serial_twin(_local_tables_numba)

# Below this many voxels of the padded volume, the local kernel runs as its serial twin with
# one table: the start of the parallel region and the thread tables cost more. Measured on
# pools of 32 spheres (32 levels, all four families, 10 threads), serial against parallel:
# 1,331 voxels x0.27, 3,375 x0.42, 6,859 x0.74, 12,167 x1.09, 19,683 x1.65.
_LOCAL_PARALLEL_MIN_SIZE = 10_000


@jit(nopython=True, nogil=True, parallel=True, cache=True)  # type: ignore
def _levels_seen_numba(vol: npt.NDArray[np.uint16], seen: npt.NDArray[np.bool_]) -> None:
    """seen[z, g] = True for each grey level g in slice z of `vol` (one row per slice, so
    the threads write their own rows)."""
    for z in prange(vol.shape[0]):
        for y in range(vol.shape[1]):
            for x in range(vol.shape[2]):
                seen[z, vol[z, y, x]] = True


# Above this many bytes of GLCM and GLRLM thread tables (2,048 grey levels: 235 MB), the
# tables hold only the levels that occur. Measured: smaller full tables are faster than
# the compact ones, which cost a pass for the levels and a copy back.
_COMPACT_TABLE_BYTES = 64 << 20


def _table_levels(
    vol: npt.NDArray[np.uint16], n_bins: int, table_bytes: int
) -> Optional[npt.NDArray[np.intp]]:
    """The grey levels (0-based) that occur in `vol`, when the GLCM and GLRLM thread tables
    of all n_bins levels would take more than _COMPACT_TABLE_BYTES and at most half of
    the levels occur; else None (the tables keep a row for each level: with more levels,
    the second pass of the compact tables costs more than they save)."""
    if table_bytes <= _COMPACT_TABLE_BYTES:
        return None
    seen = np.zeros((vol.shape[0], n_bins + 1), dtype=np.bool_)
    _levels_seen_numba(vol, seen)
    levels = np.flatnonzero(seen[:, 1:].any(axis=0))
    return levels if 2 * levels.size <= n_bins else None


_NO_TABLE_4D = np.zeros((1, 1, 1, 1), dtype=np.uint32)
_NO_TABLE_5D = np.zeros((1, 1, 1, 1, 1), dtype=np.uint32)
# The GLCM tables of a thread: the pairs of the even and the odd x alternate between them
_GLCM_ALT = 2
_GLCM_ALT_MAX_LEVELS = 64
# From this many cells (4 MB) on, the thread tables are zeroed in threads: np.zeros of a
# large block that the allocator reuses costs a serial memset (2 ms for 48 MB).
_PARALLEL_ZERO_MIN = 1 << 20


@jit(nopython=True, nogil=True, parallel=True, cache=True)  # type: ignore
def _zero_fill_numba(flat: npt.NDArray[np.uint32]) -> None:
    """flat[:] = 0, in parallel blocks."""
    n = flat.size
    for b in prange((n + 65535) // 65536):
        for i in range(b * 65536, min(n, (b + 1) * 65536)):
            flat[i] = 0


def _thread_tables(shape: tuple[int, ...]) -> npt.NDArray[np.uint32]:
    """Zeroed uint32 thread tables of `shape` (large ones zeroed in threads)."""
    if math.prod(shape) < _PARALLEL_ZERO_MIN:
        return np.zeros(shape, dtype=np.uint32)
    tables = np.empty(shape, dtype=np.uint32)
    _zero_fill_numba(tables.reshape(-1))
    return tables


# From this many table cells on, the thread tables add up in parallel. Measured on 14
# threads: 32,768 cells 0.15 against 0.20 ms (numpy, one thread), 851,968 cells (13
# directions at 256 levels) 0.70 against 5.5 ms; 16,384 cells are a tie.
_PARALLEL_SUM_MIN_CELLS = 1 << 15


@jit(nopython=True, nogil=True, parallel=True, cache=True)  # type: ignore
def _thread_sum_numba(tables: npt.NDArray[np.uint32], out: npt.NDArray[np.uint64]) -> None:
    """out[k] = the sum over the threads t of tables[t, k], for (n_threads, n) tables, in
    blocks of the cells so that each thread reads contiguous rows."""
    n_threads, n = tables.shape
    n_blocks = (n + 4095) // 4096
    for b in prange(n_blocks):
        lo = b * 4096
        hi = min(n, lo + 4096)
        for k in range(lo, hi):
            out[k] = tables[0, k]
        for t in range(1, n_threads):
            for k in range(lo, hi):
                out[k] += tables[t, k]


def _thread_sum(tables: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint64]:
    """The sum of the thread tables (first axis) as uint64."""
    if tables[0].size < _PARALLEL_SUM_MIN_CELLS:
        return cast(npt.NDArray[np.uint64], tables.sum(axis=0, dtype=np.uint64))
    out = np.empty(tables.shape[1:], dtype=np.uint64)
    _thread_sum_numba(tables.reshape(tables.shape[0], -1), out.reshape(-1))
    return out


def _local_matrices(
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
    n_bins: int,
    calc_glcm: bool,
    calc_glrlm: bool,
    calc_ngtdm: bool,
    calc_ngldm: bool,
    ngldm_alpha: int,
    merge_directions: bool,
    planar: tuple[bool, bool, bool] = (False, False, False),
    glcm_distance: int = 1,
    ngtdm_distance: int = 1,
    ngldm_distance: int = 1,
    parallel: Optional[bool] = None,
) -> tuple[
    npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]
]:
    """GLCM (n_dirs, n_bins, n_bins), GLRLM (n_dirs, n_bins, longest run + 1), NGTDM s and n
    (n_bins,) and NGLDM (n_bins, neighbourhood size) of the volume padded by one voxel.
    With `merge_directions`, the GLCM and the GLRLM hold one table, the sum over the 13
    directions (the features use only that sum). A matrix that is not asked for is a zero
    placeholder. The directions along an axis of size 1 in the image (`planar`) stay
    empty (see _directions). The GLCM pairs voxels at `glcm_distance` along each
    direction, and the NGTDM and the NGLDM use the voxels up to their Chebyshev distance
    (IBSI 1); a distance above 1 pads the volume to it. `parallel` picks the parallel
    kernel (default: from _LOCAL_PARALLEL_MIN_SIZE voxels on; the warm-up asks for both);
    else the serial twin fills one table, with the same counts."""
    pad = max(
        glcm_distance, ngtdm_distance if calc_ngtdm else 1, ngldm_distance if calc_ngldm else 1
    )
    if pad > 1:
        vol = np.pad(vol, pad - 1)
    if parallel is None:
        parallel = vol.size >= _LOCAL_PARALLEL_MIN_SIZE
    n_threads = numba.get_num_threads() if parallel else 1
    kernel = _local_tables_numba if n_threads > 1 else _local_tables_numba_serial
    n_tables = 1 if merge_directions else 13
    used = _directions(planar)
    longest = max(vol.shape) - 2 * pad
    # Many grey levels: the thread tables hold only the levels that occur
    cells = (n_bins if calc_glcm else 0) + (longest + 1 if calc_glrlm else 0)
    levels = _table_levels(vol, n_bins, 4 * n_threads * n_tables * n_bins * cells)
    rows = n_bins if levels is None else levels.size
    # Up to 64 levels, the pairs of the even and the odd x alternate between two tables
    # (shorter store chains); larger tables cost more to zero and to sum than they save.
    # The serial twin of a small volume keeps one table: it has no long rows, and its one
    # table needs no sum
    n_alt = _GLCM_ALT if rows <= _GLCM_ALT_MAX_LEVELS and parallel else 1
    glcm_t = (
        _thread_tables((n_threads, n_alt, n_tables, rows + 1, rows + 1))
        if calc_glcm
        else _NO_TABLE_5D
    )
    glrlm_t = (
        _thread_tables((n_threads, n_tables, rows, longest + 1)) if calc_glrlm else _NO_TABLE_4D
    )
    ngtdm_size = _neighbourhood_size(ngtdm_distance)
    ngtdm_d = np.zeros((n_threads if calc_ngtdm else 1, n_bins, ngtdm_size), dtype=np.int64)
    ngtdm_n = np.zeros((n_threads if calc_ngtdm else 1, n_bins), dtype=np.int64)
    ngldm_size = _neighbourhood_size(ngldm_distance)
    ngldm_t = np.zeros((n_threads if calc_ngldm else 1, n_bins, ngldm_size), dtype=np.uint32)
    steps = _flat_offsets(vol.shape, DIRECTIONS_13[used])
    tables = (
        np.zeros(used.size, dtype=np.int64) if merge_directions else used,
        ngldm_alpha,
        glcm_t,
        glrlm_t,
        ngtdm_d,
        ngtdm_n,
        ngldm_t,
    )
    # One thread: one block of all slices
    blocks = (
        _z_blocks(counts[:, 0], 4 * n_threads)
        if n_threads > 1
        else np.array([0, counts.shape[0]], dtype=np.int64)
    )
    pairs = steps if glcm_distance == 1 else steps * glcm_distance

    def run(volume: npt.NDArray[np.uint16], families: tuple[bool, ...], distance: int) -> None:
        """One kernel pass for the (GLCM, GLRLM, NGTDM, NGLDM) `families`, with the
        neighbourhood of `distance`."""
        neighbours = _flat_offsets(vol.shape, _neighbour_offsets(distance))
        kernel(
            volume,
            counts,
            blocks,
            pad,
            *families,
            neighbours,
            pairs,
            steps,
            *tables,
        )

    # The NGTDM and the NGLDM in one pass, or one each for two distances
    passes = [(calc_ngtdm, calc_ngldm, ngtdm_distance if calc_ngtdm else ngldm_distance)]
    if calc_ngtdm and calc_ngldm and ngtdm_distance != ngldm_distance:
        passes = [(True, False, ngtdm_distance), (False, True, ngldm_distance)]
    if levels is None:
        ngtdm, ngldm, distance = passes.pop(0)
        run(vol, (calc_glcm, calc_glrlm, ngtdm, ngldm), distance)
    else:
        # The GLCM and the GLRLM count the volume of the occurring levels (1..rows); the
        # NGTDM and the NGLDM read the grey levels themselves
        lut = np.zeros(n_bins + 1, dtype=np.uint16)
        lut[levels + 1] = np.arange(1, rows + 1)
        run(lut[vol], (calc_glcm, calc_glrlm, False, False), 1)
    for ngtdm, ngldm, distance in passes:
        if ngtdm or ngldm:
            run(vol, (False, False, ngtdm, ngldm), distance)
    if not calc_glcm:
        glcm = np.zeros((13, n_bins, n_bins), dtype=np.uint64)
    elif n_threads == 1 and n_alt == 1:  # the one table of the serial twin
        glcm = glcm_t[0, 0, :, 1:, 1:].astype(np.uint64)
    else:
        glcm = np.ascontiguousarray(
            _thread_sum(glcm_t.reshape((-1,) + glcm_t.shape[2:]))[:, 1:, 1:]
        )
    if not calc_glrlm:
        glrlm = np.zeros((13, n_bins, 1), dtype=np.uint64)
    else:
        glrlm = _thread_sum(glrlm_t) if n_threads > 1 else glrlm_t[0].astype(np.uint64)
    if levels is not None:  # back to a row and a column for each of the n_bins levels
        if calc_glcm:
            glcm_all = np.zeros((n_tables, n_bins, n_bins), dtype=np.uint64)
            glcm_all[:, levels[:, None], levels[None, :]] = glcm
            glcm = glcm_all
        if calc_glrlm:
            glrlm_all = np.zeros((n_tables, n_bins, longest + 1), dtype=np.uint64)
            glrlm_all[:, levels, :] = glrlm
            glrlm = glrlm_all
    if n_threads > 1:  # the sums over the threads
        ngtdm_d_sum, ngtdm_n_sum = ngtdm_d.sum(axis=0), ngtdm_n.sum(axis=0)
        ngldm_sum = ngldm_t.sum(axis=0, dtype=np.uint64)
    else:  # one thread: its tables
        ngtdm_d_sum, ngtdm_n_sum = ngtdm_d[0], ngtdm_n[0]
        ngldm_sum = ngldm_t[0].astype(np.uint64)
    # s_i is the sum over the neighbour counts n of (sum of |g n - S|) / n, in a fixed order
    ngtdm_s = (ngtdm_d_sum[:, 1:] / np.arange(1, ngtdm_size)).sum(axis=1)
    return glcm, glrlm, ngtdm_s, ngtdm_n_sum.astype(np.float64), ngldm_sum


def calculate_all_texture_matrices(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    distance_mask: Optional[npt.NDArray[np.floating[Any]]] = None,
    ngldm_alpha: int = 0,
    calc_glcm: bool = True,
    calc_glrlm: bool = True,
    calc_ngtdm: bool = True,
    calc_ngldm: bool = True,
    calc_glszm: bool = True,
    calc_gldzm: bool = True,
    glcm_distance: int = 1,
    ngtdm_distance: int = 1,
    ngldm_distance: int = 1,
) -> dict[str, Any]:
    """
    Calculate all texture matrices (GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM) in an optimized single pass.

    This function serves as the computational backbone for texture analysis. It computes the raw
    matrices required to extract specific texture features. By aggregating these calculations,
    it minimizes the number of passes over the image data, significantly improving performance.

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
            Values should be integers in the range [1, n_bins].
        mask (npt.NDArray[np.floating[Any]]): The 3D mask array defining the Region of Interest (ROI).
            Must have the same shape as `data`. Nonzero values indicate ROI membership.
        n_bins (int): The number of grey levels used for discretization (e.g., 16, 32, 64).
            This determines the size of the resulting matrices.
        distance_mask (Optional[npt.NDArray[np.floating[Any]]]): Optional mask used to calculate the distance map for GLDZM.
            If None, `mask` is used. This allows calculating distances based on the morphological mask
            while analyzing intensities from the intensity mask (e.g., after outlier filtering).
        ngldm_alpha (int): The coarseness parameter α for NGLDM calculation. Two grey levels are
            considered dependent if their absolute difference is ≤ α. Default is 0 (exact match),
            which is the IBSI standard. Use α=1 for tolerance of ±1 grey level difference.
        calc_glcm (bool): Whether to compute the GLCM. Default True.
        calc_glrlm (bool): Whether to compute the GLRLM. Default True.
        calc_ngtdm (bool): Whether to compute the NGTDM. Default True.
        calc_ngldm (bool): Whether to compute the NGLDM. Default True.
        calc_glszm (bool): Whether to compute the GLSZM. Default True.
        calc_gldzm (bool): Whether to compute the GLDZM (skipping it also skips the
            distance transform). Default True.
        glcm_distance (int): The distance δ between the two voxels of a GLCM pair along
            each of the 13 directions (IBSI 1). Default 1.
        ngtdm_distance (int): The Chebyshev distance δ of the NGTDM neighbourhood. Default 1
            (the 26 neighbours).
        ngldm_distance (int): The Chebyshev distance δ of the NGLDM neighbourhood. Default 1.
            Disabled matrices are returned as zero-filled placeholders of minimal shape.

    Returns:
        dict[str, Any]: A dictionary containing the calculated texture matrices:
            - 'glcm' (npt.NDArray[np.floating[Any]]): Grey Level Co-occurrence Matrix. Shape: (n_dirs, n_bins, n_bins).
            - 'glrlm' (npt.NDArray[np.floating[Any]]): Grey Level Run Length Matrix. Shape: (n_dirs, n_bins, max_run_length).
            - 'ngtdm_s' (npt.NDArray[np.floating[Any]]): NGTDM Sum of absolute differences. Shape: (n_bins,).
            - 'ngtdm_n' (npt.NDArray[np.floating[Any]]): NGTDM Number of valid voxels. Shape: (n_bins,).
            - 'ngldm' (npt.NDArray[np.floating[Any]]): Neighbouring Grey Level Dependence Matrix. Shape: (n_bins, n_dependence).
            - 'glszm' (npt.NDArray[np.floating[Any]]): Grey Level Size Zone Matrix. Shape: (n_bins, max_zone_size).
            - 'gldzm' (npt.NDArray[np.floating[Any]]): Grey Level Distance Zone Matrix. Shape: (n_bins, max_distance).

    Raises:
        ValueError: a mask voxel lies outside `distance_mask`.

    Example:
        Calculate all texture matrices:

        ```python
        import numpy as np
        from pictologics.features.texture import calculate_all_texture_matrices

        # Create dummy data
        data = np.random.randint(1, 33, (50, 50, 50))
        mask = np.ones((50, 50, 50))

        # Calculate matrices
        matrices = calculate_all_texture_matrices(data, mask, n_bins=32)
        print(matrices['glcm'].shape)
        # (13, 32, 32)
        ```
    """
    return _texture_matrices(
        data,
        mask,
        n_bins,
        distance_mask,
        ngldm_alpha,
        calc_glcm,
        calc_glrlm,
        calc_ngtdm,
        calc_ngldm,
        calc_glszm,
        calc_gldzm,
        compact=False,
        glcm_distance=glcm_distance,
        ngtdm_distance=ngtdm_distance,
        ngldm_distance=ngldm_distance,
    )


def _texture_matrices(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    distance_mask: Optional[npt.NDArray[np.floating[Any]]] = None,
    ngldm_alpha: int = 0,
    calc_glcm: bool = True,
    calc_glrlm: bool = True,
    calc_ngtdm: bool = True,
    calc_ngldm: bool = True,
    calc_glszm: bool = True,
    calc_gldzm: bool = True,
    compact: bool = False,
    distance_map: Optional[npt.NDArray[Any]] = None,
    planar: Optional[tuple[bool, bool, bool]] = None,
    glcm_distance: int = 1,
    ngtdm_distance: int = 1,
    ngldm_distance: int = 1,
) -> dict[str, Any]:
    """Body of `calculate_all_texture_matrices`, with a private `compact` mode.

    `compact=True` returns smaller matrices that give the same features: 'glcm' and
    'glrlm' hold one table (the sum over the 13 directions) instead of 13, and
    'glszm_cells' (in place of 'glszm') holds only the non-zero GLSZM cells, as the
    (3, n_cells) array of `_glszm_cells`; it leaves out the matrices that are not asked
    for. It also returns 'roi', the bool ROI mask of the
    texture voxels, so the callers count ROI voxels without one more pass over the mask.
    The pipeline and `calculate_all_texture_features` use it. They crop their arrays to
    the ROI box before the call, so the compact mode does not crop again. With GLDZM, the
    compact mode also returns 'distance_map' (padded by one voxel, as the kernels use
    it), which a later call with the same masks can give back as `distance_map` to skip
    the distance transform.

    An ROI voxel with a grey level outside [1, n_bins] (for example 0, the bin of NaN)
    takes no part in any matrix: it counts as a voxel outside the ROI. `planar` names the
    axes of size 1 in the image (default: those of `data`; a compact caller gives them,
    because its box can be one slice of a 3D image). The GLCM and the GLRLM then use only
    the directions in the image, and the GLDZM distance map has no border along them.
    """
    if planar is None:
        planar = _planar_axes(data.shape) if not compact else (False, False, False)
    # Crop to ROI bounding box (union with distance_mask when provided) to reduce memory traffic.
    if compact:
        data_c, mask_c, distmask_c = data, mask, distance_mask
    else:
        data_c, mask_c, distmask_c = _maybe_crop_to_bbox(data, mask, distance_mask)

    # One pass over the mask gives the ROI. Label values are membership markers, not
    # weights. Row order also for a column-order mask, so the kernels compile once.
    roi = np.not_equal(mask_c, 0, order="C")
    extra: dict[str, Any] = {"roi": roi, "n_voxels": 0} if compact else {}
    if compact and calc_glcm:
        extra["ng_eff"] = 1

    # Fast exit for empty ROI. Checked on the cropped mask; mask_c can still be empty
    # when only distance_mask has nonzero voxels.
    glszm_key = "glszm_cells" if compact else "glszm"
    if not roi.any():
        return {
            "glcm": np.zeros((13, n_bins, n_bins), dtype=np.uint64),
            "glrlm": np.zeros((13, n_bins, 1), dtype=np.uint64),
            "ngtdm_s": np.zeros((n_bins,), dtype=np.float64),
            "ngtdm_n": np.zeros((n_bins,), dtype=np.float64),
            "ngldm": np.zeros((n_bins, _neighbourhood_size(ngldm_distance)), dtype=np.uint64),
            glszm_key: np.zeros((3, 0) if compact else (n_bins, 1), dtype=np.uint32),
            "gldzm": np.zeros((n_bins, 1), dtype=np.uint32),
            **extra,
        }

    span = np.empty((data_c.shape[0], 2), dtype=np.float64)
    vol, counts = _texture_volume(data_c, roi, n_bins, span)
    if compact:
        texture_voxels, roi_voxels = counts.sum(axis=0)
        if texture_voxels != roi_voxels:
            extra["roi"] = vol[1:-1, 1:-1, 1:-1] != 0  # the ROI without invalid levels
        # The count of the texture voxels and the GLCM Ng_eff (the span of their levels,
        # as roi_min_max gives it): the feature functions need no pass over the ROI
        extra["n_voxels"] = int(texture_voxels)
        if calc_glcm and texture_voxels:
            extra["ng_eff"] = int(span[:, 1].max() - span[:, 0].min() + 1)

    # 1. Local Features (GLCM, GLRLM, NGTDM, NGLDM). The compact mode leaves out the
    # matrices that are not asked for; the full mode gives zero placeholders.
    matrices: dict[str, Any] = {}
    if calc_glcm or calc_glrlm or calc_ngtdm or calc_ngldm:
        glcm, glrlm, ngtdm_s, ngtdm_n, ngldm = _local_matrices(
            vol,
            counts,
            n_bins,
            calc_glcm,
            calc_glrlm,
            calc_ngtdm,
            calc_ngldm,
            ngldm_alpha,
            merge_directions=compact,
            planar=planar,
            glcm_distance=glcm_distance,
            ngtdm_distance=ngtdm_distance,
            ngldm_distance=ngldm_distance,
        )
        matrices.update(glcm=glcm, glrlm=glrlm, ngtdm_s=ngtdm_s, ngtdm_n=ngtdm_n, ngldm=ngldm)
    elif not compact:
        matrices.update(
            glcm=np.zeros((13, n_bins, n_bins), dtype=np.uint64),
            glrlm=np.zeros((13, n_bins, 1), dtype=np.uint64),
            ngtdm_s=np.zeros((n_bins,), dtype=np.float64),
            ngtdm_n=np.zeros((n_bins,), dtype=np.float64),
            ngldm=np.zeros((n_bins, _neighbourhood_size(ngldm_distance)), dtype=np.uint64),
        )

    # 2. Zone Features (GLSZM, GLDZM). The zone kernel uses up `vol`, so it comes last.
    if calc_glszm or calc_gldzm:
        dist = _NO_DISTANCE
        if calc_gldzm:
            if distance_map is None:
                # Distance-to-border map of distance_mask, else of the ROI, with the image
                # border treated as an edge (padded like `vol`).
                d_roi = roi if distmask_c is None else np.not_equal(distmask_c, 0, order="C")
                if distmask_c is not None and np.any(roi & ~d_roi):
                    # A texture voxel outside the distance mask would get distance 0 and
                    # its whole zone would leave the GLDZM (the distance of a zone is 1 or
                    # more): the distance mask must hold the mask
                    outside = int(np.count_nonzero(roi & ~d_roi))
                    raise ValueError(
                        f"{outside} mask voxel{' lies' if outside == 1 else 's lie'} outside "
                        "distance_mask; the GLDZM needs the mask inside the distance mask."
                    )
                distance_map = _distance_map(d_roi, planar)
            dist = distance_map
            if compact:
                extra["distance_map"] = dist
        glszm, gldzm = _zone_matrices(
            vol, counts, dist, n_bins, calc_glszm, calc_gldzm, dense_glszm=not compact
        )
        matrices.update({glszm_key: glszm, "gldzm": gldzm})
    elif not compact:
        matrices.update(
            glszm=np.zeros((n_bins, 1), dtype=np.uint32),
            gldzm=np.zeros((n_bins, 1), dtype=np.uint32),
        )
    return {**matrices, **extra}


def _planar_axes(shape: tuple[int, ...]) -> tuple[bool, bool, bool]:
    """Which axes of a 3D image have size 1 (a one-slice image has one)."""
    return (shape[0] == 1, shape[1] == 1, shape[2] == 1)


def _one_matrix(
    data: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    n_bins: int,
    family: str,
    distance_mask: Optional[npt.NDArray[Any]] = None,
    ngldm_alpha: int = 0,
    planar: tuple[bool, bool, bool] = (False, False, False),
    distance: int = 1,
) -> dict[str, Any]:
    """The compact `_texture_matrices` of one family ('glcm', 'glrlm', 'ngtdm', 'ngldm',
    'glszm' or 'gldzm') for the standalone feature functions, which crop first (and give
    the axes of size 1 of the image as `planar`). `distance` is the GLCM, NGTDM or NGLDM
    distance of the family."""
    return _texture_matrices(
        data,
        mask,
        n_bins,
        distance_mask=distance_mask,
        ngldm_alpha=ngldm_alpha,
        planar=planar,
        glcm_distance=distance,
        ngtdm_distance=distance,
        ngldm_distance=distance,
        calc_glcm=family == "glcm",
        calc_glrlm=family == "glrlm",
        calc_ngtdm=family == "ngtdm",
        calc_ngldm=family == "ngldm",
        calc_glszm=family == "glszm",
        calc_gldzm=family == "gldzm",
        compact=True,
    )


# A smaller matrix keeps every row and column: finding the empty ones costs more than the
# features save
_OCCUPIED_MIN_CELLS = 1 << 10


def _occupied(
    matrix: npt.NDArray[Any], symmetric: bool = False
) -> tuple[npt.NDArray[Any], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """`matrix` without its rows and columns of zeros, and the 1-based numbers (grey
    levels, lengths) of the rows and columns that it keeps; a symmetric matrix keeps the
    same rows and columns. A feature adds only the cells that hold counts, so the
    features are those of the whole matrix, to the last digits (their sums add fewer
    zeros); a fixed FBS start, for example, leaves many low levels empty."""
    if matrix.size < _OCCUPIED_MIN_CELLS:
        return matrix, np.arange(1, matrix.shape[0] + 1), np.arange(1, matrix.shape[1] + 1)
    rows = np.flatnonzero(matrix.any(axis=1))
    cols = rows if symmetric else np.flatnonzero(matrix.any(axis=0))
    if rows.size < matrix.shape[0] or cols.size < matrix.shape[1]:
        matrix = matrix[np.ix_(rows, cols)]
    return matrix, rows + 1, cols + 1


@jit(nopython=True, cache=True)  # type: ignore
def _glcm_sums_numba(
    P: npt.NDArray[np.float64],
    levels: npt.NDArray[np.int64],
    ng_eff: float,
    p_diff: npt.NDArray[np.float64],
    p_sum: npt.NDArray[np.float64],
) -> tuple[float, ...]:
    """The GLCM sums that do not need the joint average, over the cells of P that hold
    counts: joint maximum, joint entropy, difference average, contrast, angular second
    moment, the four inverse differences, inverse variance and autocorrelation. Adds P to the difference bins `p_diff` (|i - j|) and the sum bins
    `p_sum` (i + j). Each row adds into its own sums first, so the rounding error grows
    with the row length, not with the count of cells."""
    n = P.shape[0]
    joint_max = 0.0
    total = np.zeros(10)
    for a in range(n):
        i = levels[a]
        r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = r10 = 0.0
        for b in range(n):
            p = P[a, b]
            if p == 0.0:
                continue
            j = levels[b]
            k = abs(i - j)
            k2 = k * k
            joint_max = max(joint_max, p)
            r1 -= p * math.log2(p)
            r2 += k * p
            r3 += k2 * p
            r4 += p * p
            r5 += p / (1 + k)
            r6 += p / (1 + k / ng_eff)
            r7 += p / (1 + k2)
            r8 += p / (1 + k2 / (ng_eff * ng_eff))
            if k != 0:
                r9 += p / k2
            r10 += j * p
            p_diff[k] += p
            p_sum[i + j] += p
        total[0] += r1
        total[1] += r2
        total[2] += r3
        total[3] += r4
        total[4] += r5
        total[5] += r6
        total[6] += r7
        total[7] += r8
        total[8] += r9
        total[9] += i * r10
    return (
        joint_max,
        total[0],
        total[1],
        total[2],
        total[3],
        total[4],
        total[5],
        total[6],
        total[7],
        total[8],
        total[9],
    )


@jit(nopython=True, cache=True)  # type: ignore
def _glcm_mu_sums_numba(
    P: npt.NDArray[np.float64], levels: npt.NDArray[np.int64], mu: float
) -> tuple[float, float, float, float, float]:
    """The GLCM sums around the joint average `mu`, over the cells of P that hold counts:
    joint variance, the correlation term and the cluster tendency, shade and prominence
    (s = i + j - 2 mu). Each row adds into its own sums first (see _glcm_sums_numba)."""
    n = P.shape[0]
    variance = correlation = tendency = shade = prominence = 0.0
    for a in range(n):
        di = levels[a] - mu
        r0 = r1 = r2 = r3 = r4 = 0.0
        for b in range(n):
            p = P[a, b]
            if p == 0.0:
                continue
            dj = levels[b] - mu
            s = di + dj
            s2 = s * s
            r0 += di * di * p
            r1 += di * dj * p
            r2 += s2 * p
            r3 += s2 * s * p
            r4 += s2 * s2 * p
        variance += r0
        correlation += r1
        tendency += r2
        shade += r3
        prominence += r4
    return variance, correlation, tendency, shade, prominence


# --- Feature sums in the order of numpy ---
# The feature kernels add their terms in the order of np.sum on a contiguous float64 array
# (numpy's pairwise sum: eight accumulators up to 128 terms, else two halves), with the
# same elementwise expressions and exact integer marginals. So they give the bits of the
# numpy expressions of the features, in one call for each family.


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _pairwise_block_numba(a: npt.NDArray[np.float64], lo: int, n: int) -> float:
    """The sum of a[lo : lo + n] (n <= 128) in the order of numpy's pairwise sum: one
    accumulator below 8 terms, else eight accumulators and the rest one by one."""
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return float(res)
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = lo + 8
    stop = lo + n - n % 8
    while i < stop:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < lo + n:
        res += a[i]
        i += 1
    return float(res)


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _np_sum_numba(a: npt.NDArray[np.float64], n: int) -> float:
    """np.sum of the float64 a[:n], with its bits: numpy adds blocks of up to 128 terms
    (_pairwise_block_numba) and splits a longer range into two halves (the first a
    multiple of 8 terms), whose sums it adds. A loop with a stack does the halves (numba
    cannot load a cached caller of a recursive function). The reduction starts at 0.0."""
    if n <= 128:
        return float(0.0 + _pairwise_block_numba(a, 0, n))
    starts = np.empty(64, dtype=np.int64)
    sizes = np.empty(64, dtype=np.int64)
    stage = np.empty(64, dtype=np.int64)  # 0: first half next, 1: second half next, 2: add
    sums = np.empty(64)
    starts[0] = 0
    sizes[0] = n
    stage[0] = 0
    depth = 1
    k = 0
    while depth > 0:
        f = depth - 1
        size = sizes[f]
        if size <= 128:
            sums[k] = _pairwise_block_numba(a, starts[f], size)
            k += 1
            depth -= 1
            continue
        half = size // 2
        half -= half % 8
        if stage[f] == 0:
            stage[f] = 1
            starts[depth] = starts[f]
            sizes[depth] = half
            stage[depth] = 0
            depth += 1
        elif stage[f] == 1:
            stage[f] = 2
            starts[depth] = starts[f] + half
            sizes[depth] = size - half
            stage[depth] = 0
            depth += 1
        else:
            k -= 1
            sums[k - 1] = sums[k - 1] + sums[k]
            depth -= 1
    return float(0.0 + sums[0])


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)  # type: ignore
def _count_matrix_features_numba(
    tables: npt.NDArray[np.uint64], skip: int, min_cells: int, out: npt.NDArray[np.float64]
) -> int:
    """The features of a count matrix (GLRLM, GLDZM, NGLDM) into `out`, as the numpy
    expressions of calculate_glrlm_features give them: the matrix is the sum of `tables`
    (n_tables, rows, columns) over the tables, without its first `skip` columns. From
    `min_cells` cells on, it keeps only the rows and the columns that hold counts (see
    _occupied). out: the 16 GLRLM features in their order (out[6], the percentage, stays
    for the caller) and the energy. Returns the total count (0: `out` is not written)."""
    n_t, r, c_all = tables.shape
    c = c_all - skip
    m = np.zeros((r, c), dtype=np.uint64)
    for table in range(n_t):
        for a in range(r):
            for b in range(c):
                m[a, b] += tables[table, a, b + skip]
    total = np.uint64(0)
    for a in range(r):
        for b in range(c):
            total += m[a, b]
    if total == 0:
        return 0
    rows = np.arange(1, r + 1)
    cols = np.arange(1, c + 1)
    if r * c >= min_cells:
        row_any = np.zeros(r, dtype=np.bool_)
        col_any = np.zeros(c, dtype=np.bool_)
        for a in range(r):
            for b in range(c):
                if m[a, b] != 0:
                    row_any[a] = True
                    col_any[b] = True
        rows = np.flatnonzero(row_any) + 1
        cols = np.flatnonzero(col_any) + 1
        if rows.size < r or cols.size < c:
            kept = np.empty((rows.size, cols.size), dtype=np.uint64)
            for a in range(rows.size):
                for b in range(cols.size):
                    kept[a, b] = m[rows[a] - 1, cols[b] - 1]
            m = kept
    r = rows.size
    c = cols.size
    n = r * c
    nf = np.float64(total)
    p = np.empty(n)
    t = np.empty(n)
    for a in range(r):
        for b in range(c):
            p[a * c + b] = np.float64(m[a, b]) / nf
    # The marginals of the counts: exact (uint64, as numpy sums them)
    si2 = np.uint64(0)
    for a in range(r):
        s = np.uint64(0)
        for b in range(c):
            s += m[a, b]
        si2 += s * s
    sj2 = np.uint64(0)
    for b in range(c):
        s = np.uint64(0)
        for a in range(r):
            s += m[a, b]
        sj2 += s * s
    for a in range(r):
        for b in range(c):
            t[a * c + b] = p[a * c + b] / np.float64(cols[b] * cols[b])
    out[0] = _np_sum_numba(t, n)
    for a in range(r):
        for b in range(c):
            t[a * c + b] = p[a * c + b] * np.float64(cols[b] * cols[b])
    out[1] = _np_sum_numba(t, n)
    out[2] = np.float64(si2) / nf
    out[3] = np.float64(si2) / np.float64(total * total)
    out[4] = np.float64(sj2) / nf
    out[5] = np.float64(sj2) / np.float64(total * total)
    out[6] = 0.0
    for a in range(r):
        for b in range(c):
            t[a * c + b] = np.float64(rows[a]) * p[a * c + b]
    mu_i = _np_sum_numba(t, n)
    for a in range(r):
        d = np.float64(rows[a]) - mu_i
        for b in range(c):
            t[a * c + b] = (d * d) * p[a * c + b]
    out[7] = _np_sum_numba(t, n)
    for a in range(r):
        for b in range(c):
            t[a * c + b] = np.float64(cols[b]) * p[a * c + b]
    mu_j = _np_sum_numba(t, n)
    for a in range(r):
        for b in range(c):
            d = np.float64(cols[b]) - mu_j
            t[a * c + b] = (d * d) * p[a * c + b]
    out[8] = _np_sum_numba(t, n)
    k = 0
    for i in range(n):
        if p[i] > 0:
            t[k] = p[i] * math.log2(p[i])
            k += 1
    out[9] = -_np_sum_numba(t, k)
    for a in range(r):
        i2 = rows[a] * rows[a]
        for b in range(c):
            t[a * c + b] = p[a * c + b] / np.float64(i2)
    out[10] = _np_sum_numba(t, n)
    for a in range(r):
        i2 = rows[a] * rows[a]
        for b in range(c):
            t[a * c + b] = p[a * c + b] * np.float64(i2)
    out[11] = _np_sum_numba(t, n)
    for a in range(r):
        i2 = rows[a] * rows[a]
        for b in range(c):
            t[a * c + b] = p[a * c + b] / np.float64(i2 * (cols[b] * cols[b]))
    out[12] = _np_sum_numba(t, n)
    for a in range(r):
        i2 = rows[a] * rows[a]
        for b in range(c):
            t[a * c + b] = p[a * c + b] * np.float64(i2) / np.float64(cols[b] * cols[b])
    out[13] = _np_sum_numba(t, n)
    for a in range(r):
        i2 = rows[a] * rows[a]
        for b in range(c):
            t[a * c + b] = p[a * c + b] * np.float64(cols[b] * cols[b]) / np.float64(i2)
    out[14] = _np_sum_numba(t, n)
    for a in range(r):
        i2 = rows[a] * rows[a]
        for b in range(c):
            t[a * c + b] = p[a * c + b] * np.float64(i2) * np.float64(cols[b] * cols[b])
    out[15] = _np_sum_numba(t, n)
    for i in range(n):
        t[i] = p[i] * p[i]
    out[16] = _np_sum_numba(t, n)
    return int(total)


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)  # type: ignore
def _glszm_features_numba(
    gl_idx: npt.NDArray[np.int64],
    sz_idx: npt.NDArray[np.int64],
    c: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> float:
    """The 16 GLSZM features of the non-zero cells (0-based grey level and size, count, in
    row order) into `out`, with the bits of the numpy expressions of before (pairwise
    sums; np.bincount adds the cells in their order). out[6], the zone percentage, stays
    for the caller. Returns the zone count."""
    n = c.shape[0]
    p = np.empty(n)
    t = np.empty(n)
    total = _np_sum_numba(c, n)
    for k in range(n):
        p[k] = c[k] / total
    for k in range(n):
        j = np.float64(sz_idx[k] + 1)
        t[k] = p[k] / (j * j)
    out[0] = _np_sum_numba(t, n)
    for k in range(n):
        j = np.float64(sz_idx[k] + 1)
        t[k] = p[k] * (j * j)
    out[1] = _np_sum_numba(t, n)
    # np.bincount with weights: the cells add in their order
    s_i = np.zeros(gl_idx.max() + 1)
    for k in range(n):
        s_i[gl_idx[k]] += c[k]
    for a in range(s_i.size):
        s_i[a] = s_i[a] * s_i[a]
    sum_si2 = _np_sum_numba(s_i, s_i.size)
    out[2] = sum_si2 / total
    out[3] = sum_si2 / (total * total)
    s_j = np.zeros(sz_idx.max() + 1)
    for k in range(n):
        s_j[sz_idx[k]] += c[k]
    for b in range(s_j.size):
        s_j[b] = s_j[b] * s_j[b]
    sum_sj2 = _np_sum_numba(s_j, s_j.size)
    out[4] = sum_sj2 / total
    out[5] = sum_sj2 / (total * total)
    out[6] = 0.0
    for k in range(n):
        t[k] = np.float64(gl_idx[k] + 1) * p[k]
    mu_i = _np_sum_numba(t, n)
    for k in range(n):
        d = np.float64(gl_idx[k] + 1) - mu_i
        t[k] = (d * d) * p[k]
    out[7] = _np_sum_numba(t, n)
    for k in range(n):
        t[k] = np.float64(sz_idx[k] + 1) * p[k]
    mu_j = _np_sum_numba(t, n)
    for k in range(n):
        d = np.float64(sz_idx[k] + 1) - mu_j
        t[k] = (d * d) * p[k]
    out[8] = _np_sum_numba(t, n)
    for k in range(n):
        t[k] = p[k] * np.log2(p[k])
    out[9] = -_np_sum_numba(t, n)
    for k in range(n):
        i = np.float64(gl_idx[k] + 1)
        t[k] = p[k] / (i * i)
    out[10] = _np_sum_numba(t, n)
    for k in range(n):
        i = np.float64(gl_idx[k] + 1)
        t[k] = p[k] * (i * i)
    out[11] = _np_sum_numba(t, n)
    for k in range(n):
        i = np.float64(gl_idx[k] + 1)
        j = np.float64(sz_idx[k] + 1)
        t[k] = p[k] / ((i * i) * (j * j))
    out[12] = _np_sum_numba(t, n)
    for k in range(n):
        i = np.float64(gl_idx[k] + 1)
        j = np.float64(sz_idx[k] + 1)
        t[k] = p[k] * (i * i) / (j * j)
    out[13] = _np_sum_numba(t, n)
    for k in range(n):
        i = np.float64(gl_idx[k] + 1)
        j = np.float64(sz_idx[k] + 1)
        t[k] = p[k] * (j * j) / (i * i)
    out[14] = _np_sum_numba(t, n)
    for k in range(n):
        i = np.float64(gl_idx[k] + 1)
        j = np.float64(sz_idx[k] + 1)
        t[k] = p[k] * (i * i) * (j * j)
    out[15] = _np_sum_numba(t, n)
    return float(total)


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)  # type: ignore
def _glcm_features_numba(
    glcm: npt.NDArray[np.uint64],
    n_bins: int,
    ng_eff: float,
    min_cells: int,
    out: npt.NDArray[np.float64],
) -> int:
    """The 25 GLCM features of the count tables `glcm` (n_tables, n, n) into `out`, in the
    order and with the bits of calculate_glcm_features (its numpy path): the symmetric sum
    over the tables, the levels that hold counts (from `min_cells` cells on, see
    _occupied), P, the sums of _glcm_sums_numba and _glcm_mu_sums_numba, the difference
    and the sum bins and the marginal entropy. Returns the total count (0: `out` is not
    written)."""
    n_t, n, _ = glcm.shape
    tab = np.zeros((n, n), dtype=np.uint64)
    for d in range(n_t):
        for a in range(n):
            for b in range(n):
                tab[a, b] += glcm[d, a, b]
    sym = np.empty((n, n), dtype=np.uint64)
    total = np.uint64(0)
    for a in range(n):
        for b in range(n):
            v = tab[a, b] + tab[b, a]
            sym[a, b] = v
            total += v
    if total == 0:
        return 0
    if n * n < min_cells:
        levels = np.arange(1, n + 1)
    else:
        occupied = np.zeros(n, dtype=np.bool_)
        for a in range(n):
            for b in range(n):
                if sym[a, b] != 0:
                    occupied[a] = True
                    break
        levels = np.flatnonzero(occupied) + 1
    m = levels.size
    totalf = np.float64(total)
    p = np.empty((m, m))
    row_counts = np.empty(m, dtype=np.uint64)
    for a in range(m):
        ia = levels[a] - 1
        rc = np.uint64(0)
        for b in range(m):
            v = sym[ia, levels[b] - 1]
            rc += v
            p[a, b] = np.float64(v) / totalf
        row_counts[a] = rc
    nb = max(n_bins, levels[m - 1])
    p_diff = np.zeros(nb)
    p_sum = np.zeros(2 * nb + 1)
    sums = _glcm_sums_numba(p, levels, ng_eff, p_diff, p_sum)
    # The joint average from the integer counts (exact: both stay below 2^53)
    num = np.int64(0)
    for a in range(m):
        num += levels[a] * np.int64(row_counts[a])
    joint_average = np.float64(num) / np.float64(total)
    variance, correlation, tendency, shade, prominence = _glcm_mu_sums_numba(
        p, levels, joint_average
    )
    t = np.empty(2 * nb + 1)
    for k in range(nb):
        dk = np.float64(k) - sums[2]
        t[k] = (dk * dk) * p_diff[k]
    difference_variance = _np_sum_numba(t, nb)
    j = 0
    for k in range(nb):
        if p_diff[k] > 0:
            t[j] = p_diff[k] * math.log2(p_diff[k])
            j += 1
    difference_entropy = -_np_sum_numba(t, j)
    ns = 2 * nb - 1
    for k in range(ns):
        t[k] = np.float64(k + 2) * p_sum[k + 2]
    sum_average = _np_sum_numba(t, ns)
    for k in range(ns):
        dk = np.float64(k + 2) - sum_average
        t[k] = (dk * dk) * p_sum[k + 2]
    sum_variance = _np_sum_numba(t, ns)
    j = 0
    for k in range(ns):
        if p_sum[k + 2] > 0:
            t[j] = p_sum[k + 2] * math.log2(p_sum[k + 2])
            j += 1
    sum_entropy = -_np_sum_numba(t, j)
    # The marginal entropy from the exact row counts
    j = 0
    for a in range(m):
        px = np.float64(row_counts[a]) / totalf
        if px > 0:
            t[j] = px * math.log2(px)
            j += 1
    hx = -_np_sum_numba(t, j)
    hxy = sums[1]
    out[0] = sums[0]
    out[1] = joint_average
    out[2] = variance
    out[3] = hxy
    out[4] = sums[2]
    out[5] = difference_variance
    out[6] = difference_entropy
    out[7] = sum_average
    out[8] = sum_variance
    out[9] = sum_entropy
    out[10] = sums[4]
    out[11] = sums[3]
    out[12] = sums[2]
    out[13] = sums[5]
    out[14] = sums[6]
    out[15] = sums[7]
    out[16] = sums[8]
    out[17] = sums[9]
    out[18] = correlation / variance if variance != 0 else 1.0
    out[19] = sums[10]
    out[20] = tendency
    out[21] = shade
    out[22] = prominence
    out[23] = (hxy - 2 * hx) / hx if hx != 0 else np.nan
    out[24] = np.sqrt(1 - np.exp(-2 * (2 * hx - hxy)))
    return int(total)


_GLCM_KEYS = (
    "joint_maximum_GYBY", "joint_average_60VM", "joint_variance_UR99", "joint_entropy_TU9B",
    "difference_average_TF7R", "difference_variance_D3YU", "difference_entropy_NTRS",
    "sum_average_ZGXS", "sum_variance_OEEB", "sum_entropy_P6QZ", "angular_second_moment_8ZQL",
    "contrast_ACUI", "dissimilarity_8S9J", "inverse_difference_IB1Z",
    "normalised_inverse_difference_NDRX", "inverse_difference_moment_WF0Z",
    "normalised_inverse_difference_moment_1QCO", "inverse_variance_E8JP", "correlation_NI2N",
    "autocorrelation_QWB0", "cluster_tendency_DG8W", "cluster_shade_7NFM",
    "cluster_prominence_AE86", "information_correlation_1_R8DG", "information_correlation_2_JN9H",
)  # fmt: skip
_GLRLM_KEYS = (
    "short_runs_emphasis_22OV", "long_runs_emphasis_W4KF", "grey_level_non_uniformity_R5YN",
    "normalised_grey_level_non_uniformity_OVBL", "run_length_non_uniformity_W92Y",
    "normalised_run_length_non_uniformity_IC23", "run_percentage_9ZK5",
    "grey_level_variance_8CE5", "run_length_variance_SXLW", "run_entropy_HJ9O",
    "low_grey_level_run_emphasis_V3SW", "high_grey_level_run_emphasis_G3QZ",
    "short_run_low_grey_level_emphasis_HTZT", "short_run_high_grey_level_emphasis_GD3A",
    "long_run_low_grey_level_emphasis_IVPO", "long_run_high_grey_level_emphasis_3KUM",
)  # fmt: skip
_GLDZM_KEYS = (
    "small_distance_emphasis_0GBI", "large_distance_emphasis_MB4I", "grey_level_non_uniformity_VFT7",
    "normalised_grey_level_non_uniformity_7HP3", "zone_distance_non_uniformity_V294",
    "normalised_zone_distance_non_uniformity_IATH", "zone_percentage_VIWW",
    "grey_level_variance_QK93", "zone_distance_variance_7WT1", "zone_distance_entropy_GBDU",
    "low_grey_level_zone_emphasis_S1RA", "high_grey_level_zone_emphasis_K26C",
    "small_distance_low_grey_level_emphasis_RUVG", "small_distance_high_grey_level_emphasis_DKNJ",
    "large_distance_low_grey_level_emphasis_A7WM", "large_distance_high_grey_level_emphasis_KLTH",
)  # fmt: skip
# The NGLDM features in their order, with the index of each in `out` of
# _count_matrix_features_numba
_NGLDM_KEYS = (
    ("low_dependence_emphasis_SODN", 0), ("high_dependence_emphasis_IMOQ", 1),
    ("low_grey_level_count_emphasis_TL9H", 10), ("high_grey_level_count_emphasis_OAE7", 11),
    ("low_dependence_low_grey_level_emphasis_EQ3F", 12),
    ("low_dependence_high_grey_level_emphasis_JA6D", 13),
    ("high_dependence_low_grey_level_emphasis_NBZI", 14),
    ("high_dependence_high_grey_level_emphasis_9QMG", 15),
    ("grey_level_non_uniformity_FP8K", 2), ("normalised_grey_level_non_uniformity_5SPA", 3),
    ("dependence_count_non_uniformity_Z87G", 4),
    ("normalised_dependence_count_non_uniformity_OKJI", 5),
    ("dependence_count_percentage_6XV8", 6), ("grey_level_variance_1PFV", 7),
    ("dependence_count_variance_DNX2", 8), ("dependence_count_entropy_FCBV", 9),
    ("dependence_count_energy_CAS9", 16),
)  # fmt: skip
_GLSZM_KEYS = (
    "small_zone_emphasis_P001", "large_zone_emphasis_48P8", "grey_level_non_uniformity_JNSA",
    "normalised_grey_level_non_uniformity_Y1RO", "zone_size_non_uniformity_4JP3",
    "normalised_zone_size_non_uniformity_VB3A", "zone_percentage_P30P",
    "grey_level_variance_BYLV", "zone_size_variance_3NSA", "zone_size_entropy_GU8N",
    "low_grey_level_zone_emphasis_XMSY", "high_grey_level_zone_emphasis_5GN9",
    "small_zone_low_grey_level_emphasis_5RAI", "small_zone_high_grey_level_emphasis_HW1V",
    "large_zone_low_grey_level_emphasis_YH51", "large_zone_high_grey_level_emphasis_J17V",
)  # fmt: skip


def _ng_eff(data: npt.NDArray[Any], mask: npt.NDArray[Any]) -> int:
    """Ng_eff of the GLCM: the span of the grey levels of the ROI (1 for no level)."""
    roi_span = roi_min_max(data, mask)
    return int(roi_span[1] - roi_span[0] + 1) if roi_span is not None else 1


def _glcm_features(glcm: npt.NDArray[Any], n_bins: int, ng_eff: int) -> dict[str, float]:
    """The GLCM features of the unsigned count tables `glcm` (n_tables, n, n), or of one
    (n, n) table."""
    out = np.empty(len(_GLCM_KEYS))
    tables = np.ascontiguousarray(glcm, dtype=np.uint64).reshape((-1,) + glcm.shape[-2:])
    if _glcm_features_numba(tables, n_bins, float(ng_eff), _OCCUPIED_MIN_CELLS, out) == 0:
        return {}
    return dict(zip(_GLCM_KEYS, out, strict=True))


def _count_features(tables: npt.NDArray[Any], skip: int) -> tuple[int, npt.NDArray[np.float64]]:
    """The total count and the `out` of _count_matrix_features_numba for the unsigned
    count matrix (rows, columns) or tables (n_tables, rows, columns)."""
    counts = np.ascontiguousarray(tables, dtype=np.uint64).reshape((-1,) + tables.shape[-2:])
    out = np.empty(17)
    return _count_matrix_features_numba(counts, skip, _OCCUPIED_MIN_CELLS, out), out


def _glrlm_features(
    glrlm: npt.NDArray[Any],
    mask: Optional[npt.NDArray[Any]],
    n_directions: Optional[int],
    n_voxels: Optional[int] = None,
) -> dict[str, float]:
    """The GLRLM features of the unsigned run tables; n_voxels: the ROI voxel count (None:
    count `mask`)."""
    n_runs, out = _count_features(glrlm, 1)
    if n_runs == 0:
        return {}
    if n_voxels is None:
        n_voxels = _roi_voxel_count(cast(npt.NDArray[Any], mask))
    out[6] = np.uint64(n_runs) / (n_voxels * (n_directions or 13))
    return dict(zip(_GLRLM_KEYS, out[:16], strict=True))


def _gldzm_features(
    gldzm: npt.NDArray[Any], mask: Optional[npt.NDArray[Any]], n_voxels: Optional[int] = None
) -> dict[str, float]:
    """The GLDZM features of the unsigned matrix (see _glrlm_features)."""
    n_zones, out = _count_features(gldzm, 0)
    if n_zones == 0:
        return {}
    if n_voxels is None:
        n_voxels = _roi_voxel_count(cast(npt.NDArray[Any], mask))
    out[6] = np.uint64(n_zones) / n_voxels
    return dict(zip(_GLDZM_KEYS, out[:16], strict=True))


def _ngldm_features(
    ngldm: npt.NDArray[Any], mask: Optional[npt.NDArray[Any]], n_voxels: Optional[int] = None
) -> dict[str, float]:
    """The NGLDM features of the unsigned matrix (see _glrlm_features)."""
    n_s, out = _count_features(ngldm, 0)
    if n_s == 0:
        return {}
    if n_voxels is None:
        n_voxels = _roi_voxel_count(cast(npt.NDArray[Any], mask))
    out[6] = np.uint64(n_s) / n_voxels
    return {key: out[i] for key, i in _NGLDM_KEYS}


def calculate_glcm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    glcm_matrix: Optional[npt.NDArray[np.floating[Any]]] = None,
    glcm_distance: int = 1,
) -> dict[str, float]:
    r"""
    Calculate Grey Level Co-occurrence Matrix (GLCM) features.

    The GLCM describes the second-order statistical distribution of grey levels in the ROI.
    It counts how often pairs of grey levels occur at a specific distance and direction.
    This implementation computes features based on the 3D merged GLCM (averaged over all 13 directions),
    making the features rotationally invariant.

    **IBSI Reference**: Section 3.6 (Grey Level Co-occurrence Based Features).

    **Mathematical Definition**:
    Let $P(i,j)$ be the co-occurrence matrix, where $i$ and $j$ are grey levels.
    The matrix is normalized such that $\sum_{i,j} P(i,j) = 1$.

    **Calculated Features**:
    *   Joint Maximum (GYBY)
    *   Joint Average (60VM)
    *   Joint Variance (UR99)
    *   Joint Entropy (TU9B)
    *   Difference Average (TF7R)
    *   Difference Variance (D3YU)
    *   Difference Entropy (NTRS)
    *   Sum Average (ZGXS)
    *   Sum Variance (OEEB)
    *   Sum Entropy (P6QZ)
    *   Angular Second Moment (8ZQL)
    *   Contrast (ACUI)
    *   Dissimilarity (8S9J)
    *   Inverse Difference (IB1Z)
    *   Normalised Inverse Difference (NDRX)
    *   Inverse Difference Moment (WF0Z)
    *   Normalised Inverse Difference Moment (1QCO)
    *   Inverse Variance (E8JP)
    *   Correlation (NI2N)
    *   Autocorrelation (QWB0)
    *   Cluster Tendency (DG8W)
    *   Cluster Shade (7NFM)
    *   Cluster Prominence (AE86)
    *   Information Correlation 1 (R8DG)
    *   Information Correlation 2 (JN9H)

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
        mask (npt.NDArray[np.floating[Any]]): The 3D mask array defining the ROI. Nonzero values indicate ROI membership.
        n_bins (int): The number of grey levels.
        glcm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated GLCM matrix. If provided, `data` and `mask`
            are ignored for matrix calculation, but `data` is still used for `Ng` estimation if needed.
            If None, the matrix is calculated from scratch.
        glcm_distance (int): The distance δ between the two voxels of a pair along each
            of the 13 directions (IBSI 1), for a matrix calculated from scratch. Default 1.

    Returns:
        dict[str, float]: A dictionary of calculated GLCM features, keyed by their name and IBSI code.
            Example keys: 'joint_maximum_GYBY', 'contrast_ACUI', 'correlation_NI2N'.

    Example:
        ```python
        import numpy as np
        from pictologics.features.texture import calculate_glcm_features

        rng = np.random.default_rng(0)
        data = rng.integers(1, 33, (20, 20, 20)).astype(np.float64)
        mask = np.ones((20, 20, 20), dtype=np.uint8)
        features = calculate_glcm_features(data, mask, n_bins=32)
        print(round(features["contrast_ACUI"], 1))
        ```
    """
    if glcm_matrix is None:
        # Standalone path: crop to the ROI bbox and rebind, so the kernel and the
        # Ng_eff scan below run on the cropped arrays instead of the full volume.
        planar = _planar_axes(data.shape)
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        matrices = _one_matrix(data, mask, n_bins, "glcm", planar=planar, distance=glcm_distance)
        return _glcm_features(matrices["glcm"], n_bins, matrices["ng_eff"])
    glcm = glcm_matrix
    if glcm.dtype.kind == "u":  # counts: the kernel
        if not glcm.any():
            return {}
        return _glcm_features(glcm, n_bins, _ng_eff(data, mask))

    # Merge (Sum) -> IAZD
    glcm_sum = np.sum(glcm, axis=0)
    glcm_sym = glcm_sum + glcm_sum.T

    # Normalize
    total_sum = np.sum(glcm_sym)
    if total_sum == 0:
        return {}
    glcm_sym, levels, _ = _occupied(glcm_sym, symmetric=True)

    P = glcm_sym / total_sum

    # Ng_eff is the ROI grey-level span
    Ng_eff = _ng_eff(data, mask)

    # The sums over the cells that hold counts, in two serial passes (_glcm_sums_numba,
    # _glcm_mu_sums_numba); levels: the 1-based grey levels of the rows and columns of P
    levels = levels.astype(np.int64, copy=False)
    # The bins hold every level of P (a given matrix can hold more levels than n_bins)
    n_bins = max(n_bins, int(levels[-1]))
    p_diff = np.zeros(n_bins)
    p_sum_full = np.zeros(2 * n_bins + 1)
    (
        joint_max,
        joint_entropy,
        difference_average,
        contrast,
        angular_second_moment,
        inverse_difference,
        normalised_inverse_difference,
        inverse_difference_moment,
        normalised_inverse_difference_moment,
        inverse_variance,
        autocorrelation,
    ) = _glcm_sums_numba(P, levels, float(Ng_eff), p_diff, p_sum_full)
    # The joint average from the integer counts: exact to half a unit in the last place, as
    # the cluster shade (whose terms cancel) is sensitive to it
    row_counts = glcm_sym.sum(axis=1)
    joint_average = int(levels @ row_counts.astype(np.int64)) / int(total_sum)
    mu = joint_average
    joint_variance, correlation_term, cluster_tendency, cluster_shade, cluster_prominence = (
        _glcm_mu_sums_numba(P, levels, mu)
    )

    features = {}
    features["joint_maximum_GYBY"] = joint_max
    features["joint_average_60VM"] = joint_average
    features["joint_variance_UR99"] = joint_variance
    features["joint_entropy_TU9B"] = joint_entropy
    features["difference_average_TF7R"] = difference_average

    # Difference Variance - D3YU and Difference Entropy - NTRS, from the difference bins
    dk = np.arange(n_bins) - difference_average
    features["difference_variance_D3YU"] = np.sum(dk * dk * p_diff)
    mask_pd = p_diff > 0
    features["difference_entropy_NTRS"] = -np.sum(p_diff[mask_pd] * np.log2(p_diff[mask_pd]))

    # Sum Average - ZGXS, Sum Variance - OEEB and Sum Entropy - P6QZ, from the sum bins
    p_sum = p_sum_full[2:]
    k_vals_sum = np.arange(2, 2 * n_bins + 1)
    features["sum_average_ZGXS"] = np.sum(k_vals_sum * p_sum)
    dk = k_vals_sum - features["sum_average_ZGXS"]
    features["sum_variance_OEEB"] = np.sum(dk * dk * p_sum)
    mask_ps = p_sum > 0
    features["sum_entropy_P6QZ"] = -np.sum(p_sum[mask_ps] * np.log2(p_sum[mask_ps]))

    features["angular_second_moment_8ZQL"] = angular_second_moment
    features["contrast_ACUI"] = contrast
    features["dissimilarity_8S9J"] = difference_average
    features["inverse_difference_IB1Z"] = inverse_difference
    features["normalised_inverse_difference_NDRX"] = normalised_inverse_difference
    features["inverse_difference_moment_WF0Z"] = inverse_difference_moment
    features["normalised_inverse_difference_moment_1QCO"] = normalised_inverse_difference_moment
    features["inverse_variance_E8JP"] = inverse_variance
    if joint_variance != 0:
        features["correlation_NI2N"] = correlation_term / joint_variance
    else:
        features["correlation_NI2N"] = 1.0  # Or NaN? IBSI doesn't specify for 0 variance.
    features["autocorrelation_QWB0"] = autocorrelation
    features["cluster_tendency_DG8W"] = cluster_tendency
    features["cluster_shade_7NFM"] = cluster_shade
    features["cluster_prominence_AE86"] = cluster_prominence

    # Information Correlation 1 - R8DG and 2 - JN9H
    HXY = joint_entropy
    # The marginal from the counts: exact, so it does not depend on the order of the sums
    p_x = row_counts / total_sum
    mask_px = p_x > 0
    HX = -np.sum(p_x[mask_px] * np.log2(p_x[mask_px]))
    # P is symmetric, so p_y = p_x, and Σ_j P_ij = p_i. The two entropies of the IBSI
    # definitions have closed forms (the sums run over the cells that hold counts):
    #   HXY1 = -Σ_ij P_ij log2(p_i p_j) = -Σ_i p_i log2 p_i - Σ_j p_j log2 p_j = 2 HX
    #   HXY2 = -Σ_ij p_i p_j log2(p_i p_j) = 2 HX Σ_j p_j = 2 HX
    if HX != 0:
        features["information_correlation_1_R8DG"] = (HXY - 2 * HX) / HX
    else:
        features["information_correlation_1_R8DG"] = np.nan
    features["information_correlation_2_JN9H"] = np.sqrt(1 - np.exp(-2 * (2 * HX - HXY)))
    return features


def calculate_glrlm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    glrlm_matrix: Optional[npt.NDArray[np.floating[Any]]] = None,
    n_directions: Optional[int] = None,
) -> dict[str, float]:
    """
    Calculate Grey Level Run Length Matrix (GLRLM) features.

    The GLRLM quantifies grey level runs, which are defined as the length in number of pixels,
    of consecutive pixels that have the same grey level value.
    This implementation computes features based on the 3D merged GLRLM (averaged over all 13 directions).

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
        mask (npt.NDArray[np.floating[Any]]): The 3D mask array defining the ROI. Nonzero values indicate ROI membership.
        n_bins (int): The number of grey levels.
        glrlm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated GLRLM matrix.
        n_directions (Optional[int]): The number of directions in a merged (one-table)
            `glrlm_matrix`, for the run percentage: 13 in 3D, 4 for a one-slice image.
            None: 13, or for 13 direction tables the number of them that hold runs.

    Returns:
        dict[str, float]: A dictionary of calculated GLRLM features.
            Example keys: 'short_runs_emphasis_22OV', 'grey_level_non_uniformity_R5YN'.
    """
    if glrlm_matrix is None:
        # Standalone path: crop to the ROI bbox and rebind, so the kernel and the
        # run-percentage voxel count below run on the cropped arrays.
        planar = _planar_axes(data.shape)
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        matrices = _one_matrix(data, mask, n_bins, "glrlm", planar=planar)
        return _glrlm_features(
            matrices["glrlm"], None, _directions(planar).size, n_voxels=matrices["n_voxels"]
        )
    glrlm = glrlm_matrix
    if n_directions is None and glrlm.shape[0] == 13:
        # The directions of a one-slice image that leave it have no runs at all
        n_directions = int(np.count_nonzero(glrlm.reshape(13, -1).any(axis=1)))
    if glrlm.dtype.kind == "u":  # counts: the kernel
        return _glrlm_features(glrlm, mask, n_directions)

    # Merge (Sum) -> IAZD
    glrlm_sum = np.sum(glrlm, axis=0)

    # Remove length 0 (column 0)
    glrlm = glrlm_sum[:, 1:]

    N_runs = np.sum(glrlm)
    if N_runs == 0:
        return {}

    glrlm, rows, cols = _occupied(glrlm)
    P = glrlm / N_runs

    # The 1-based numbers of the rows and columns that hold counts
    I, J = np.meshgrid(rows, cols, indexing="ij")  # noqa: E741
    I2 = I**2
    J2 = J**2

    features = {}

    # Short Run Emphasis (SRE) - 22OV
    features["short_runs_emphasis_22OV"] = np.sum(P / J2)

    # Long Run Emphasis (LRE) - W4KF
    features["long_runs_emphasis_W4KF"] = np.sum(P * J2)

    # Grey Level Non-Uniformity (GLNU) - R5YN
    s_i = np.sum(glrlm, axis=1)
    features["grey_level_non_uniformity_R5YN"] = np.sum(s_i**2) / N_runs

    # Normalised Grey Level Non-Uniformity (GLNN) - OVBL
    features["normalised_grey_level_non_uniformity_OVBL"] = np.sum(s_i**2) / (N_runs**2)

    # Run Length Non-Uniformity (RLNU) - W92Y
    s_j = np.sum(glrlm, axis=0)
    features["run_length_non_uniformity_W92Y"] = np.sum(s_j**2) / N_runs

    # Normalised Run Length Non-Uniformity (RLNN) - IC23
    features["normalised_run_length_non_uniformity_IC23"] = np.sum(s_j**2) / (N_runs**2)

    # Run Percentage (RP) - 9ZK5
    n_voxels = _roi_voxel_count(mask)
    features["run_percentage_9ZK5"] = N_runs / (n_voxels * (n_directions or 13))

    # Grey Level Variance (GLV) - 8CE5
    mu_i = np.sum(I * P)
    features["grey_level_variance_8CE5"] = np.sum(((I - mu_i) ** 2) * P)

    # Run Length Variance (RLV) - SXLW
    mu_j = np.sum(J * P)
    features["run_length_variance_SXLW"] = np.sum(((J - mu_j) ** 2) * P)

    # Run Entropy (RE) - HJ9O
    mask_p = P > 0
    features["run_entropy_HJ9O"] = -np.sum(P[mask_p] * np.log2(P[mask_p]))

    # Low Grey Level Run Emphasis (LGLRE) - V3SW
    features["low_grey_level_run_emphasis_V3SW"] = np.sum(P / I2)

    # High Grey Level Run Emphasis (HGLRE) - G3QZ
    features["high_grey_level_run_emphasis_G3QZ"] = np.sum(P * I2)

    # Short Run Low Grey Level Emphasis (SRLGLE) - HTZT
    features["short_run_low_grey_level_emphasis_HTZT"] = np.sum(P / (I2 * J2))

    # Short Run High Grey Level Emphasis (SRHGLE) - GD3A
    features["short_run_high_grey_level_emphasis_GD3A"] = np.sum(P * I2 / J2)

    # Long Run Low Grey Level Emphasis (LRLGLE) - IVPO
    features["long_run_low_grey_level_emphasis_IVPO"] = np.sum(P * J2 / I2)

    # Long Run High Grey Level Emphasis (LRHGLE) - 3KUM
    features["long_run_high_grey_level_emphasis_3KUM"] = np.sum(P * I2 * J2)

    return features


# --- Combined Zone Features Kernel ---

# The GLSZM cell builder counts zones in a small dense table of about this many cells
# (grey levels x zone sizes). The few larger zones are sorted instead. Above 1,024 grey
# levels the table has more cells, as the small table sends most zones to the sort there.
_GLSZM_DENSE_CELLS = 1 << 14
_GLSZM_MANY_LEVELS = 1024
_GLSZM_MANY_LEVEL_CELLS = 1 << 18


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _glszm_cells(
    zone_gl: npt.NDArray[np.int32],
    zone_size: npt.NDArray[np.int32],
    n_bins: int,
    dense_cells: int,
) -> npt.NDArray[np.uint32]:
    """Non-zero GLSZM cells from the grey level (1-based) and size of every zone.

    Returns a (3, n_cells) uint32 array: grey level index, size index (both 0-based) and
    count. The cells come in row-major order: the same cells, in the same order, that
    np.nonzero gives on the dense (n_bins, largest zone) GLSZM. That dense matrix is
    mostly zeros (one large zone makes it very wide), so this function does not make it.
    """
    n = zone_gl.shape[0]
    max_sz = 0
    for t in range(n):
        if zone_size[t] > max_sz:
            max_sz = zone_size[t]
    width = min(max_sz, max(1, dense_cells // n_bins))
    dense = np.zeros((n_bins, width), dtype=np.int64)
    n_large = 0
    for t in range(n):
        if zone_size[t] <= width:
            dense[zone_gl[t] - 1, zone_size[t] - 1] += 1
        else:
            n_large += 1
    # Zones wider than the table: sort a (grey level, size) key, so equal cells touch.
    stride = np.int64(max_sz) + 1
    large = np.empty(n_large, dtype=np.int64)
    k = 0
    for t in range(n):
        if zone_size[t] > width:
            large[k] = (np.int64(zone_gl[t]) - 1) * stride + (zone_size[t] - 1)
            k += 1
    large.sort()
    cells: npt.NDArray[np.uint32] = _glszm_cells_of(dense, large, stride)
    return cells


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _glszm_cells_of(
    dense: npt.NDArray[np.int64], large: npt.NDArray[np.int64], stride: int
) -> npt.NDArray[np.uint32]:
    """The (3, n_cells) cells of `_glszm_cells` from the dense (n_bins, width) table of the
    small zones and the sorted keys (grey level index * stride + size index) of the large
    zones."""
    n_bins, width = dense.shape
    n_large = large.shape[0]
    n_cells = 0
    for g in range(n_bins):
        for s in range(width):
            if dense[g, s] > 0:
                n_cells += 1
    for k in range(n_large):
        if k == 0 or large[k] != large[k - 1]:
            n_cells += 1
    cells = np.empty((3, n_cells), dtype=np.uint32)
    c = 0
    p = 0
    for g in range(n_bins):
        for s in range(width):
            if dense[g, s] > 0:
                cells[0, c] = g
                cells[1, c] = s
                cells[2, c] = dense[g, s]
                c += 1
        while p < n_large and large[p] // stride == g:  # sizes above the table width
            key = large[p]
            run = 0
            while p < n_large and large[p] == key:
                run += 1
                p += 1
            cells[0, c] = g
            cells[1, c] = key % stride
            cells[2, c] = run
            c += 1
    return cells


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _uf_find(parent: npt.NDArray[np.floating[Any]], a: int) -> int:
    """Union-find root lookup with full path compression."""
    root = a
    while parent[root] != root:
        root = parent[root]
    while parent[a] != root:
        nxt = parent[a]
        parent[a] = root
        a = nxt
    return root


@lru_cache(maxsize=16)
def _zone_offsets(shape: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """`_zone_offsets_numba` of `shape`, made once per shape (the kernels only read it)."""
    return cast(npt.NDArray[np.int64], _zone_offsets_numba(shape))


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _zone_offsets_numba(shape: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """The 26 neighbour steps in the flat index of an array of `shape`."""
    offsets = np.empty(26, dtype=np.int64)
    k = 0
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dz != 0 or dy != 0 or dx != 0:
                    offsets[k] = dz * shape[1] * shape[2] + dy * shape[2] + dx
                    k += 1
    return offsets


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _fill_zones_numba(
    flat: npt.NDArray[np.uint16],
    flat_dist: npt.NDArray[np.uint16],
    offsets: npt.NDArray[np.int64],
    lo: int,
    hi: int,
    stride_z: int,
    base: int,
    end: int,
    calc_gldzm: bool,
    res_gl: npt.NDArray[np.uint16],
    res_size: npt.NDArray[np.int32],
    res_dist: npt.NDArray[np.uint16],
    parent: npt.NDArray[np.int32],
    lab_lo: npt.NDArray[np.int32],
    lab_hi: npt.NDArray[np.int32],
) -> int:
    """Flood-fill the zones of the flat padded volume between the indices lo and hi (whole
    slices) and return their number. Zone t gets the id base + t, with its level, size and
    smallest distance, and its voxels on the first and the last slice get the label id + 1
    in lab_lo and lab_hi (when these have a slice of room). Visited voxels become 0.

    The stack of the fill lives in `parent`, growing down from `end` (base + the texture
    voxels of the chunk): the zones so far and the pending voxels are distinct voxels of
    the chunk, so the two never meet; a zone's parent slot is written when it is done.
    The neighbour steps come in the order dz = -1, 0, +1 (9 each), so a voxel on the first
    slice of the chunk skips the first 9 and one on the last slice the last 9: the other
    steps stay in the chunk (the padding ring is never pushed)."""
    first_end = lo + stride_z
    last = hi - stride_z  # first index of the last slice
    faces = lab_lo.size > 1
    n_zones = 0
    for i in range(lo, hi):
        gl = flat[i]
        if gl == 0:
            continue
        zid = base + n_zones
        flat[i] = 0
        top = end - 1
        parent[top] = i
        size = 0
        min_dist = flat_dist[i] if calc_gldzm else 0
        while top < end:
            cur = parent[top]
            top += 1
            size += 1
            on_first = cur < first_end
            on_last = cur >= last
            if faces and on_first:
                lab_lo[cur - lo] = zid + 1
            if faces and on_last:
                lab_hi[cur - last] = zid + 1
            if calc_gldzm and flat_dist[cur] < min_dist:
                min_dist = flat_dist[cur]
            k0 = 9 if on_first else 0
            k1 = 17 if on_last else 26
            for k in range(k0, k1):
                nb = cur + offsets[k]
                if flat[nb] == gl:
                    flat[nb] = 0
                    top -= 1
                    parent[top] = nb
        parent[zid] = zid
        res_gl[zid] = gl
        res_size[zid] = size
        res_dist[zid] = min_dist
        n_zones += 1
    return n_zones


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _face_pairs_numba(
    hi: npt.NDArray[np.int32],
    lo: npt.NDArray[np.int32],
    res_gl: npt.NDArray[np.int32],
    height: int,
    width: int,
    pairs: npt.NDArray[np.int32],
    start: int,
) -> int:
    """Write the pairs (zone a, zone b) of one grey level that touch across a chunk face
    into the rows of `pairs` from `start`, and return their number. Zone a has a voxel in
    the last slice of the chunk (labels `hi`), zone b one of the 9 voxels around it in the
    first slice of the next chunk (labels `lo`). The pairs come in the order y, x, dy, dx.
    A pair that repeats the pair before it is left out: its union changes nothing."""
    n = 0
    last_a = -1
    last_b = -1
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            la = hi[y * width + x]
            if la == 0:
                continue
            gl = res_gl[la - 1]
            for dy in range(-1, 2):
                row = (y + dy) * width + x
                for dx in range(-1, 2):
                    lb = lo[row + dx]
                    if lb == 0 or res_gl[lb - 1] != gl or (la == last_a and lb == last_b):
                        continue
                    last_a = la
                    last_b = lb
                    pairs[start + n, 0] = la - 1
                    pairs[start + n, 1] = lb - 1
                    n += 1
    return n


@jit(nopython=True, nogil=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _label_zones_numba(
    vol: npt.NDArray[np.uint16],
    dist: npt.NDArray[np.uint16],
    bounds: npt.NDArray[np.int64],
    roi_base: npt.NDArray[np.int64],
    calc_gldzm: bool,
    res_gl: npt.NDArray[np.uint16],
    res_size: npt.NDArray[np.int32],
    res_dist: npt.NDArray[np.uint16],
    parent: npt.NDArray[np.int32],
    lab_lo: npt.NDArray[np.int32],
    lab_hi: npt.NDArray[np.int32],
    zone_counts: npt.NDArray[np.int64],
    slots: npt.NDArray[np.int64],
) -> None:
    """Find the zones (26-connected voxels of one grey level) of the padded volume `vol`,
    with one flood fill (`_fill_zones_numba`) per chunk of padded slices bounds[c] to
    bounds[c + 1].

    The chunks run in parallel. Zone t of chunk c gets the id roi_base[c] + t (roi_base
    counts the texture voxels of the earlier chunks, so the ids of the chunks never
    overlap; the stack of chunk c grows down from roi_base[c + 1] in `parent`). Zones that
    touch across a chunk face are then joined
    (union-find): the size of the root is the sum and its distance the minimum, so the
    result is the same for every chunk split. The faces find their pairs of zones in
    parallel: face c writes them into the rows slots[c] to slots[c + 1] (9 rows for each
    texture voxel of the last slice of chunk c). `vol` is all 0 afterwards.
    """
    stride_z = vol.shape[1] * vol.shape[2]
    flat = vol.ravel()
    flat_dist = dist.ravel()
    offsets = _zone_offsets_numba(vol.shape)
    n_chunks = bounds.shape[0] - 1
    n_ids = parent.shape[0]
    for c in prange(n_chunks):
        zone_counts[c] = _fill_zones_numba(
            flat,
            flat_dist,
            offsets,
            bounds[c] * stride_z,
            bounds[c + 1] * stride_z,
            stride_z,
            roi_base[c],
            roi_base[c + 1] if c + 1 < n_chunks else n_ids,
            calc_gldzm,
            res_gl,
            res_size,
            res_dist,
            parent,
            lab_lo[c],
            lab_hi[c],
        )

    # Join the zones that touch across each chunk face (26-connected: the 9 voxels of
    # the next slice around each voxel of the last slice). The faces find their pairs in
    # parallel; the unions then run in the order of one walk over the faces.
    found = np.zeros(n_chunks, dtype=np.int64)
    pairs = np.empty((slots[-1], 2), dtype=np.int32)
    for c in prange(n_chunks - 1):
        found[c] = _face_pairs_numba(
            lab_hi[c], lab_lo[c + 1], res_gl, vol.shape[1], vol.shape[2], pairs, slots[c]
        )
    # Each union makes one root a zone of another root. Only these joined zones add their
    # size and distance to a root, after all unions.
    joined = np.empty(found.sum(), dtype=np.int32)
    n_joined = 0
    for c in range(n_chunks - 1):
        for k in range(slots[c], slots[c] + found[c]):
            ra = _uf_find(parent, pairs[k, 0])
            rb = _uf_find(parent, pairs[k, 1])
            if ra < rb:
                parent[rb] = ra
                joined[n_joined] = rb
                n_joined += 1
            elif rb < ra:
                parent[ra] = rb
                joined[n_joined] = ra
                n_joined += 1
    for k in range(n_joined):
        zid = joined[k]
        r = _uf_find(parent, zid)
        res_size[r] += res_size[zid]
        if res_dist[zid] < res_dist[r]:
            res_dist[r] = res_dist[zid]


@jit(nopython=True, nogil=True, cache=True)  # type: ignore
def _zone_tables_numba(
    res_gl: npt.NDArray[np.uint16],
    res_size: npt.NDArray[np.int32],
    res_dist: npt.NDArray[np.uint16],
    parent: npt.NDArray[np.int32],
    zone_counts: npt.NDArray[np.int64],
    roi_base: npt.NDArray[np.int64],
    n_bins: int,
    calc_glszm: bool,
    calc_gldzm: bool,
    dense_glszm: bool,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """GLSZM and GLDZM from the root zones of `_label_zones_numba`. The GLSZM is the dense
    (n_bins, largest zone) matrix, or with `dense_glszm=False` only its non-zero cells
    (the (3, n_cells) array of `_glszm_cells`)."""
    n_roots = 0
    for c in range(zone_counts.shape[0]):
        for t in range(zone_counts[c]):
            zid = roi_base[c] + t
            if parent[zid] == zid:
                n_roots += 1
    gl = np.empty(n_roots, dtype=np.int32)
    size = np.empty(n_roots, dtype=np.int32)
    dist = np.empty(n_roots, dtype=np.int32)
    r = 0
    for c in range(zone_counts.shape[0]):
        for t in range(zone_counts[c]):
            zid = roi_base[c] + t
            if parent[zid] == zid:
                gl[r] = res_gl[zid]
                size[r] = res_size[zid]
                dist[r] = res_dist[zid]
                r += 1

    glszm = np.zeros((n_bins, 1), dtype=np.uint32)
    if not dense_glszm:
        glszm = np.zeros((3, 0), dtype=np.uint32)
        if calc_glszm:
            cells = _GLSZM_DENSE_CELLS if n_bins <= _GLSZM_MANY_LEVELS else _GLSZM_MANY_LEVEL_CELLS
            glszm = _glszm_cells(gl, size, n_bins, cells)
    elif calc_glszm and n_roots > 0:
        glszm = np.zeros((n_bins, size.max()), dtype=np.uint32)
        for r in range(n_roots):
            glszm[gl[r] - 1, size[r] - 1] += 1

    gldzm = np.zeros((n_bins, 1), dtype=np.uint32)
    if calc_gldzm and n_roots > 0:
        # A zone of distance 0 holds a texture voxel outside the distance mask: the caller
        # checks for it (the distance of a zone is 1 or more)
        gldzm = np.zeros((n_bins, max(1, int(dist.max()))), dtype=np.uint32)
        for r in range(n_roots):
            if dist[r] > 0:
                gldzm[gl[r] - 1, dist[r] - 1] += 1
    return glszm, gldzm


@jit(nopython=True, nogil=True, parallel=PRANGE_ONLY, cache=True)  # type: ignore
def _zone_cells_parallel_numba(
    res_gl: npt.NDArray[Any],
    res_size: npt.NDArray[np.int32],
    res_dist: npt.NDArray[Any],
    parent: npt.NDArray[np.int32],
    zone_counts: npt.NDArray[np.int64],
    roi_base: npt.NDArray[np.int64],
    n_bins: int,
    calc_glszm: bool,
    calc_gldzm: bool,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """The GLSZM cells (as `_glszm_cells`) and the GLDZM of `_zone_tables_numba` with
    `dense_glszm=False`, with each chunk of zones in a thread. Each chunk moves its root
    zones to the start of its id range, then counts them in its own tables; the tables add
    up. The large zones (wider than the dense GLSZM table) go into the chunk's range of
    `parent` (as zone ids), and their keys are sorted after the chunks. Afterwards the
    buffers hold only the root zones."""
    n_chunks = zone_counts.shape[0]
    # roots, largest size, largest distance and large zones of each chunk
    stats = np.zeros((n_chunks, 4), dtype=np.int64)
    for c in prange(n_chunks):
        base = roi_base[c]
        k = 0
        largest = 0
        farthest = 0
        for t in range(zone_counts[c]):
            zid = base + t
            if parent[zid] == zid:
                s = res_size[zid]
                d = res_dist[zid]
                res_gl[base + k] = res_gl[zid]
                res_size[base + k] = s
                res_dist[base + k] = d
                if s > largest:
                    largest = s
                if d > farthest:
                    farthest = d
                k += 1
        stats[c, 0] = k
        stats[c, 1] = largest
        stats[c, 2] = farthest
    max_sz = int(stats[:, 1].max())
    width = min(max_sz, max(1, _GLSZM_DENSE_CELLS // n_bins)) if calc_glszm else 0
    n_dist = max(1, int(stats[:, 2].max())) if calc_gldzm else 1
    small = np.zeros((n_chunks, n_bins, width), dtype=np.uint32)
    near = np.zeros((n_chunks, n_bins, n_dist if calc_gldzm else 0), dtype=np.uint32)
    for c in prange(n_chunks):
        base = roi_base[c]
        n_large = 0
        for k in range(stats[c, 0]):
            g = res_gl[base + k] - 1
            if calc_glszm:
                s = res_size[base + k]
                if s <= width:
                    small[c, g, s - 1] += 1
                else:
                    parent[base + n_large] = base + k
                    n_large += 1
            if calc_gldzm and res_dist[base + k] > 0:
                near[c, g, res_dist[base + k] - 1] += 1
        stats[c, 3] = n_large

    gldzm = np.zeros((n_bins, n_dist), dtype=np.uint32)
    dense = np.zeros((n_bins, width), dtype=np.int64)
    # The tables add up in this thread (a parallel region costs about 0.1 ms)
    for g in range(n_bins):
        for c in range(n_chunks):
            for s in range(width):
                dense[g, s] += small[c, g, s]
            if calc_gldzm:
                for d in range(n_dist):
                    gldzm[g, d] += near[c, g, d]
    glszm = np.zeros((3, 0), dtype=np.uint32)
    if calc_glszm:
        stride = np.int64(max_sz) + 1
        large = np.empty(stats[:, 3].sum(), dtype=np.int64)
        j = 0
        for c in range(n_chunks):
            for q in range(stats[c, 3]):
                zid = parent[roi_base[c] + q]
                large[j] = (np.int64(res_gl[zid]) - 1) * stride + (res_size[zid] - 1)
                j += 1
        large.sort()
        glszm = _glszm_cells_of(dense, large, stride)
    return glszm, gldzm


# From this many zones on (sum of zone_counts), the compact zone tables run in threads.
# Measured on 10 threads: x0.85 at 180,000 zones, x0.94 at 113,000, x1.0-1.16 below
# 70,000 (two parallel regions cost about 0.2 ms).
_ZONE_TABLES_PARALLEL_MIN = 1 << 17


_NO_DISTANCE = np.zeros((1, 1, 1), dtype=np.uint16)
_NO_FACE = np.zeros(1, dtype=np.int32)
# Below this many voxels of the padded volume, one serial zone fill is faster than the
# parallel chunks (measured).
_ZONE_PARALLEL_MIN_SIZE = 1 << 17


def _zone_matrices(
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
    dist: npt.NDArray[np.uint16],
    n_bins: int,
    calc_glszm: bool,
    calc_gldzm: bool,
    dense_glszm: bool,
    parallel: Optional[bool] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """GLSZM and GLDZM of the padded volume, with the padded distance map `dist` (any
    array when `calc_gldzm` is False). `vol` is all 0 afterwards. The buffers are made
    for each call: the zones need 12 bytes per texture voxel, and a face label pair per
    chunk. `parallel` picks the parallel zone kernel (default: from
    _ZONE_PARALLEL_MIN_SIZE voxels on). True: the parallel zone kernel and, for the
    compact matrices, the parallel tables (the warm-up asks for them on a small volume)."""
    per_slice = counts[:, 0]
    n_voxels = max(1, int(per_slice.sum()))
    # The zone level, size and distance, and the parents (the fill keeps its stack of flat
    # indices of the padded volume in `parent`)
    res_gl = np.empty(n_voxels, dtype=np.uint16)
    res_size = np.empty(n_voxels, dtype=np.int32)
    res_dist = np.empty(n_voxels, dtype=np.uint16)
    parent = np.empty(n_voxels, dtype=np.int32)
    tables_in_threads = parallel is True  # the warm-up asks for them on a small volume
    if parallel is None:
        parallel = vol.size >= _ZONE_PARALLEL_MIN_SIZE
    if not parallel or (n_chunks := numba.get_num_threads()) == 1:
        # One serial fill: below this size the parallel start costs more than the fill
        stride_z = vol.shape[1] * vol.shape[2]
        n_zones = _fill_zones_numba(
            vol.ravel(),
            dist.ravel(),
            _zone_offsets(vol.shape),
            stride_z,
            vol.size - stride_z,
            stride_z,
            0,
            n_voxels,
            calc_gldzm,
            res_gl,
            res_size,
            res_dist,
            parent,
            _NO_FACE,
            _NO_FACE,
        )
        return cast(
            tuple[npt.NDArray[Any], npt.NDArray[Any]],
            _zone_tables_numba(
                res_gl,
                res_size,
                res_dist,
                parent,
                np.array([n_zones], dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                n_bins,
                calc_glszm,
                calc_gldzm,
                dense_glszm,
            ),
        )
    bounds = _z_blocks(per_slice, n_chunks)
    roi_base = np.concatenate(([0], np.cumsum(per_slice)))[bounds[:-1]].astype(np.int64)
    # The rows of the zone pairs of each chunk face: 9 for each texture voxel of the last
    # slice of the chunk
    slots = np.concatenate(([0], np.cumsum(9 * per_slice[bounds[1:-1] - 1]))).astype(np.int64)
    slice_size = vol.shape[1] * vol.shape[2]
    lab_lo = np.zeros((bounds.size - 1, slice_size), dtype=np.int32)
    lab_hi = np.zeros((bounds.size - 1, slice_size), dtype=np.int32)
    zone_counts = np.empty(bounds.size - 1, dtype=np.int64)
    _label_zones_numba(
        vol,
        dist,
        bounds + 1,  # padded slices
        roi_base,
        calc_gldzm,
        res_gl,
        res_size,
        res_dist,
        parent,
        lab_lo,
        lab_hi,
        zone_counts,
        slots,
    )
    if (
        not dense_glszm
        and n_bins <= _GLSZM_MANY_LEVELS
        and (tables_in_threads or zone_counts.sum() >= _ZONE_TABLES_PARALLEL_MIN)
    ):
        return cast(
            tuple[npt.NDArray[Any], npt.NDArray[Any]],
            _zone_cells_parallel_numba(
                res_gl,
                res_size,
                res_dist,
                parent,
                zone_counts,
                roi_base,
                n_bins,
                calc_glszm,
                calc_gldzm,
            ),
        )
    return cast(
        tuple[npt.NDArray[Any], npt.NDArray[Any]],
        _zone_tables_numba(
            res_gl,
            res_size,
            res_dist,
            parent,
            zone_counts,
            roi_base,
            n_bins,
            calc_glszm,
            calc_gldzm,
            dense_glszm,
        ),
    )


def calculate_zone_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    dist_map: npt.NDArray[np.floating[Any]],
    n_bins: int,
    calc_glszm: bool = True,
    calc_gldzm: bool = True,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    GLSZM and GLDZM of the ROI (`mask != 0`), with the distance map `dist_map`.

    Args:
        data: 3D discretized image data.
        mask: 3D mask array (not modified - copied internally by JIT function).
        dist_map: 3D distance map for GLDZM, with whole numbers in [0, 65535].
        n_bins: Number of grey level bins.
        calc_glszm: Whether to calculate GLSZM.
        calc_gldzm: Whether to calculate GLDZM.

    Returns:
        Tuple of (glszm, gldzm) matrices.

    Raises:
        ValueError: a `dist_map` value is not in [0, 65535] (with `calc_gldzm`).

    Example:
        ```python
        import numpy as np
        from pictologics.features.texture import calculate_zone_features

        rng = np.random.default_rng(0)
        data = rng.integers(1, 5, size=(8, 8, 8)).astype(np.float64)
        mask = np.ones((8, 8, 8), dtype=np.uint8)
        dist_map = np.zeros((8, 8, 8), dtype=np.int32)

        glszm, gldzm = calculate_zone_features(data, mask, dist_map, n_bins=4, calc_gldzm=False)
        print(int(glszm.sum()))
        # 13
        ```
    """
    vol, counts = _texture_volume(_level_form(data), np.not_equal(mask, 0, order="C"), n_bins)
    dist = _NO_DISTANCE
    if calc_gldzm:
        dist_map = np.asarray(dist_map)
        # The kernels read a uint16 map: a value outside [0, 65535] (or NaN) would wrap
        if not np.all((dist_map >= 0) & (dist_map <= np.iinfo(np.uint16).max)):
            raise ValueError("dist_map values must lie in [0, 65535].")
        dist = np.pad(dist_map.astype(np.uint16), 1)
    return _zone_matrices(vol, counts, dist, n_bins, calc_glszm, calc_gldzm, dense_glszm=True)


def calculate_glszm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    glszm_matrix: Optional[npt.NDArray[np.floating[Any]]] = None,
) -> dict[str, float]:
    """
    Calculate Grey Level Size Zone Matrix (GLSZM) features.

    The GLSZM counts the number of zones (connected components) of linked voxels
    that share the same grey level intensity. A zone is defined as a group of connected voxels
    with the same grey level. This matrix is rotationally invariant by definition.

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
        mask (npt.NDArray[np.floating[Any]]): The 3D mask array defining the ROI. Nonzero values indicate ROI membership.
        n_bins (int): The number of grey levels.
        glszm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated GLSZM matrix.

    Returns:
        dict[str, float]: A dictionary of calculated GLSZM features.
            Example keys: 'small_zone_emphasis_P001', 'zone_percentage_P30P'.
    """
    if glszm_matrix is None:
        # Standalone path: crop to the ROI bbox and rebind, so the zone kernel and the
        # zone-percentage voxel count below run on the cropped arrays.
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        # The kernel returns only the non-zero cells: no dense (n_bins, largest zone) matrix.
        matrices = _one_matrix(data, mask, n_bins, "glszm")
        return _glszm_features_from_cells(
            matrices["glszm_cells"], None, n_voxels=matrices["n_voxels"]
        )

    # np.nonzero gives strided index arrays: row-order copies, for the compiled kernel
    gl_idx, sz_idx = (np.ascontiguousarray(i) for i in np.nonzero(glszm_matrix))
    return _glszm_features(gl_idx, sz_idx, glszm_matrix[gl_idx, sz_idx].astype(np.float64), mask)


def _glszm_features_from_cells(
    cells: npt.NDArray[Any], mask: Optional[npt.NDArray[Any]], n_voxels: Optional[int] = None
) -> dict[str, float]:
    """GLSZM features from the (3, n_cells) cell array of `_glszm_cells`."""
    return _glszm_features(
        cells[0].astype(np.int64),
        cells[1].astype(np.int64),
        cells[2].astype(np.float64),
        mask,
        n_voxels,
    )


def _glszm_features(
    gl_idx: npt.NDArray[np.int64],
    sz_idx: npt.NDArray[np.int64],
    c: npt.NDArray[np.float64],
    mask: Optional[npt.NDArray[Any]],
    n_voxels: Optional[int] = None,
) -> dict[str, float]:
    """GLSZM features from the non-zero cells: 0-based grey level and size, and count, in
    row order (as np.nonzero gives them). The GLSZM is very sparse along the zone sizes,
    and every term carries a factor of P, so the non-zero cells give the features of the
    whole matrix. n_voxels: the ROI voxel count (None: count `mask`)."""
    if gl_idx.size == 0:
        return {}
    out = np.empty(len(_GLSZM_KEYS))
    n_zones = _glszm_features_numba(gl_idx, sz_idx, c, out)
    if n_voxels is None:
        n_voxels = _roi_voxel_count(cast(npt.NDArray[Any], mask))
    out[6] = np.float64(n_zones) / n_voxels
    return dict(zip(_GLSZM_KEYS, out, strict=True))


# --- GLDZM ---


def calculate_gldzm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    gldzm_matrix: Optional[npt.NDArray[np.floating[Any]]] = None,
    distance_mask: Optional[npt.NDArray[np.floating[Any]]] = None,
) -> dict[str, float]:
    """
    Calculate Grey Level Distance Zone Matrix (GLDZM) features.

    The GLDZM counts the number of zones of linked voxels with the same grey level,
    categorized by the distance of the zone from the ROI border.
    This captures information about the spatial distribution of textures relative to the boundary.

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
        mask (npt.NDArray[np.floating[Any]]): The 3D mask array defining the ROI. Nonzero values indicate ROI membership.
        n_bins (int): The number of grey levels.
        gldzm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated GLDZM matrix.
        distance_mask (Optional[npt.NDArray[np.floating[Any]]]): Optional mask used to calculate the distance map.
            If None, `mask` is used. This allows calculating distances based on the morphological mask
            while analyzing intensities from the intensity mask (e.g., after outlier filtering).

    Returns:
        dict[str, float]: A dictionary of calculated GLDZM features.
            Example keys: 'small_distance_emphasis_0GBI', 'zone_distance_entropy_GBDU'.

    Raises:
        ValueError: a mask voxel lies outside `distance_mask`.
    """
    if gldzm_matrix is None:
        # Standalone path: crop to the bbox of mask ∪ distance_mask and rebind. The
        # union keeps the distance map identical (the distance region is fully inside
        # the bbox), and the voxel count below runs on the cropped mask.
        planar = _planar_axes(data.shape)
        data, mask, distance_mask = _maybe_crop_to_bbox(data, mask, distance_mask)
        matrices = _one_matrix(
            data, mask, n_bins, "gldzm", distance_mask=distance_mask, planar=planar
        )
        return _gldzm_features(matrices["gldzm"], None, n_voxels=matrices["n_voxels"])
    gldzm = gldzm_matrix
    if gldzm.dtype.kind == "u":  # counts: the kernel
        return _gldzm_features(gldzm, mask)

    N_zones = np.sum(gldzm)
    if N_zones == 0:
        return {}

    gldzm, rows, cols = _occupied(gldzm)
    P = gldzm / N_zones

    # The 1-based numbers of the rows and columns that hold counts
    I, J = np.meshgrid(rows, cols, indexing="ij")  # noqa: E741
    I2 = I**2
    J2 = J**2

    features = {}

    # Small Distance Emphasis (SDE) - 0GBI
    features["small_distance_emphasis_0GBI"] = np.sum(P / J2)

    # Large Distance Emphasis (LDE) - MB4I
    features["large_distance_emphasis_MB4I"] = np.sum(P * J2)

    # Grey Level Non-Uniformity (GLNU) - VFT7
    s_i = np.sum(gldzm, axis=1)
    features["grey_level_non_uniformity_VFT7"] = np.sum(s_i**2) / N_zones

    # Normalised Grey Level Non-Uniformity (GLNN) - 7HP3
    features["normalised_grey_level_non_uniformity_7HP3"] = np.sum(s_i**2) / (N_zones**2)

    # Zone Distance Non-Uniformity (ZDNU) - V294
    s_j = np.sum(gldzm, axis=0)
    features["zone_distance_non_uniformity_V294"] = np.sum(s_j**2) / N_zones

    # Normalised Zone Distance Non-Uniformity (ZDNN) - IATH
    features["normalised_zone_distance_non_uniformity_IATH"] = np.sum(s_j**2) / (N_zones**2)

    # Zone Percentage (ZP) - VIWW
    n_voxels = _roi_voxel_count(mask)
    features["zone_percentage_VIWW"] = N_zones / n_voxels

    # Grey Level Variance (GLV) - QK93
    mu_i = np.sum(I * P)
    features["grey_level_variance_QK93"] = np.sum(((I - mu_i) ** 2) * P)

    # Zone Distance Variance (ZDV) - 7WT1
    mu_j = np.sum(J * P)
    features["zone_distance_variance_7WT1"] = np.sum(((J - mu_j) ** 2) * P)

    # Zone Distance Entropy (ZDE) - GBDU
    mask_p = P > 0
    features["zone_distance_entropy_GBDU"] = -np.sum(P[mask_p] * np.log2(P[mask_p]))

    # Low Grey Level Zone Emphasis (LGLZE) - S1RA
    features["low_grey_level_zone_emphasis_S1RA"] = np.sum(P / I2)

    # High Grey Level Zone Emphasis (HGLZE) - K26C
    features["high_grey_level_zone_emphasis_K26C"] = np.sum(P * I2)

    # Small Distance Low Grey Level Emphasis (SDLGLE) - RUVG
    features["small_distance_low_grey_level_emphasis_RUVG"] = np.sum(P / (I2 * J2))

    # Small Distance High Grey Level Emphasis (SDHGLE) - DKNJ
    features["small_distance_high_grey_level_emphasis_DKNJ"] = np.sum(P * I2 / J2)

    # Large Distance Low Grey Level Emphasis (LDLGLE) - A7WM
    features["large_distance_low_grey_level_emphasis_A7WM"] = np.sum(P * J2 / I2)

    # Large Distance High Grey Level Emphasis (LDHGLE) - KLTH
    features["large_distance_high_grey_level_emphasis_KLTH"] = np.sum(P * I2 * J2)

    return features


# --- NGTDM ---


@jit(nopython=True, cache=True)  # type: ignore
def _ngtdm_complexity_numba(
    levels: npt.NDArray[np.int64], p: npt.NDArray[np.float64], s: npt.NDArray[np.float64]
) -> float:
    """Σ_ij |i - j| (p_i s_i + p_j s_j) / (p_i + p_j) over the pairs of non-zero grey levels
    (the NGTDM complexity times N_vp): twice the sum over i < j, as the terms are symmetric
    and 0 for i = j. Each row adds into its own sum first."""
    m = len(levels)
    total = 0.0
    for a in range(m):
        ps_a = p[a] * s[a]
        row = 0.0
        for b in range(a + 1, m):
            row += abs(levels[a] - levels[b]) * (ps_a + p[b] * s[b]) / (p[a] + p[b])
        total += row
    return 2.0 * total


def calculate_ngtdm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    ngtdm_matrices: Optional[
        tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
    ] = None,
    ngtdm_distance: int = 1,
) -> dict[str, float]:
    """
    Calculate Neighbourhood Grey Tone Difference Matrix (NGTDM) features.

    The NGTDM quantifies the difference between a grey value and the average grey value
    of its neighbours. It captures the coarseness and contrast of the texture.

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
        mask (npt.NDArray[np.floating[Any]]): The 3D mask array defining the ROI. Nonzero values indicate ROI membership.
        n_bins (int): The number of grey levels.
        ngtdm_matrices (Optional[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]): Pre-calculated NGTDM matrices
            (sum of absolute differences `s`, and count `n`).
        ngtdm_distance (int): The Chebyshev distance δ of the neighbourhood, for matrices
            calculated from scratch. Default 1 (the 26 neighbours).

    Returns:
        dict[str, float]: A dictionary of calculated NGTDM features.
            Example keys: 'coarseness_QCDE', 'contrast_65HE', 'busyness_NQ30'.
    """
    if ngtdm_matrices is None:
        # Standalone path: crop to the ROI bbox so the kernel runs on the cropped arrays.
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        matrices = _one_matrix(data, mask, n_bins, "ngtdm", distance=ngtdm_distance)
        s, n = matrices["ngtdm_s"], matrices["ngtdm_n"]
    else:
        s, n = ngtdm_matrices

    # s[i] is sum of absolute differences for grey level i+1
    # n[i] is number of voxels of grey level i+1 with valid neighborhood

    N_vp = np.sum(n)
    if N_vp == 0:
        return {}

    p = n / N_vp

    # Indices
    i_idx = np.arange(n_bins)
    I = i_idx + 1  # noqa: E741

    features = {}

    # Filter for non-zero probabilities (required for Busyness, Complexity, Strength)
    mask_p = p > 0
    p_nz = p[mask_p]
    s_nz = s[mask_p]
    I_nz = I[mask_p]
    Ng_p = len(p_nz)
    sum_s = np.sum(s)

    # The sums over the pairs of non-zero grey levels have closed forms in sums over the
    # levels, written without the difference of two large sums: with S0 = Σ p_i, the mean
    # level mu = Σ p_i i / S0, the mean c and the sum V = Σ_j (j - c)^2 of the m levels,
    #   Σ_ij p_i p_j (i - j)^2       = 2 S0 Σ_i p_i (i - mu)^2
    #   Σ_ij (p_i + p_j) (i - j)^2   = 2 Σ_i p_i (m (i - c)^2 + V)
    #   Σ_ij |a_i - a_j|             = 2 Σ_k k (m - k) (a_(k) - a_(k-1))  (a = i p_i, sorted)
    # The complexity has no closed form (_ngtdm_complexity_numba).

    # Coarseness - QCDE
    sum_ps = np.sum(p_nz * s_nz)
    if sum_ps > 1e-10:
        features["coarseness_QCDE"] = 1 / sum_ps
    else:
        features["coarseness_QCDE"] = 1e6

    # Contrast - 65HE
    if Ng_p > 1:
        # Term 1: Dynamic range variance
        S0 = np.sum(p_nz)
        dm = I_nz - np.sum(p_nz * I_nz) / S0
        term1 = 2 * S0 * np.sum(p_nz * dm * dm) / (Ng_p * (Ng_p - 1))
        # Term 2: Intensity change
        features["contrast_65HE"] = term1 * (sum_s / N_vp)
    else:
        features["contrast_65HE"] = 0.0

    # Busyness - NQ30
    a = np.sort(I_nz * p_nz)
    k = np.arange(1, Ng_p)
    denom_busyness = 2 * np.sum(k * (Ng_p - k) * np.diff(a))
    if denom_busyness > 1e-10:
        features["busyness_NQ30"] = sum_ps / denom_busyness
    else:
        features["busyness_NQ30"] = 0.0

    # Complexity - HDEZ
    complexity = _ngtdm_complexity_numba(I_nz.astype(np.int64), p_nz, s_nz.astype(np.float64))
    features["complexity_HDEZ"] = (1 / N_vp) * complexity

    # Strength - 1X9X
    dc = I_nz - np.mean(I_nz)
    sum_term_str = 2 * np.sum(p_nz * (Ng_p * dc * dc + np.sum(dc * dc)))
    if sum_s > 1e-10:
        features["strength_1X9X"] = sum_term_str / sum_s
    else:
        features["strength_1X9X"] = 0.0

    return features


# --- NGLDM ---


def calculate_ngldm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    ngldm_matrix: Optional[npt.NDArray[np.floating[Any]]] = None,
    ngldm_alpha: int = 0,
    ngldm_distance: int = 1,
) -> dict[str, float]:
    """
    Calculate Neighbourhood Grey Level Dependence Matrix (NGLDM) features.

    The NGLDM captures the dependence of grey levels on their neighbours.
    A "dependence" is defined as a connected voxel having a similar grey level (within a tolerance α).

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
        mask (npt.NDArray[np.floating[Any]]): The 3D mask array defining the ROI. Nonzero values indicate ROI membership.
        n_bins (int): The number of grey levels.
        ngldm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated NGLDM matrix.
        ngldm_alpha (int): The coarseness parameter α. Two grey levels are considered dependent
            if their absolute difference is ≤ α. Default is 0 (exact match, IBSI standard).
        ngldm_distance (int): The Chebyshev distance δ of the neighbourhood, for a matrix
            calculated from scratch. Default 1.

    Returns:
        dict[str, float]: A dictionary of calculated NGLDM features.
            Example keys: 'low_dependence_emphasis_SODN', 'dependence_count_entropy_FCBV'.
    """
    if ngldm_matrix is None:
        # Standalone path: crop to the ROI bbox and rebind, so the kernel and the
        # dependence-count-percentage voxel count below run on the cropped arrays.
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        matrices = _one_matrix(
            data, mask, n_bins, "ngldm", ngldm_alpha=ngldm_alpha, distance=ngldm_distance
        )
        return _ngldm_features(matrices["ngldm"], None, n_voxels=matrices["n_voxels"])
    ngldm = ngldm_matrix
    if ngldm.dtype.kind == "u":  # counts: the kernel
        return _ngldm_features(ngldm, mask)

    N_s = np.sum(ngldm)
    if N_s == 0:
        return {}

    ngldm, rows, cols = _occupied(ngldm)
    P = ngldm / N_s

    # The 1-based numbers of the rows and columns that hold counts
    I, J = np.meshgrid(rows, cols, indexing="ij")  # noqa: E741
    I2 = I**2
    J2 = J**2

    features = {}

    # Low Dependence Emphasis (LDE) - SODN
    features["low_dependence_emphasis_SODN"] = np.sum(P / J2)

    # High Dependence Emphasis (HDE) - IMOQ
    features["high_dependence_emphasis_IMOQ"] = np.sum(P * J2)

    # Low Grey Level Count Emphasis (LGCE) - TL9H
    features["low_grey_level_count_emphasis_TL9H"] = np.sum(P / I2)

    # High Grey Level Count Emphasis (HGCE) - OAE7
    features["high_grey_level_count_emphasis_OAE7"] = np.sum(P * I2)

    # Low Dependence Low Grey Level Emphasis (LDLGE) - EQ3F
    features["low_dependence_low_grey_level_emphasis_EQ3F"] = np.sum(P / (I2 * J2))

    # Low Dependence High Grey Level Emphasis (LDHGE) - JA6D
    features["low_dependence_high_grey_level_emphasis_JA6D"] = np.sum(P * I2 / J2)

    # High Dependence Low Grey Level Emphasis (HDLGE) - NBZI
    features["high_dependence_low_grey_level_emphasis_NBZI"] = np.sum(P * J2 / I2)

    # High Dependence High Grey Level Emphasis (HDHGE) - 9QMG
    features["high_dependence_high_grey_level_emphasis_9QMG"] = np.sum(P * I2 * J2)

    # Grey Level Non-Uniformity - FP8K
    s_i = np.sum(ngldm, axis=1)
    features["grey_level_non_uniformity_FP8K"] = np.sum(s_i**2) / N_s

    # Normalised Grey Level Non-Uniformity - 5SPA
    features["normalised_grey_level_non_uniformity_5SPA"] = np.sum(s_i**2) / (N_s**2)

    # Dependence Count Non-Uniformity - Z87G
    s_j = np.sum(ngldm, axis=0)
    features["dependence_count_non_uniformity_Z87G"] = np.sum(s_j**2) / N_s

    # Normalised Dependence Count Non-Uniformity - OKJI
    features["normalised_dependence_count_non_uniformity_OKJI"] = np.sum(s_j**2) / (N_s**2)

    # Dependence Count Percentage - 6XV8
    n_voxels = _roi_voxel_count(mask)
    features["dependence_count_percentage_6XV8"] = N_s / n_voxels

    # Grey Level Variance - 1PFV
    mu_i = np.sum(I * P)
    features["grey_level_variance_1PFV"] = np.sum(((I - mu_i) ** 2) * P)

    # Dependence Count Variance - DNX2
    mu_j = np.sum(J * P)
    features["dependence_count_variance_DNX2"] = np.sum(((J - mu_j) ** 2) * P)

    # Dependence Count Entropy - FCBV
    mask_p = P > 0
    features["dependence_count_entropy_FCBV"] = -np.sum(P[mask_p] * np.log2(P[mask_p]))

    # Dependence Count Energy - CAS9
    features["dependence_count_energy_CAS9"] = np.sum(P**2)

    return features


def calculate_all_texture_features(
    disc_array: npt.NDArray[np.floating[Any]],
    mask_array: npt.NDArray[np.floating[Any]],
    n_bins: int,
    distance_mask_array: Optional[npt.NDArray[np.floating[Any]]] = None,
    ngldm_alpha: int = 0,
    glcm_distance: int = 1,
    ngtdm_distance: int = 1,
    ngldm_distance: int = 1,
    families: Optional[str | Iterable[str]] = None,
) -> dict[str, float]:
    """
    Calculate all texture features (GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM).

    This is a convenience wrapper that computes the texture matrices once and then
    extracts the features. `families` chooses the families; the matrices of the other
    families are not computed.

    Args:
        disc_array: Discretised image array.
        mask_array: Mask array (ROI). Nonzero values indicate ROI membership.
        n_bins: Number of bins.
        distance_mask_array: Optional mask for GLDZM distance calculation.
                             If None, mask_array is used.
        ngldm_alpha: The coarseness parameter α for NGLDM. Two grey levels are considered
            dependent if their absolute difference is ≤ α. Default is 0 (IBSI standard).
        glcm_distance: The distance δ between the two voxels of a GLCM pair along each of
            the 13 directions (IBSI 1). Default 1.
        ngtdm_distance: The Chebyshev distance δ of the NGTDM neighbourhood. Default 1.
        ngldm_distance: The Chebyshev distance δ of the NGLDM neighbourhood. Default 1.
        families: The families to calculate, from "glcm", "glrlm", "glszm", "gldzm",
            "ngtdm" and "ngldm" (one name, or a list). None (default): all six.

    Returns:
        Dictionary of the texture features of the families, in the order above.

    Raises:
        ValueError: If a family name is not one of the six.
        ValueError: a mask voxel lies outside `distance_mask_array`.

    Example:
        ```python
        import numpy as np
        from pictologics.features.texture import calculate_all_texture_features

        rng = np.random.default_rng(0)
        disc_array = rng.integers(1, 9, size=(10, 10, 10)).astype(np.float64)
        mask_array = np.ones((10, 10, 10), dtype=np.uint8)

        features = calculate_all_texture_features(disc_array, mask_array, n_bins=8)
        print(len(features))
        # 95
        print(round(features["contrast_ACUI"], 3))
        # 10.526
        glcm_only = calculate_all_texture_features(disc_array, mask_array, 8, families="glcm")
        ```
    """
    chosen = set(
        _TEXTURE_FAMILIES
        if families is None
        else [families]
        if isinstance(families, str)
        else families
    )
    for name in chosen - set(_TEXTURE_FAMILIES):
        close = difflib.get_close_matches(str(name), _TEXTURE_FAMILIES, n=1)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        raise ValueError(
            f"Unknown texture family {name!r}{hint}. The families: {', '.join(_TEXTURE_FAMILIES)}."
        )
    results: dict[str, float] = {}

    # Crop once to the ROI bounding box and use the cropped arrays everywhere below.
    # The per-family feature functions scan the mask (ROI voxel counts, GLCM Ng_eff),
    # so passing full-volume arrays would repeat full-volume scans per family.
    # Feature values are unchanged: cropping only removes zero-mask voxels.
    disc_c, mask_c, distmask_c = _maybe_crop_to_bbox(disc_array, mask_array, distance_mask_array)

    # Calculate all matrices once (compact: same features, smaller matrices)
    planar = _planar_axes(disc_array.shape)
    texture_matrices = _texture_matrices(
        disc_c,
        mask_c,
        n_bins,
        distance_mask=distmask_c,
        ngldm_alpha=ngldm_alpha,
        calc_glcm="glcm" in chosen,
        calc_glrlm="glrlm" in chosen,
        calc_ngtdm="ngtdm" in chosen,
        calc_ngldm="ngldm" in chosen,
        calc_glszm="glszm" in chosen,
        calc_gldzm="gldzm" in chosen,
        compact=True,
        planar=planar,
        glcm_distance=glcm_distance,
        ngtdm_distance=ngtdm_distance,
        ngldm_distance=ngldm_distance,
    )
    # The bool ROI gives the same ROI voxel counts as mask_c, from a fast count.
    # The texture voxel count and the GLCM Ng_eff come with the matrices: the feature
    # functions make no pass over the ROI
    roi = texture_matrices["roi"]
    n_voxels = texture_matrices["n_voxels"]

    if "glcm" in chosen:
        results.update(_glcm_features(texture_matrices["glcm"], n_bins, texture_matrices["ng_eff"]))
    if "glrlm" in chosen:
        results.update(
            _glrlm_features(
                texture_matrices["glrlm"], roi, _directions(planar).size, n_voxels=n_voxels
            )
        )
    if "glszm" in chosen:
        results.update(
            _glszm_features_from_cells(texture_matrices["glszm_cells"], roi, n_voxels=n_voxels)
        )
    if "gldzm" in chosen:
        results.update(_gldzm_features(texture_matrices["gldzm"], roi, n_voxels=n_voxels))
    if "ngtdm" in chosen:
        results.update(
            calculate_ngtdm_features(
                disc_c,
                mask_c,
                n_bins,
                ngtdm_matrices=(texture_matrices["ngtdm_s"], texture_matrices["ngtdm_n"]),
            )
        )
    if "ngldm" in chosen:
        results.update(_ngldm_features(texture_matrices["ngldm"], roi, n_voxels=n_voxels))
    return results
