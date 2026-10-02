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
The main entry point is `calculate_all_texture_matrices`, which computes all raw matrices.
Then, specific feature calculation functions (e.g., `calculate_glcm_features`) can be called
using these matrices.

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

import math
from functools import lru_cache
from typing import Any, Optional, cast

import numba
import numpy as np
from numba import jit, prange
from numba.np.ufunc.parallel import get_thread_id
from numpy import typing as npt
from scipy.ndimage import distance_transform_cdt

from ._utils import compute_nonzero_bbox, merge_bboxes, roi_min_max


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
        return data, mask, distance_mask

    data_c = data[bbox]
    mask_c = mask[bbox]
    dist_c = distance_mask[bbox] if distance_mask is not None else None
    return data_c, mask_c, dist_c


def _roi_voxel_count(mask: npt.NDArray[np.floating[Any]]) -> int:
    """Count ROI voxels using nonzero mask membership semantics."""
    # np.count_nonzero has no fast path for float dtypes (it inspects each 8-byte
    # value), so on a float mask it is ~5x slower than counting a bool array.
    # `mask != 0` yields the identical count (NaN/Inf -> True, -0.0 -> False) via the
    # fast bool popcount path; integer/bool masks already hit that path directly.
    if mask.dtype.kind == "f":
        return int(np.count_nonzero(mask != 0))
    return int(np.count_nonzero(mask))


@jit(nopython=True, cache=True)  # type: ignore
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

    Returns a `(depth+2, height+2, width+2)` int32 array; the caller slices
    `[1:-1, 1:-1, 1:-1]` to recover the unpadded distance map, matching the layout the
    scipy-based padded-then-sliced code previously produced.
    """
    depth, height, width = mask_bool.shape
    p_depth, p_height, p_width = depth + 2, height + 2, width + 2
    dist = np.zeros((p_depth, p_height, p_width), dtype=np.int32)
    inf = np.int32(depth + height + width + 1)

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


def _gldzm_distance_map(
    mask_bool: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """
    Compute the GLDZM distance-to-border map for a binary ROI mask.

    Equivalent to `scipy.ndimage.distance_transform_cdt(mask, metric="taxicab")` on `mask`
    zero-padded by one voxel on every side, sliced back to the original shape: background
    voxels get distance 0, foreground voxels get the taxicab (cityblock) distance to the
    nearest background voxel, and voxels outside the array are treated as background.

    3D masks (the only shape the texture pipeline produces) use the exact numba chamfer
    kernel `_chamfer_distance_taxicab_numba`. Any other dimensionality falls back to the
    original scipy-based implementation.
    """
    if mask_bool.ndim == 3:
        dist_padded = _chamfer_distance_taxicab_numba(mask_bool)
        return cast(npt.NDArray[Any], dist_padded[1:-1, 1:-1, 1:-1])

    mask_padded = np.pad(mask_bool, 1, mode="constant", constant_values=0)
    dist_map_padded = distance_transform_cdt(mask_padded, metric="taxicab").astype(np.int32)
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


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _texture_volume_numba(
    data: npt.NDArray[Any],
    roi: npt.NDArray[np.bool_],
    n_bins: int,
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
) -> None:
    """Write the level of each ROI voxel with a level in [1, n_bins] into the interior of
    the zero array `vol`. counts[z, 0] gets the number of these voxels in slice z, and
    counts[z, 1] the number of ROI voxels."""
    depth, height, width = data.shape
    for z in prange(depth):
        valid = 0
        inside = 0
        for y in range(height):
            for x in range(width):
                if roi[z, y, x]:
                    inside += 1
                    g = data[z, y, x]
                    if g >= 1 and g <= n_bins:
                        vol[z + 1, y + 1, x + 1] = g
                        valid += 1
        counts[z, 0] = valid
        counts[z, 1] = inside


@jit(nopython=True, cache=True)  # type: ignore
def _texture_volume_serial_numba(
    data: npt.NDArray[Any],
    roi: npt.NDArray[np.bool_],
    n_bins: int,
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
) -> None:
    """Serial variant of `_texture_volume_numba` for small volumes, where starting the
    threads costs more than the pass (about 85 us)."""
    depth, height, width = data.shape
    for z in range(depth):
        valid = 0
        inside = 0
        for y in range(height):
            for x in range(width):
                if roi[z, y, x]:
                    inside += 1
                    g = data[z, y, x]
                    if g >= 1 and g <= n_bins:
                        vol[z + 1, y + 1, x + 1] = g
                        valid += 1
        counts[z, 0] = valid
        counts[z, 1] = inside


# From this many voxels on, the grey-level volume is built in threads
_PARALLEL_VOLUME_MIN = 1 << 16


def _texture_volume(
    data: npt.NDArray[Any], roi: npt.NDArray[np.bool_], n_bins: int
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.int64]]:
    """The padded grey-level volume of the texture kernels, and the counts of
    `_texture_volume_numba` (texture voxels and ROI voxels of each slice)."""
    if n_bins > _MAX_TEXTURE_LEVELS:
        raise ValueError(
            f"Texture features take at most {_MAX_TEXTURE_LEVELS:,} grey levels, not {n_bins:,}"
        )
    vol = np.zeros(tuple(s + 2 for s in data.shape), dtype=np.uint16)
    counts = np.empty((data.shape[0], 2), dtype=np.int64)
    if data.size < _PARALLEL_VOLUME_MIN:
        _texture_volume_serial_numba(data, roi, n_bins, vol, counts)
    else:
        _texture_volume_numba(data, roi, n_bins, vol, counts)
    return vol, counts


def _z_blocks(per_slice: npt.NDArray[np.int64], n_blocks: int) -> npt.NDArray[np.int64]:
    """Bounds of at most `n_blocks` contiguous slice ranges with about the same number of
    texture voxels, so that each thread gets about the same work."""
    cum = np.concatenate(([0], np.cumsum(per_slice)))
    bounds = np.searchsorted(cum, cum[-1] * np.arange(n_blocks + 1) / n_blocks, side="left")
    bounds[0], bounds[-1] = 0, per_slice.size
    return np.unique(bounds).astype(np.int64)


def _directions(planar: tuple[bool, bool, bool]) -> npt.NDArray[np.int64]:
    """The indices of the 13 directions that stay in the image. A direction that steps
    along an axis of size 1 (a one-slice image) leaves the image at once: in it, every
    voxel would be a run of length 1, so the GLCM and the GLRLM use the in-plane
    directions only, as for a 2D image."""
    steps_out = (DIRECTIONS_13 != 0) & np.array(planar)
    return np.flatnonzero(~steps_out.any(axis=1)).astype(np.int64)


def _flat_offsets(shape: tuple[int, ...], offsets: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    """The (dz, dy, dx) offsets as steps in the flat index of an array of `shape`."""
    steps = offsets.astype(np.int64) @ np.array([shape[1] * shape[2], shape[2], 1])
    return cast(npt.NDArray[np.int64], steps.astype(np.int64))


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _local_tables_numba(
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
    blocks: npt.NDArray[np.int64],
    calc_glcm: bool,
    calc_glrlm: bool,
    calc_ngtdm: bool,
    calc_ngldm: bool,
    off26: npt.NDArray[np.int64],
    dir_off: npt.NDArray[np.int64],
    dir_table: npt.NDArray[np.int64],
    ngldm_alpha: int,
    glcm: npt.NDArray[np.uint32],
    glrlm: npt.NDArray[np.uint32],
    ngtdm_d: npt.NDArray[np.int64],
    ngtdm_n: npt.NDArray[np.int64],
    ngldm: npt.NDArray[np.uint32],
) -> None:
    """Add the GLCM, GLRLM, NGTDM and NGLDM counts of the padded volume `vol` to the
    tables of the thread that does the work (first axis of each table).

    The slice blocks `blocks` run in parallel. Direction d steps `dir_off[d]` in the
    flat index and adds to table `dir_table[d]` of the GLCM and the GLRLM. The NGTDM
    keeps whole numbers: for a voxel of level g whose n valid neighbours have the level
    sum S, it adds |g n - S| (n times |g - S / n|) to ngtdm_d[tid, g - 1, n]. Integer sums
    do not depend on the order of the voxels, so the NGTDM does not depend on the number
    of threads. The tables are made and summed outside, so the kernel has no other
    parallel region.
    """
    height = vol.shape[1] - 2
    width = vol.shape[2] - 2
    s0 = vol.shape[1] * vol.shape[2]
    s1 = vol.shape[2]
    flat = vol.ravel()
    for b in prange(blocks.shape[0] - 1):
        tid = get_thread_id()
        for z in range(blocks[b], blocks[b + 1]):
            if counts[z, 0] == 0:
                continue
            for y in range(height):
                base = (z + 1) * s0 + (y + 1) * s1 + 1
                for x in range(width):
                    v = base + x
                    g = np.int64(flat[v])
                    if g == 0:
                        continue
                    i = g - 1
                    if calc_ngtdm or calc_ngldm:
                        n_sum = np.int64(0)
                        n_count = np.int64(0)
                        dependence = np.int64(1)
                        for k in range(26):
                            nv = np.int64(flat[v + off26[k]])
                            m = np.int64(nv != 0)
                            n_sum += nv
                            n_count += m
                            dependence += m * np.int64(abs(nv - g) <= ngldm_alpha)
                        if calc_ngtdm and n_count > 0:
                            ngtdm_d[tid, i, n_count] += abs(g * n_count - n_sum)
                            ngtdm_n[tid, i] += 1
                        if calc_ngldm:
                            ngldm[tid, i, dependence - 1] += 1
                    if calc_glcm or calc_glrlm:
                        for d in range(dir_off.shape[0]):
                            od = dir_off[d]
                            t = dir_table[d]
                            if calc_glcm:
                                nv = np.int64(flat[v + od])
                                if nv != 0:
                                    glcm[tid, t, i, nv - 1] += 1
                            # A run starts where the previous voxel has another level
                            if calc_glrlm and np.int64(flat[v - od]) != g:
                                length = 1
                                c = v + od
                                while np.int64(flat[c]) == g:
                                    length += 1
                                    c += od
                                glrlm[tid, t, i, length] += 1


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
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
# From this many cells (4 MB) on, the thread tables are zeroed in threads: np.zeros of a
# large block that the allocator reuses costs a serial memset (2 ms for 48 MB).
_PARALLEL_ZERO_MIN = 1 << 20


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
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


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
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
) -> tuple[
    npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]
]:
    """GLCM (n_dirs, n_bins, n_bins), GLRLM (n_dirs, n_bins, longest run + 1), NGTDM s and n
    (n_bins,) and NGLDM (n_bins, 27) of the padded volume. With `merge_directions`, the
    GLCM and the GLRLM hold one table, the sum over the 13 directions (the features use
    only that sum). A matrix that is not asked for is a zero placeholder. The directions
    along an axis of size 1 in the image (`planar`) stay empty (see _directions)."""
    n_threads = numba.get_num_threads()
    n_tables = 1 if merge_directions else 13
    used = _directions(planar)
    longest = max(vol.shape) - 2
    # Many grey levels: the thread tables hold only the levels that occur
    cells = (n_bins if calc_glcm else 0) + (longest + 1 if calc_glrlm else 0)
    levels = _table_levels(vol, n_bins, 4 * n_threads * n_tables * n_bins * cells)
    rows = n_bins if levels is None else levels.size
    glcm_t = _thread_tables((n_threads, n_tables, rows, rows)) if calc_glcm else _NO_TABLE_4D
    glrlm_t = (
        _thread_tables((n_threads, n_tables, rows, longest + 1)) if calc_glrlm else _NO_TABLE_4D
    )
    ngtdm_d = np.zeros((n_threads if calc_ngtdm else 1, n_bins, 27), dtype=np.int64)
    ngtdm_n = np.zeros((n_threads if calc_ngtdm else 1, n_bins), dtype=np.int64)
    ngldm_t = np.zeros((n_threads if calc_ngldm else 1, n_bins, 27), dtype=np.uint32)
    args = (
        _flat_offsets(vol.shape, OFFSETS_26),
        _flat_offsets(vol.shape, DIRECTIONS_13[used]),
        np.zeros(used.size, dtype=np.int64) if merge_directions else used,
        ngldm_alpha,
        glcm_t,
        glrlm_t,
        ngtdm_d,
        ngtdm_n,
        ngldm_t,
    )
    blocks = _z_blocks(counts[:, 0], 4 * n_threads)
    if levels is None:
        _local_tables_numba(
            vol, counts, blocks, calc_glcm, calc_glrlm, calc_ngtdm, calc_ngldm, *args
        )
    else:
        # The GLCM and the GLRLM count the volume of the occurring levels (1..rows); the
        # NGTDM and the NGLDM read the grey levels themselves
        lut = np.zeros(n_bins + 1, dtype=np.uint16)
        lut[levels + 1] = np.arange(1, rows + 1)
        _local_tables_numba(lut[vol], counts, blocks, calc_glcm, calc_glrlm, False, False, *args)
        if calc_ngtdm or calc_ngldm:
            _local_tables_numba(vol, counts, blocks, False, False, calc_ngtdm, calc_ngldm, *args)
    glcm = _thread_sum(glcm_t) if calc_glcm else np.zeros((13, n_bins, n_bins), dtype=np.uint64)
    glrlm = _thread_sum(glrlm_t) if calc_glrlm else np.zeros((13, n_bins, 1), dtype=np.uint64)
    if levels is not None:  # back to a row and a column for each of the n_bins levels
        if calc_glcm:
            glcm_all = np.zeros((n_tables, n_bins, n_bins), dtype=np.uint64)
            glcm_all[:, levels[:, None], levels[None, :]] = glcm
            glcm = glcm_all
        if calc_glrlm:
            glrlm_all = np.zeros((n_tables, n_bins, longest + 1), dtype=np.uint64)
            glrlm_all[:, levels, :] = glrlm
            glrlm = glrlm_all
    # s_i is the sum over the neighbour counts n of (sum of |g n - S|) / n, in a fixed order
    ngtdm_s = (ngtdm_d.sum(axis=0)[:, 1:] / np.arange(1, 27)).sum(axis=1)
    return (
        glcm,
        glrlm,
        ngtdm_s,
        ngtdm_n.sum(axis=0).astype(np.float64),
        ngldm_t.sum(axis=0, dtype=np.uint64),
    )


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
    extra = {"roi": roi} if compact else {}

    # Fast exit for empty ROI. Checked on the cropped mask; mask_c can still be empty
    # when only distance_mask has nonzero voxels.
    glszm_key = "glszm_cells" if compact else "glszm"
    if not roi.any():
        return {
            "glcm": np.zeros((13, n_bins, n_bins), dtype=np.uint64),
            "glrlm": np.zeros((13, n_bins, 1), dtype=np.uint64),
            "ngtdm_s": np.zeros((n_bins,), dtype=np.float64),
            "ngtdm_n": np.zeros((n_bins,), dtype=np.float64),
            "ngldm": np.zeros((n_bins, 27), dtype=np.uint64),
            glszm_key: np.zeros((3, 0) if compact else (n_bins, 1), dtype=np.uint32),
            "gldzm": np.zeros((n_bins, 1), dtype=np.uint32),
            **extra,
        }

    vol, counts = _texture_volume(data_c, roi, n_bins)
    if compact:
        texture_voxels, roi_voxels = counts.sum(axis=0)
        if texture_voxels != roi_voxels:
            extra["roi"] = vol[1:-1, 1:-1, 1:-1] != 0  # the ROI without invalid levels

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
        )
        matrices.update(glcm=glcm, glrlm=glrlm, ngtdm_s=ngtdm_s, ngtdm_n=ngtdm_n, ngldm=ngldm)
    elif not compact:
        matrices.update(
            glcm=np.zeros((13, n_bins, n_bins), dtype=np.uint64),
            glrlm=np.zeros((13, n_bins, 1), dtype=np.uint64),
            ngtdm_s=np.zeros((n_bins,), dtype=np.float64),
            ngtdm_n=np.zeros((n_bins,), dtype=np.float64),
            ngldm=np.zeros((n_bins, 27), dtype=np.uint64),
        )

    # 2. Zone Features (GLSZM, GLDZM). The zone kernel uses up `vol`, so it comes last.
    if calc_glszm or calc_gldzm:
        dist = _NO_DISTANCE
        if calc_gldzm:
            if distance_map is None:
                # Distance-to-border map of distance_mask, else of the ROI, with the image
                # border treated as an edge (padded like `vol`).
                d_roi = roi if distmask_c is None else np.not_equal(distmask_c, 0, order="C")
                distance_map = _chamfer_distance_taxicab_numba(d_roi, *planar)
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
) -> dict[str, Any]:
    """The compact `_texture_matrices` of one family ('glcm', 'glrlm', 'ngtdm', 'ngldm',
    'glszm' or 'gldzm') for the standalone feature functions, which crop first (and give
    the axes of size 1 of the image as `planar`)."""
    return _texture_matrices(
        data,
        mask,
        n_bins,
        distance_mask=distance_mask,
        ngldm_alpha=ngldm_alpha,
        planar=planar,
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


def calculate_glcm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    glcm_matrix: Optional[npt.NDArray[np.floating[Any]]] = None,
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

        Returns:
            dict[str, float]: A dictionary of calculated GLCM features, keyed by their name and IBSI code.
                Example keys: 'joint_maximum_GYBY', 'contrast_ACUI', 'correlation_NI2N'.

        Example:
            ```python
            import numpy as np
    from numpy import typing as npt
            # ... assuming data and mask defined ...
            features = calculate_glcm_features(data, mask, n_bins=32)
            print(features['contrast_ACUI'])
            ```
            12.5
    """
    if glcm_matrix is None:
        # Standalone path: crop to the ROI bbox and rebind, so the kernel and the
        # Ng_eff scan below run on the cropped arrays instead of the full volume.
        planar = _planar_axes(data.shape)
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        matrices = _one_matrix(data, mask, n_bins, "glcm", planar=planar)
        glcm, mask = matrices["glcm"], matrices["roi"].view(np.uint8)
    else:
        glcm = glcm_matrix

    # Merge (Sum) -> IAZD
    glcm_sum = np.sum(glcm, axis=0)
    glcm_sym = glcm_sum + glcm_sum.T

    # Normalize
    total_sum = np.sum(glcm_sym)
    if total_sum == 0:
        return {}
    glcm_sym, levels, _ = _occupied(glcm_sym, symmetric=True)

    P = glcm_sym / total_sum

    # The 1-based grey levels of the rows and columns that hold counts
    I, J = np.meshgrid(levels, levels, indexing="ij")  # noqa: E741

    features = {}

    # Joint Maximum - GYBY
    features["joint_maximum_GYBY"] = np.max(P)

    # Joint Average - 60VM
    features["joint_average_60VM"] = np.sum(I * P)

    # Joint Variance - UR99
    mu = features["joint_average_60VM"]
    features["joint_variance_UR99"] = np.sum(((I - mu) ** 2) * P)

    # Joint Entropy - TU9B
    mask_p = P > 0
    features["joint_entropy_TU9B"] = -np.sum(P[mask_p] * np.log2(P[mask_p]))

    # Difference Average - TF7R
    k_diff = np.abs(I - J)
    features["difference_average_TF7R"] = np.sum(k_diff * P)

    # Optimized using bincount
    k_diff_flat = k_diff.ravel().astype(np.int32)
    P_flat = P.ravel()
    p_diff = np.bincount(k_diff_flat, weights=P_flat, minlength=n_bins)

    mu_diff = features["difference_average_TF7R"]
    k_vals = np.arange(n_bins)
    features["difference_variance_D3YU"] = np.sum(((k_vals - mu_diff) ** 2) * p_diff)

    # Difference Entropy - NTRS
    mask_pd = p_diff > 0
    features["difference_entropy_NTRS"] = -np.sum(p_diff[mask_pd] * np.log2(p_diff[mask_pd]))

    # Sum Average - ZGXS
    k_sum_grid = I + J

    # Optimized using bincount
    k_sum_flat = k_sum_grid.ravel().astype(np.int32)
    # P_flat is already defined in Difference Variance block
    p_sum_full = np.bincount(k_sum_flat, weights=P_flat, minlength=2 * n_bins + 1)

    # Slice from 2.
    p_sum = p_sum_full[2:]

    k_vals_sum = np.arange(2, 2 * n_bins + 1)
    features["sum_average_ZGXS"] = np.sum(k_vals_sum * p_sum)

    # Sum Variance - OEEB
    mu_sum = features["sum_average_ZGXS"]
    features["sum_variance_OEEB"] = np.sum(((k_vals_sum - mu_sum) ** 2) * p_sum)

    # Sum Entropy - P6QZ
    mask_ps = p_sum > 0
    features["sum_entropy_P6QZ"] = -np.sum(p_sum[mask_ps] * np.log2(p_sum[mask_ps]))

    # Angular Second Moment (Energy) - 8ZQL
    features["angular_second_moment_8ZQL"] = np.sum(P**2)

    # Contrast - ACUI
    sq_diff = (I - J) ** 2
    features["contrast_ACUI"] = np.sum(sq_diff * P)

    # Dissimilarity - 8S9J
    features["dissimilarity_8S9J"] = np.sum(k_diff * P)

    # Inverse Difference - IB1Z
    features["inverse_difference_IB1Z"] = np.sum(P / (1 + k_diff))

    # Ng_eff is the ROI grey-level span; only ROI min/max are needed. Fused single-pass
    # kernel: no bbox rescan and no boolean-gather temporaries.
    roi_span = roi_min_max(data, mask)
    if roi_span is not None:
        Ng_eff = int(roi_span[1] - roi_span[0] + 1)
    else:
        Ng_eff = 1  # Fallback

    # Normalised Inverse Difference - NDRX
    features["normalised_inverse_difference_NDRX"] = np.sum(P / (1 + k_diff / Ng_eff))

    # Inverse Difference Moment - WF0Z
    features["inverse_difference_moment_WF0Z"] = np.sum(P / (1 + sq_diff))

    # Normalised Inverse Difference Moment - 1QCO
    features["normalised_inverse_difference_moment_1QCO"] = np.sum(P / (1 + sq_diff / (Ng_eff**2)))

    # Inverse Variance - E8JP
    mask_neq = I != J
    features["inverse_variance_E8JP"] = np.sum(P[mask_neq] / ((I[mask_neq] - J[mask_neq]) ** 2))

    # Correlation - NI2N
    term1 = np.sum((I - mu) * (J - mu) * P)
    if features["joint_variance_UR99"] != 0:
        features["correlation_NI2N"] = term1 / features["joint_variance_UR99"]
    else:
        features["correlation_NI2N"] = 1.0  # Or NaN? IBSI doesn't specify for 0 variance.

    # Autocorrelation - QWB0
    features["autocorrelation_QWB0"] = np.sum(I * J * P)

    # Cluster Tendency - DG8W
    sum_diff2mu = I + J - 2 * mu
    features["cluster_tendency_DG8W"] = np.sum((sum_diff2mu**2) * P)

    # Cluster Shade - 7NFM
    features["cluster_shade_7NFM"] = np.sum((sum_diff2mu**3) * P)

    # Cluster Prominence - AE86
    features["cluster_prominence_AE86"] = np.sum((sum_diff2mu**4) * P)

    # Information Correlation 1 - R8DG
    HXY = features["joint_entropy_TU9B"]
    # The marginal from the counts: exact, so it does not depend on the order of the sums
    p_x = glcm_sym.sum(axis=1) / total_sum
    mask_px = p_x > 0
    HX = -np.sum(p_x[mask_px] * np.log2(p_x[mask_px]))

    rows, cols = np.nonzero(mask_p)  # in the order of P[mask_p]
    HXY1 = -np.sum(P[mask_p] * np.log2(p_x[rows] * p_x[cols]))

    if HX != 0:
        features["information_correlation_1_R8DG"] = (HXY - HXY1) / HX
    else:
        features["information_correlation_1_R8DG"] = np.nan

    # Information Correlation 2 - JN9H
    P_prod = np.outer(p_x, p_x)
    mask_prod = P_prod > 0
    HXY2 = -np.sum(P_prod[mask_prod] * np.log2(P_prod[mask_prod]))

    features["information_correlation_2_JN9H"] = np.sqrt(1 - np.exp(-2 * (HXY2 - HXY)))
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
        glrlm, mask = matrices["glrlm"], matrices["roi"]
        n_directions = _directions(planar).size
    else:
        glrlm = glrlm_matrix
        if n_directions is None and glrlm.shape[0] == 13:
            # The directions of a one-slice image that leave it have no runs at all
            n_directions = int(np.count_nonzero(glrlm.reshape(13, -1).any(axis=1)))

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
# (grey levels x zone sizes). The few larger zones are sorted instead.
_GLSZM_DENSE_CELLS = 1 << 18


@jit(nopython=True, cache=True)  # type: ignore
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


@jit(nopython=True, cache=True)  # type: ignore
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


@jit(nopython=True, cache=True)  # type: ignore
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


@jit(nopython=True, cache=True)  # type: ignore
def _fill_zones_numba(
    flat: npt.NDArray[np.uint16],
    flat_dist: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.int64],
    lo: int,
    hi: int,
    stride_z: int,
    base: int,
    calc_gldzm: bool,
    res_gl: npt.NDArray[np.int32],
    res_size: npt.NDArray[np.int32],
    res_dist: npt.NDArray[np.int32],
    stack: npt.NDArray[np.int32],
    parent: npt.NDArray[np.int32],
    lab_lo: npt.NDArray[np.int32],
    lab_hi: npt.NDArray[np.int32],
) -> int:
    """Flood-fill the zones of the flat padded volume between the indices lo and hi (whole
    slices) and return their number. Zone t gets the id base + t, with its level, size and
    smallest distance, and its voxels on the first and the last slice get the label id + 1
    in lab_lo and lab_hi (when these have a slice of room). Visited voxels become 0."""
    last = hi - stride_z  # first index of the last slice
    faces = lab_lo.size > 1
    n_zones = 0
    for i in range(lo, hi):
        gl = flat[i]
        if gl == 0:
            continue
        zid = base + n_zones
        parent[zid] = zid
        flat[i] = 0
        stack[base] = i
        top = base + 1
        size = 0
        min_dist = flat_dist[i] if calc_gldzm else 0
        while top > base:
            top -= 1
            cur = stack[top]
            size += 1
            if faces and cur < lo + stride_z:
                lab_lo[cur - lo] = zid + 1
            if faces and cur >= last:
                lab_hi[cur - last] = zid + 1
            if calc_gldzm and flat_dist[cur] < min_dist:
                min_dist = flat_dist[cur]
            for k in range(26):
                nb = cur + offsets[k]
                if lo <= nb < hi and flat[nb] == gl:
                    flat[nb] = 0
                    stack[top] = nb
                    top += 1
        res_gl[zid] = gl
        res_size[zid] = size
        res_dist[zid] = min_dist
        n_zones += 1
    return n_zones


@jit(nopython=True, parallel=True, cache=True)  # type: ignore
def _label_zones_numba(
    vol: npt.NDArray[np.uint16],
    dist: npt.NDArray[np.int32],
    bounds: npt.NDArray[np.int64],
    roi_base: npt.NDArray[np.int64],
    calc_gldzm: bool,
    res_gl: npt.NDArray[np.int32],
    res_size: npt.NDArray[np.int32],
    res_dist: npt.NDArray[np.int32],
    stack: npt.NDArray[np.int32],
    parent: npt.NDArray[np.int32],
    lab_lo: npt.NDArray[np.int32],
    lab_hi: npt.NDArray[np.int32],
    zone_counts: npt.NDArray[np.int64],
) -> None:
    """Find the zones (26-connected voxels of one grey level) of the padded volume `vol`,
    with one flood fill (`_fill_zones_numba`) per chunk of padded slices bounds[c] to
    bounds[c + 1].

    The chunks run in parallel. Zone t of chunk c gets the id roi_base[c] + t (roi_base
    counts the texture voxels of the earlier chunks, so the ids and the stacks of the
    chunks never overlap). Zones that touch across a chunk face are then joined
    (union-find, serial): the size of the root is the sum and its distance the minimum, so
    the result is the same for every chunk split. `vol` is all 0 afterwards.
    """
    stride_z = vol.shape[1] * vol.shape[2]
    stride_y = vol.shape[2]
    flat = vol.ravel()
    flat_dist = dist.ravel()
    offsets = _zone_offsets_numba(vol.shape)
    n_chunks = bounds.shape[0] - 1
    for c in prange(n_chunks):
        zone_counts[c] = _fill_zones_numba(
            flat,
            flat_dist,
            offsets,
            bounds[c] * stride_z,
            bounds[c + 1] * stride_z,
            stride_z,
            roi_base[c],
            calc_gldzm,
            res_gl,
            res_size,
            res_dist,
            stack,
            parent,
            lab_lo[c],
            lab_hi[c],
        )

    # Join the zones that touch across each chunk face (26-connected: the 9 voxels of
    # the next slice around each voxel of the last slice).
    for c in range(n_chunks - 1):
        for y in range(1, vol.shape[1] - 1):
            for x in range(1, vol.shape[2] - 1):
                la = lab_hi[c, y * stride_y + x]
                if la == 0:
                    continue
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        lb = lab_lo[c + 1, (y + dy) * stride_y + (x + dx)]
                        if lb != 0 and res_gl[lb - 1] == res_gl[la - 1]:
                            ra = _uf_find(parent, la - 1)
                            rb = _uf_find(parent, lb - 1)
                            if ra < rb:
                                parent[rb] = ra
                            elif rb < ra:
                                parent[ra] = rb
    for c in range(n_chunks):
        for t in range(zone_counts[c]):
            zid = roi_base[c] + t
            r = _uf_find(parent, zid)
            if r != zid:
                res_size[r] += res_size[zid]
                if res_dist[zid] < res_dist[r]:
                    res_dist[r] = res_dist[zid]


@jit(nopython=True, cache=True)  # type: ignore
def _zone_tables_numba(
    res_gl: npt.NDArray[np.int32],
    res_size: npt.NDArray[np.int32],
    res_dist: npt.NDArray[np.int32],
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
            glszm = _glszm_cells(gl, size, n_bins, _GLSZM_DENSE_CELLS)
    elif calc_glszm and n_roots > 0:
        glszm = np.zeros((n_bins, size.max()), dtype=np.uint32)
        for r in range(n_roots):
            glszm[gl[r] - 1, size[r] - 1] += 1

    gldzm = np.zeros((n_bins, 1), dtype=np.uint32)
    if calc_gldzm and n_roots > 0:
        gldzm = np.zeros((n_bins, max(1, int(dist.max()))), dtype=np.uint32)
        for r in range(n_roots):
            if dist[r] > 0:
                gldzm[gl[r] - 1, dist[r] - 1] += 1
    return glszm, gldzm


_NO_DISTANCE = np.zeros((1, 1, 1), dtype=np.int32)
_NO_FACE = np.zeros(1, dtype=np.int32)
# Below this many voxels of the padded volume, one serial zone fill is faster than the
# parallel chunks (measured).
_ZONE_PARALLEL_MIN_SIZE = 1 << 17


def _zone_matrices(
    vol: npt.NDArray[np.uint16],
    counts: npt.NDArray[np.int64],
    dist: npt.NDArray[np.int32],
    n_bins: int,
    calc_glszm: bool,
    calc_gldzm: bool,
    dense_glszm: bool,
    parallel: Optional[bool] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """GLSZM and GLDZM of the padded volume, with the padded distance map `dist` (any
    array when `calc_gldzm` is False). `vol` is all 0 afterwards. The buffers are made
    for each call: the zones need 20 bytes per texture voxel, and a face label pair per
    chunk. `parallel` picks the parallel zone kernel (default: from
    _ZONE_PARALLEL_MIN_SIZE voxels on; the warm-up asks for it on a small volume)."""
    per_slice = counts[:, 0]
    n_voxels = max(1, int(per_slice.sum()))
    # One buffer for the zone level, size and distance, the parents and the stack (flat
    # indices of the padded volume)
    res_gl, res_size, res_dist, parent, stack = np.empty((5, n_voxels), dtype=np.int32)
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
            calc_gldzm,
            res_gl,
            res_size,
            res_dist,
            stack,
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
        stack,
        parent,
        lab_lo,
        lab_hi,
        zone_counts,
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
        dist_map: 3D distance map for GLDZM.
        n_bins: Number of grey level bins.
        calc_glszm: Whether to calculate GLSZM.
        calc_gldzm: Whether to calculate GLDZM.

    Returns:
        Tuple of (glszm, gldzm) matrices.

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
    vol, counts = _texture_volume(data, mask != 0, n_bins)
    dist = np.pad(np.asarray(dist_map, dtype=np.int32), 1) if calc_gldzm else _NO_DISTANCE
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
        return _glszm_features_from_cells(matrices["glszm_cells"], matrices["roi"])

    gl_idx, sz_idx = np.nonzero(glszm_matrix)
    return _glszm_features(gl_idx, sz_idx, glszm_matrix[gl_idx, sz_idx].astype(np.float64), mask)


def _glszm_features_from_cells(
    cells: npt.NDArray[Any], mask: npt.NDArray[np.floating[Any]]
) -> dict[str, float]:
    """GLSZM features from the (3, n_cells) cell array of `_glszm_cells`."""
    return _glszm_features(
        cells[0].astype(np.int64), cells[1].astype(np.int64), cells[2].astype(np.float64), mask
    )


def _glszm_features(
    gl_idx: npt.NDArray[np.int64],
    sz_idx: npt.NDArray[np.int64],
    c: npt.NDArray[np.float64],
    mask: npt.NDArray[np.floating[Any]],
) -> dict[str, float]:
    """GLSZM features from the non-zero cells: 0-based grey level and size, and count.

    The GLSZM is extremely sparse in the zone-size dimension (a single large zone can push
    the size axis into the tens of thousands while only a few hundred cells are non-zero).
    Working on the non-zero cells is arithmetically identical (every term carries a factor
    of P, so zero cells contribute exactly zero). The cells must come in row-major order,
    as np.nonzero gives them.
    """
    if gl_idx.size == 0:
        return {}

    N_zones = c.sum()

    I = (gl_idx + 1).astype(np.float64)  # noqa: E741
    J = (sz_idx + 1).astype(np.float64)  # Zone size
    I2 = I * I
    J2 = J * J
    P = c / N_zones

    features: dict[str, float] = {}

    # Small Zone Emphasis (SZE) - P001
    features["small_zone_emphasis_P001"] = np.sum(P / J2)

    # Large Zone Emphasis (LZE) - 48P8
    features["large_zone_emphasis_48P8"] = np.sum(P * J2)

    # Grey Level Non-Uniformity (GLNU) - JNSA
    s_i = np.bincount(gl_idx, weights=c)
    sum_si2 = np.sum(s_i**2)
    features["grey_level_non_uniformity_JNSA"] = sum_si2 / N_zones

    # Normalised Grey Level Non-Uniformity (GLNN) - Y1RO
    features["normalised_grey_level_non_uniformity_Y1RO"] = sum_si2 / (N_zones**2)

    # Zone Size Non-Uniformity (ZSNU) - 4JP3
    s_j = np.bincount(sz_idx, weights=c)
    sum_sj2 = np.sum(s_j**2)
    features["zone_size_non_uniformity_4JP3"] = sum_sj2 / N_zones

    # Normalised Zone Size Non-Uniformity (ZSNN) - VB3A
    features["normalised_zone_size_non_uniformity_VB3A"] = sum_sj2 / (N_zones**2)

    # Zone Percentage (ZP) - P30P
    n_voxels = _roi_voxel_count(mask)
    features["zone_percentage_P30P"] = N_zones / n_voxels

    # Grey Level Variance (GLV) - BYLV
    mu_i = np.sum(I * P)
    features["grey_level_variance_BYLV"] = np.sum(((I - mu_i) ** 2) * P)

    # Zone Size Variance (ZSV) - 3NSA
    mu_j = np.sum(J * P)
    features["zone_size_variance_3NSA"] = np.sum(((J - mu_j) ** 2) * P)

    # Zone Size Entropy (ZSE) - GU8N
    features["zone_size_entropy_GU8N"] = -np.sum(P * np.log2(P))

    # Low Grey Level Zone Emphasis (LGLZE) - XMSY
    features["low_grey_level_zone_emphasis_XMSY"] = np.sum(P / I2)

    # High Grey Level Zone Emphasis (HGLZE) - 5GN9
    features["high_grey_level_zone_emphasis_5GN9"] = np.sum(P * I2)

    # Small Zone Low Grey Level Emphasis (SZLGLE) - 5RAI
    features["small_zone_low_grey_level_emphasis_5RAI"] = np.sum(P / (I2 * J2))

    # Small Zone High Grey Level Emphasis (SZHGLE) - HW1V
    features["small_zone_high_grey_level_emphasis_HW1V"] = np.sum(P * I2 / J2)

    # Large Zone Low Grey Level Emphasis (LZLGLE) - YH51
    features["large_zone_low_grey_level_emphasis_YH51"] = np.sum(P * J2 / I2)

    # Large Zone High Grey Level Emphasis (LZHGLE) - J17V
    features["large_zone_high_grey_level_emphasis_J17V"] = np.sum(P * I2 * J2)

    return features


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
        gldzm, mask = matrices["gldzm"], matrices["roi"]
    else:
        gldzm = gldzm_matrix

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


def calculate_ngtdm_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    ngtdm_matrices: Optional[
        tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
    ] = None,
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

    Returns:
        dict[str, float]: A dictionary of calculated NGTDM features.
            Example keys: 'coarseness_QCDE', 'contrast_65HE', 'busyness_NQ30'.
    """
    if ngtdm_matrices is None:
        # Standalone path: crop to the ROI bbox so the kernel runs on the cropped arrays.
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        matrices = _one_matrix(data, mask, n_bins, "ngtdm")
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

    # Pairwise grids over the non-zero grey levels, shared by Contrast/Complexity/Strength.
    Pi, Pj = np.meshgrid(p_nz, p_nz, indexing="ij")
    Ii, Ij = np.meshgrid(I_nz, I_nz, indexing="ij")
    Si, Sj = np.meshgrid(s_nz, s_nz, indexing="ij")

    # Coarseness - QCDE
    sum_ps = np.sum(p_nz * s_nz)
    if sum_ps > 1e-10:
        features["coarseness_QCDE"] = 1 / sum_ps
    else:
        features["coarseness_QCDE"] = 1e6

    # Contrast - 65HE
    Ng_p = len(p_nz)

    if Ng_p > 1:
        # Term 1: Dynamic range variance
        term1_sum = np.sum(Pi * Pj * ((Ii - Ij) ** 2))
        term1 = term1_sum / (Ng_p * (Ng_p - 1))

        # Term 2: Intensity change
        sum_s = np.sum(s)
        term2 = sum_s / N_vp

        features["contrast_65HE"] = term1 * term2
    else:
        features["contrast_65HE"] = 0.0

    # Busyness - NQ30
    IPi = I_nz * p_nz

    # Grid
    IPi_grid, IPj_grid = np.meshgrid(IPi, IPi, indexing="ij")
    denom_busyness = np.sum(np.abs(IPi_grid - IPj_grid))

    if denom_busyness > 1e-10:
        features["busyness_NQ30"] = sum_ps / denom_busyness
    else:
        features["busyness_NQ30"] = 0.0

    # Complexity - HDEZ
    denom_comp = Pi + Pj
    term_comp = np.abs(Ii - Ij) * (Pi * Si + Pj * Sj) / denom_comp

    features["complexity_HDEZ"] = (1 / N_vp) * np.sum(term_comp)

    # Strength - 1X9X
    sum_s = np.sum(s)

    term_str = (Pi + Pj) * ((Ii - Ij) ** 2)
    sum_term_str = np.sum(term_str)

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

    Returns:
        dict[str, float]: A dictionary of calculated NGLDM features.
            Example keys: 'low_dependence_emphasis_SODN', 'dependence_count_entropy_FCBV'.
    """
    if ngldm_matrix is None:
        # Standalone path: crop to the ROI bbox and rebind, so the kernel and the
        # dependence-count-percentage voxel count below run on the cropped arrays.
        data, mask, _ = _maybe_crop_to_bbox(data, mask, None)
        matrices = _one_matrix(data, mask, n_bins, "ngldm", ngldm_alpha=ngldm_alpha)
        ngldm, mask = matrices["ngldm"], matrices["roi"]
    else:
        ngldm = ngldm_matrix

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
) -> dict[str, float]:
    """
    Calculate all texture features (GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM).

    This is a convenience wrapper that computes all texture matrices and then
    extracts all available features.

    Args:
        disc_array: Discretised image array.
        mask_array: Mask array (ROI). Nonzero values indicate ROI membership.
        n_bins: Number of bins.
        distance_mask_array: Optional mask for GLDZM distance calculation.
                             If None, mask_array is used.
        ngldm_alpha: The coarseness parameter α for NGLDM. Two grey levels are considered
            dependent if their absolute difference is ≤ α. Default is 0 (IBSI standard).

    Returns:
        Dictionary of all texture features.

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
        ```
    """
    results = {}

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
        compact=True,
        planar=planar,
    )
    # The bool ROI gives the same ROI voxel counts as mask_c, from a fast count.
    roi = texture_matrices["roi"]

    # GLCM (the texture ROI, in one mask type and layout)
    results.update(
        calculate_glcm_features(
            disc_c, roi.view(np.uint8), n_bins, glcm_matrix=texture_matrices["glcm"]
        )
    )

    # GLRLM
    results.update(
        calculate_glrlm_features(
            disc_c,
            roi,
            n_bins,
            glrlm_matrix=texture_matrices["glrlm"],
            n_directions=_directions(planar).size,
        )
    )

    # GLSZM
    results.update(_glszm_features_from_cells(texture_matrices["glszm_cells"], roi))

    # GLDZM
    results.update(
        calculate_gldzm_features(
            disc_c,
            roi,
            n_bins,
            gldzm_matrix=texture_matrices["gldzm"],
            distance_mask=(distmask_c if distmask_c is not None else mask_c),
        )
    )

    # NGTDM
    results.update(
        calculate_ngtdm_features(
            disc_c,
            mask_c,
            n_bins,
            ngtdm_matrices=(texture_matrices["ngtdm_s"], texture_matrices["ngtdm_n"]),
        )
    )
    # NGLDM
    results.update(
        calculate_ngldm_features(disc_c, roi, n_bins, ngldm_matrix=texture_matrices["ngldm"])
    )

    return results
