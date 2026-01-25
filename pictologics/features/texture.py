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

from typing import Any, Optional, cast

import numba
import numpy as np
from numba import jit, prange
from numba.np.ufunc.parallel import get_thread_id
from numpy import typing as npt
from scipy.ndimage import distance_transform_cdt

from ._utils import compute_nonzero_bbox


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

    union = mask != 0
    if distance_mask is not None:
        union = union | (distance_mask != 0)

    bbox = compute_nonzero_bbox(union)
    if bbox is None:
        return data, mask, distance_mask

    data_c = data[bbox]
    mask_c = mask[bbox]
    dist_c = distance_mask[bbox] if distance_mask is not None else None
    return data_c, mask_c, dist_c


# --- Zone Features Buffer Pool ---
# Pre-allocated buffers for _calculate_zone_features_numba to reduce allocation overhead
class _ZoneBufferPool:
    """Buffer pool for zone feature calculation to avoid repeated allocations."""

    _instance: Optional["_ZoneBufferPool"] = None

    def __init__(self) -> None:
        self._max_zones = 0
        self._res_gl: Optional[npt.NDArray[np.floating[Any]]] = None
        self._res_size: Optional[npt.NDArray[np.floating[Any]]] = None
        self._res_dist: Optional[npt.NDArray[np.floating[Any]]] = None
        self._stack: Optional[npt.NDArray[np.floating[Any]]] = None

    @classmethod
    def get_instance(cls) -> "_ZoneBufferPool":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_buffers(self, max_zones: int) -> tuple[
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
    ]:
        """
        Get pre-allocated buffers, resizing if necessary.

        Args:
            max_zones: Maximum number of zones (depth * height * width).

        Returns:
            Tuple of (res_gl, res_size, res_dist, stack) arrays.
        """
        if max_zones > self._max_zones:
            # Need to allocate larger buffers
            self._max_zones = max_zones
            self._res_gl = np.zeros(max_zones, dtype=np.int32)
            self._res_size = np.zeros(max_zones, dtype=np.int32)
            self._res_dist = np.zeros(max_zones, dtype=np.float64)
            self._stack = np.zeros(max_zones, dtype=np.int32)
        else:
            # Reuse existing buffers - just reset to zero
            assert self._res_gl is not None
            self._res_gl[:max_zones] = 0
            assert self._res_size is not None
            self._res_size[:max_zones] = 0
            assert self._res_dist is not None
            self._res_dist[:max_zones] = 0.0

        assert self._res_gl is not None
        assert self._res_size is not None
        assert self._res_dist is not None
        assert self._stack is not None
        return self._res_gl, self._res_size, self._res_dist, self._stack


# --- Combined Local Features Kernel ---

# Pre-calculate 26-neighbor offsets for NGTDM/NGLDM
# Generate all combinations of -1, 0, 1 using mgrid
_z, _y, _x = np.mgrid[-1:2, -1:2, -1:2]
_offsets = np.stack([_z.ravel(), _y.ravel(), _x.ravel()], axis=1)
# Remove the center pixel (0, 0, 0) and ensure int32 type
OFFSETS_26 = _offsets[np.any(_offsets != 0, axis=1)].astype(np.int32)

# Define 6-neighbor offsets (Manhattan distance 1)
_manhattan_dist = (
    np.abs(OFFSETS_26[:, 0]) + np.abs(OFFSETS_26[:, 1]) + np.abs(OFFSETS_26[:, 2])
)
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


@jit(nopython=True, parallel=True, fastmath=True, cache=True, error_model="numpy")  # type: ignore
def _calculate_local_features_numba(
    data_int: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    calc_glcm: bool,
    calc_glrlm: bool,
    calc_ngtdm: bool,
    calc_ngldm: bool,
    offsets_26: npt.NDArray[np.floating[Any]],
    directions_13: npt.NDArray[np.floating[Any]],
    ngldm_alpha: int,
    n_threads: int,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """
    Calculate GLCM, GLRLM, NGTDM, and NGLDM in a single pass.
    Optimized with parallel execution, thread-local storage, and interior loop optimization.

    Returns:
        glcm: (n_dirs, n_bins, n_bins)
        glrlm: (n_dirs, n_bins, max_run_length)
        ngtdm_s: (n_bins,)
        ngtdm_n: (n_bins,)
        ngldm: (n_bins, n_dependence)
    """
    depth, height, width = data_int.shape
    n_dirs = 13
    max_dim = max(depth, height, width)

    # Initialize thread-local result arrays conditionally.
    if calc_glcm:
        glcm_local = np.zeros((n_threads, n_dirs, n_bins, n_bins), dtype=np.uint32)
    else:
        glcm_local = np.zeros((1, 1, 1, 1), dtype=np.uint32)

    if calc_glrlm:
        glrlm_local = np.zeros(
            (n_threads, n_dirs, n_bins, max_dim + 1), dtype=np.uint32
        )
    else:
        glrlm_local = np.zeros((1, 1, 1, 1), dtype=np.uint32)

    if calc_ngtdm:
        ngtdm_s_local = np.zeros((n_threads, n_bins), dtype=np.float64)
        ngtdm_n_local = np.zeros((n_threads, n_bins), dtype=np.float64)
    else:
        ngtdm_s_local = np.zeros((1, 1), dtype=np.float64)
        ngtdm_n_local = np.zeros((1, 1), dtype=np.float64)

    if calc_ngldm:
        ngldm_local = np.zeros((n_threads, n_bins, 27), dtype=np.uint32)
    else:
        ngldm_local = np.zeros((1, 1, 1), dtype=np.uint32)

    # Determine safe margin for interior loop
    # NGTDM/NGLDM use 1-neighborhood. GLCM/GLRLM with dist=1 also use 1-neighborhood.
    margin = 1

    # Ensure margin is within bounds
    if margin >= min(depth, height, width) // 2:
        margin = 0  # Fallback to full checks if image is too small relative to margin

    # Pre-compute which z-slices have any ROI voxels (for slice-level skipping)
    z_has_voxels = np.zeros(depth, dtype=np.uint8)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if mask[z, y, x] > 0:
                    z_has_voxels[z] = 1
                    break
            if z_has_voxels[z] > 0:
                break

    # Iterate over all voxels (parallelized over z)
    for z in prange(depth):
        # Skip entire z-slices with no ROI voxels
        if z_has_voxels[z] == 0:
            continue

        tid = get_thread_id()

        # Check if Z is in the safe interior
        z_safe = (z >= margin) and (z < depth - margin)

        for y in range(height):
            y_safe = (y >= margin) and (y < height - margin)

            # Determine X range for fast path (interior) vs slow path (boundary)
            # If Z and Y are safe, we can have a safe X range
            if margin > 0 and z_safe and y_safe:
                # Split X loop into: Left Boundary, Interior, Right Boundary

                # 1. Left Boundary
                for x in range(0, margin):
                    _process_voxel(
                        x,
                        y,
                        z,
                        data_int,
                        mask,
                        n_bins,
                        calc_glcm,
                        calc_glrlm,
                        calc_ngtdm,
                        calc_ngldm,
                        glcm_local,
                        glrlm_local,
                        ngtdm_s_local,
                        ngtdm_n_local,
                        ngldm_local,
                        offsets_26,
                        directions_13,
                        tid,
                        depth,
                        height,
                        width,
                        max_dim,
                        False,
                        ngldm_alpha,
                    )

                # 2. Interior (Safe)
                for x in range(margin, width - margin):
                    _process_voxel(
                        x,
                        y,
                        z,
                        data_int,
                        mask,
                        n_bins,
                        calc_glcm,
                        calc_glrlm,
                        calc_ngtdm,
                        calc_ngldm,
                        glcm_local,
                        glrlm_local,
                        ngtdm_s_local,
                        ngtdm_n_local,
                        ngldm_local,
                        offsets_26,
                        directions_13,
                        tid,
                        depth,
                        height,
                        width,
                        max_dim,
                        True,
                        ngldm_alpha,
                    )

                # 3. Right Boundary
                for x in range(width - margin, width):
                    _process_voxel(
                        x,
                        y,
                        z,
                        data_int,
                        mask,
                        n_bins,
                        calc_glcm,
                        calc_glrlm,
                        calc_ngtdm,
                        calc_ngldm,
                        glcm_local,
                        glrlm_local,
                        ngtdm_s_local,
                        ngtdm_n_local,
                        ngldm_local,
                        offsets_26,
                        directions_13,
                        tid,
                        depth,
                        height,
                        width,
                        max_dim,
                        False,
                        ngldm_alpha,
                    )
            else:
                # Full Boundary Row (Z or Y is boundary, or margin is 0)
                for x in range(width):
                    _process_voxel(
                        x,
                        y,
                        z,
                        data_int,
                        mask,
                        n_bins,
                        calc_glcm,
                        calc_glrlm,
                        calc_ngtdm,
                        calc_ngldm,
                        glcm_local,
                        glrlm_local,
                        ngtdm_s_local,
                        ngtdm_n_local,
                        ngldm_local,
                        offsets_26,
                        directions_13,
                        tid,
                        depth,
                        height,
                        width,
                        max_dim,
                        False,
                        ngldm_alpha,
                    )

    # Aggregate results from all threads.
    # Keep stable output shapes even when a matrix is not requested.
    if calc_glcm:
        glcm = np.sum(glcm_local, axis=0)
    else:
        glcm = np.zeros((n_dirs, n_bins, n_bins), dtype=np.uint64)

    if calc_glrlm:
        glrlm = np.sum(glrlm_local, axis=0)
    else:
        glrlm = np.zeros((n_dirs, n_bins, 1), dtype=np.uint64)

    if calc_ngtdm:
        ngtdm_s = np.sum(ngtdm_s_local, axis=0)
        ngtdm_n = np.sum(ngtdm_n_local, axis=0)
    else:
        ngtdm_s = np.zeros((n_bins,), dtype=np.float64)
        ngtdm_n = np.zeros((n_bins,), dtype=np.float64)

    if calc_ngldm:
        ngldm = np.sum(ngldm_local, axis=0)
    else:
        ngldm = np.zeros((n_bins, 27), dtype=np.uint64)

    return glcm, glrlm, ngtdm_s, ngtdm_n, ngldm


@jit(nopython=True, inline="always", fastmath=True, cache=True, error_model="numpy")  # type: ignore
def _process_voxel(
    x: int,
    y: int,
    z: int,
    data_int: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    calc_glcm: bool,
    calc_glrlm: bool,
    calc_ngtdm: bool,
    calc_ngldm: bool,
    glcm_local: npt.NDArray[np.floating[Any]],
    glrlm_local: npt.NDArray[np.floating[Any]],
    ngtdm_s_local: npt.NDArray[np.floating[Any]],
    ngtdm_n_local: npt.NDArray[np.floating[Any]],
    ngldm_local: npt.NDArray[np.floating[Any]],
    offsets_26: npt.NDArray[np.floating[Any]],
    directions_13: npt.NDArray[np.floating[Any]],
    tid: int,
    depth: int,
    height: int,
    width: int,
    max_dim: int,
    is_safe: bool,
    ngldm_alpha: int = 0,
) -> None:
    """Process single voxel for GLCM/GLRLM/NGTDM/NGLDM; inlined for performance."""
    if mask[z, y, x] == 0:
        return

    i_val = data_int[z, y, x]
    if not (0 <= i_val < n_bins):
        return

    # --- NGTDM & NGLDM (Neighborhood Analysis) ---
    if calc_ngtdm or calc_ngldm:
        s_val = 0.0
        dependence_count = 1
        valid_neighbors = False
        neighbor_sum = 0.0
        neighbor_count = 0

        # Iterate over 26 neighbors
        for i in range(26):
            dz = offsets_26[i, 0]
            dy = offsets_26[i, 1]
            dx = offsets_26[i, 2]
            nz, ny, nx = z + dz, y + dy, x + dx

            if is_safe:
                # No boundary checks needed for 1-neighborhood
                if mask[nz, ny, nx]:
                    valid_neighbors = True
                    n_val = data_int[nz, ny, nx]
                    if calc_ngtdm:
                        neighbor_sum += float(n_val) + 1.0
                        neighbor_count += 1
                    if calc_ngldm:
                        if abs(int(n_val) - int(i_val)) <= ngldm_alpha:
                            dependence_count += 1
            else:
                # Boundary checks needed
                if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                    if mask[nz, ny, nx]:
                        valid_neighbors = True
                        n_val = data_int[nz, ny, nx]
                        if calc_ngtdm:
                            neighbor_sum += float(n_val) + 1.0
                            neighbor_count += 1
                        if calc_ngldm:
                            if abs(int(n_val) - int(i_val)) <= ngldm_alpha:
                                dependence_count += 1

        if calc_ngtdm and valid_neighbors and neighbor_count > 0:
            avg = neighbor_sum / neighbor_count
            s_val = abs(float(i_val) + 1.0 - avg)
            ngtdm_n_local[tid, i_val] += 1
            ngtdm_s_local[tid, i_val] += s_val

        if calc_ngldm:
            if dependence_count > 0:
                ngldm_local[tid, i_val, dependence_count - 1] += 1

    # --- GLCM & GLRLM (Directional Analysis) ---
    if calc_glcm or calc_glrlm:
        # Iterate over 13 directions
        # Using range(13) and global array access is efficient enough and avoids literal_unroll issues
        for d in range(13):
            dz = directions_13[d, 0]
            dy = directions_13[d, 1]
            dx = directions_13[d, 2]

            # GLCM (dist=1)
            if calc_glcm:
                nz, ny, nx = z + dz, y + dy, x + dx

                if is_safe:
                    # Safe because margin >= 1
                    if mask[nz, ny, nx]:
                        j_val = data_int[nz, ny, nx]
                        if 0 <= j_val < n_bins:
                            glcm_local[tid, d, i_val, j_val] += 1
                else:
                    if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                        if mask[nz, ny, nx]:
                            j_val = data_int[nz, ny, nx]
                            if 0 <= j_val < n_bins:
                                glcm_local[tid, d, i_val, j_val] += 1

            # GLRLM
            if calc_glrlm:
                # Check start condition
                pz, py, px = z - dz, y - dy, x - dx
                is_start = False

                if is_safe:
                    # Safe to access previous voxel without bounds check
                    if not mask[pz, py, px]:
                        is_start = True
                    elif data_int[pz, py, px] != i_val:
                        is_start = True
                else:
                    if not (
                        0 <= pz < depth
                        and 0 <= py < height
                        and 0 <= px < width
                        and mask[pz, py, px]
                    ):
                        is_start = True
                    elif data_int[pz, py, px] != i_val:
                        is_start = True

                if is_start:
                    length = 1
                    cz, cy, cx = z + dz, y + dy, x + dx

                    # Optimized GLRLM loop with max_steps
                    if 0 <= cz < depth and 0 <= cy < height and 0 <= cx < width:
                        steps_z = (
                            (depth - 1 - cz)
                            if dz == 1
                            else (cz if dz == -1 else max_dim)
                        )
                        steps_y = (
                            (height - 1 - cy)
                            if dy == 1
                            else (cy if dy == -1 else max_dim)
                        )
                        steps_x = (
                            (width - 1 - cx)
                            if dx == 1
                            else (cx if dx == -1 else max_dim)
                        )

                        max_steps = min(steps_z, steps_y, steps_x)

                        for _ in range(max_steps + 1):
                            if not mask[cz, cy, cx]:
                                break
                            if data_int[cz, cy, cx] != i_val:
                                break
                            length += 1
                            cz += dz
                            cy += dy
                            cx += dx

                    if length <= max_dim:
                        glrlm_local[tid, d, i_val, length] += 1


def calculate_all_texture_matrices(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    n_bins: int,
    distance_mask: Optional[npt.NDArray[np.floating[Any]]] = None,
    ngldm_alpha: int = 0,
) -> dict[str, Any]:
    """
    Calculate all texture matrices (GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM) in an optimized single pass.

    This function serves as the computational backbone for texture analysis. It computes the raw
    matrices required to extract specific texture features. By aggregating these calculations,
    it minimizes the number of passes over the image data, significantly improving performance.

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
            Values should be integers in the range [1, n_bins].
        mask (npt.NDArray[np.floating[Any]]): The 3D binary mask array defining the Region of Interest (ROI).
            Must have the same shape as `data`. Non-zero values indicate the ROI.
        n_bins (int): The number of grey levels used for discretization (e.g., 16, 32, 64).
            This determines the size of the resulting matrices.
        distance_mask (Optional[npt.NDArray[np.floating[Any]]]): Optional mask used to calculate the distance map for GLDZM.
            If None, `mask` is used. This allows calculating distances based on the morphological mask
            while analyzing intensities from the intensity mask (e.g., after outlier filtering).
        ngldm_alpha (int): The coarseness parameter α for NGLDM calculation. Two grey levels are
            considered dependent if their absolute difference is ≤ α. Default is 0 (exact match),
            which is the IBSI standard. Use α=1 for tolerance of ±1 grey level difference.

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
    # Fast exit for empty ROI
    if not bool(np.any(mask != 0)):
        return {
            "glcm": np.zeros((13, n_bins, n_bins), dtype=np.uint64),
            "glrlm": np.zeros((13, n_bins, 1), dtype=np.uint64),
            "ngtdm_s": np.zeros((n_bins,), dtype=np.float64),
            "ngtdm_n": np.zeros((n_bins,), dtype=np.float64),
            "ngldm": np.zeros((n_bins, 27), dtype=np.uint64),
            "glszm": np.zeros((n_bins, 1), dtype=np.uint32),
            "gldzm": np.zeros((n_bins, 1), dtype=np.uint32),
        }

    # Crop to ROI bounding box (union with distance_mask when provided) to reduce memory traffic.
    data_c, mask_c, distmask_c = _maybe_crop_to_bbox(data, mask, distance_mask)

    # Use a compact mask representation for kernels.
    if mask_c.dtype == np.uint8:
        mask_u8 = mask_c
    else:
        mask_u8 = (mask_c != 0).astype(np.uint8)
    # 1. Local Features (GLCM, GLRLM, NGTDM, NGLDM)
    # Pre-cast data to smallest possible int type (0-based)
    # Input data is 1-based, so we subtract 1.
    if n_bins <= 256:
        data_int = (data_c - 1).astype(np.uint8)
    else:
        data_int = (data_c - 1).astype(np.int32)

    try:
        n_threads = int(numba.config.NUMBA_NUM_THREADS)
    except (ValueError, TypeError):
        n_threads = 1  # Fallback

    glcm, glrlm, ngtdm_s, ngtdm_n, ngldm = _calculate_local_features_numba(
        data_int,
        mask_u8,
        n_bins,
        calc_glcm=True,
        calc_glrlm=True,
        calc_ngtdm=True,
        calc_ngldm=True,
        offsets_26=OFFSETS_26,
        directions_13=DIRECTIONS_13,
        ngldm_alpha=ngldm_alpha,
        n_threads=n_threads,
    )

    # 2. Zone Features (GLSZM, GLDZM)
    # Pre-calculate distance map for GLDZM
    # Use distance_mask if provided, else mask
    d_mask = distmask_c if distmask_c is not None else mask_u8

    # Pad the mask with 0s to ensure the image border is treated as an edge.
    mask_bool = d_mask > 0
    mask_padded = np.pad(mask_bool, 1, mode="constant", constant_values=0)
    dist_map_padded = distance_transform_cdt(mask_padded, metric="taxicab").astype(
        np.int32
    )
    dist_map = dist_map_padded[1:-1, 1:-1, 1:-1]

    glszm, gldzm = calculate_zone_features(
        data_c,
        mask_u8,
        dist_map,
        n_bins,
        calc_glszm=True,
        calc_gldzm=True,
    )

    return {
        "glcm": glcm,
        "glrlm": glrlm,
        "ngtdm_s": ngtdm_s,
        "ngtdm_n": ngtdm_n,
        "ngldm": ngldm,
        "glszm": glszm,
        "gldzm": gldzm,
    }


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
            mask (npt.NDArray[np.floating[Any]]): The 3D binary mask array defining the ROI.
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
        data_c, mask_c, _ = _maybe_crop_to_bbox(data, mask, None)
        if mask_c.dtype == np.uint8:
            mask_u8 = mask_c
        else:
            mask_u8 = (mask_c != 0).astype(np.uint8)

        if n_bins <= 256:
            data_int = (data_c - 1).astype(np.uint8)
        else:
            data_int = (data_c - 1).astype(np.int32)

        # Determine n_threads for JIT call
        try:
            n_threads = int(numba.config.NUMBA_NUM_THREADS)
        except (ValueError, TypeError):
            n_threads = 1  # Fallback

        # Call combined kernel to calculate only GLCM
        glcm, _, _, _, _ = _calculate_local_features_numba(
            data_int,
            mask_u8,
            n_bins,
            calc_glcm=True,
            calc_glrlm=False,
            calc_ngtdm=False,
            calc_ngldm=False,
            offsets_26=OFFSETS_26,
            directions_13=DIRECTIONS_13,
            ngldm_alpha=0,
            n_threads=n_threads,
        )
    else:
        glcm = glcm_matrix

    # Merge (Sum) -> IAZD
    glcm_sum = np.sum(glcm, axis=0)
    glcm_sym = glcm_sum + glcm_sum.T

    # Normalize
    total_sum = np.sum(glcm_sym)
    if total_sum == 0:
        return {}

    P = glcm_sym / total_sum

    # Indices (0-based from np.indices, convert to 1-based for IBSI)
    i_idx, j_idx = np.indices((n_bins, n_bins))
    I = i_idx + 1  # noqa: E741
    J = j_idx + 1

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
    p_diff = np.bincount(k_diff_flat, weights=P_flat)

    mu_diff = features["difference_average_TF7R"]
    k_vals = np.arange(n_bins)
    features["difference_variance_D3YU"] = np.sum(((k_vals - mu_diff) ** 2) * p_diff)

    # Difference Entropy - NTRS
    mask_pd = p_diff > 0
    features["difference_entropy_NTRS"] = -np.sum(
        p_diff[mask_pd] * np.log2(p_diff[mask_pd])
    )

    # Sum Average - ZGXS
    k_sum_grid = I + J

    # Optimized using bincount
    k_sum_flat = k_sum_grid.ravel().astype(np.int32)
    # P_flat is already defined in Difference Variance block
    p_sum_full = np.bincount(k_sum_flat, weights=P_flat)

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
    features["contrast_ACUI"] = np.sum(((I - J) ** 2) * P)

    # Dissimilarity - 8S9J
    features["dissimilarity_8S9J"] = np.sum(np.abs(I - J) * P)

    # Inverse Difference - IB1Z
    features["inverse_difference_IB1Z"] = np.sum(P / (1 + np.abs(I - J)))

    roi_data = data[mask > 0]
    if len(roi_data) > 0:
        Ng_eff = int(np.max(roi_data) - np.min(roi_data) + 1)
    else:
        Ng_eff = 1  # Fallback

    # Normalised Inverse Difference - NDRX
    features["normalised_inverse_difference_NDRX"] = np.sum(
        P / (1 + np.abs(I - J) / Ng_eff)
    )

    # Inverse Difference Moment - WF0Z
    features["inverse_difference_moment_WF0Z"] = np.sum(P / (1 + (I - J) ** 2))

    # Normalised Inverse Difference Moment - 1QCO
    features["normalised_inverse_difference_moment_1QCO"] = np.sum(
        P / (1 + ((I - J) ** 2) / (Ng_eff**2))
    )

    # Inverse Variance - E8JP
    mask_neq = I != J
    features["inverse_variance_E8JP"] = np.sum(
        P[mask_neq] / ((I[mask_neq] - J[mask_neq]) ** 2)
    )

    # Correlation - NI2N
    term1 = np.sum((I - mu) * (J - mu) * P)
    if features["joint_variance_UR99"] != 0:
        features["correlation_NI2N"] = term1 / features["joint_variance_UR99"]
    else:
        features["correlation_NI2N"] = (
            1.0  # Or NaN? IBSI doesn't specify for 0 variance.
        )

    # Autocorrelation - QWB0
    features["autocorrelation_QWB0"] = np.sum(I * J * P)

    # Cluster Tendency - DG8W
    features["cluster_tendency_DG8W"] = np.sum(((I + J - 2 * mu) ** 2) * P)

    # Cluster Shade - 7NFM
    features["cluster_shade_7NFM"] = np.sum(((I + J - 2 * mu) ** 3) * P)

    # Cluster Prominence - AE86
    features["cluster_prominence_AE86"] = np.sum(((I + J - 2 * mu) ** 4) * P)

    # Information Correlation 1 - R8DG
    HXY = features["joint_entropy_TU9B"]
    p_x = np.sum(P, axis=1)
    mask_px = p_x > 0
    HX = -np.sum(p_x[mask_px] * np.log2(p_x[mask_px]))

    HXY1 = -np.sum(P[mask_p] * np.log2(p_x[I[mask_p] - 1] * p_x[J[mask_p] - 1]))

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
) -> dict[str, float]:
    """
    Calculate Grey Level Run Length Matrix (GLRLM) features.

    The GLRLM quantifies grey level runs, which are defined as the length in number of pixels,
    of consecutive pixels that have the same grey level value.
    This implementation computes features based on the 3D merged GLRLM (averaged over all 13 directions).

    Args:
        data (npt.NDArray[np.floating[Any]]): The 3D image array containing discretised grey levels.
        mask (npt.NDArray[np.floating[Any]]): The 3D binary mask array defining the ROI.
        n_bins (int): The number of grey levels.
        glrlm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated GLRLM matrix.

    Returns:
        dict[str, float]: A dictionary of calculated GLRLM features.
            Example keys: 'short_runs_emphasis_22OV', 'grey_level_non_uniformity_R5YN'.
    """
    if glrlm_matrix is None:
        if n_bins <= 256:
            data_int = (data - 1).astype(np.uint8)
        else:
            data_int = (data - 1).astype(np.int32)  # pragma: no cover

        # Determine n_threads for JIT call
        try:
            n_threads = int(numba.config.NUMBA_NUM_THREADS)
        except (ValueError, TypeError):
            n_threads = 1  # Fallback

        # Call combined kernel
        _, glrlm, _, _, _ = _calculate_local_features_numba(
            data_int,
            mask,
            n_bins,
            calc_glcm=False,
            calc_glrlm=True,
            calc_ngtdm=False,
            calc_ngldm=False,
            offsets_26=OFFSETS_26,
            directions_13=DIRECTIONS_13,
            ngldm_alpha=0,
            n_threads=n_threads,
        )
    else:
        glrlm = glrlm_matrix

    # Merge (Sum) -> IAZD
    glrlm_sum = np.sum(glrlm, axis=0)

    # Remove length 0 (column 0)
    glrlm = glrlm_sum[:, 1:]

    N_runs = np.sum(glrlm)
    if N_runs == 0:
        return {}

    P = glrlm / N_runs

    n_g, n_r = glrlm.shape
    i_idx, j_idx = np.indices((n_g, n_r))
    I = i_idx + 1  # noqa: E741
    J = j_idx + 1

    features = {}

    # Short Run Emphasis (SRE) - 22OV
    features["short_runs_emphasis_22OV"] = np.sum(P / (J**2))

    # Long Run Emphasis (LRE) - W4KF
    features["long_runs_emphasis_W4KF"] = np.sum(P * (J**2))

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
    n_voxels = np.sum(mask)
    n_dirs = 13  # Fixed for 3D
    features["run_percentage_9ZK5"] = N_runs / (n_voxels * n_dirs)

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
    features["low_grey_level_run_emphasis_V3SW"] = np.sum(P / (I**2))

    # High Grey Level Run Emphasis (HGLRE) - G3QZ
    features["high_grey_level_run_emphasis_G3QZ"] = np.sum(P * (I**2))

    # Short Run Low Grey Level Emphasis (SRLGLE) - HTZT
    features["short_run_low_grey_level_emphasis_HTZT"] = np.sum(P / ((I**2) * (J**2)))

    # Short Run High Grey Level Emphasis (SRHGLE) - GD3A
    features["short_run_high_grey_level_emphasis_GD3A"] = np.sum(P * (I**2) / (J**2))

    # Long Run Low Grey Level Emphasis (LRLGLE) - IVPO
    features["long_run_low_grey_level_emphasis_IVPO"] = np.sum(P * (J**2) / (I**2))

    # Long Run High Grey Level Emphasis (LRHGLE) - 3KUM
    features["long_run_high_grey_level_emphasis_3KUM"] = np.sum(P * (I**2) * (J**2))

    return features


# --- Combined Zone Features Kernel ---


@jit(nopython=True, fastmath=True, cache=True)  # type: ignore
def _calculate_zone_features_numba(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    dist_map: npt.NDArray[np.floating[Any]],
    n_bins: int,
    res_gl: npt.NDArray[np.floating[Any]],
    res_size: npt.NDArray[np.floating[Any]],
    res_dist: npt.NDArray[np.floating[Any]],
    stack: npt.NDArray[np.floating[Any]],
    calc_glszm: bool = True,
    calc_gldzm: bool = True,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    Calculate GLSZM and GLDZM in a single pass using an optimized 1D flattened approach.

    Optimization Strategy:
    1. Flatten 3D arrays to 1D with 1-voxel padding.
       - Padding avoids boundary checks in the inner loop.
       - Flattening improves cache locality and simplifies indexing.
    2. Reuse the mask array to track visited voxels.
       - Sets mask to 0 after visiting, avoiding a separate boolean array.
    3. Accept pre-allocated result buffers to minimize allocation overhead.

    Args:
        data: 3D discretized image data.
        mask: 3D mask array (will be modified - set voxels to 0 when visited).
        dist_map: 3D distance map for GLDZM.
        n_bins: Number of grey level bins.
        res_gl: Pre-allocated buffer for zone grey levels (size: max_zones).
        res_size: Pre-allocated buffer for zone sizes (size: max_zones).
        res_dist: Pre-allocated buffer for zone distances (size: max_zones).
        stack: Pre-allocated DFS stack (size: max_zones).
        calc_glszm: Whether to calculate GLSZM.
        calc_gldzm: Whether to calculate GLDZM.
    """
    depth, height, width = data.shape

    # 1. Setup Padded Dimensions
    p_depth = depth + 2
    p_height = height + 2
    p_width = width + 2
    flat_size = p_depth * p_height * p_width

    # Strides for the padded array
    stride_z = p_height * p_width
    stride_y = p_width

    # 2. Allocate and Populate Padded Arrays
    padded_mask = np.zeros((p_depth, p_height, p_width), dtype=np.uint8)
    padded_mask[1:-1, 1:-1, 1:-1] = mask
    flat_mask = padded_mask.ravel()

    padded_data = np.zeros((p_depth, p_height, p_width), dtype=data.dtype)
    padded_data[1:-1, 1:-1, 1:-1] = data
    flat_data = padded_data.ravel()

    if calc_gldzm:
        padded_dist = np.zeros((p_depth, p_height, p_width), dtype=dist_map.dtype)
        padded_dist[1:-1, 1:-1, 1:-1] = dist_map
        flat_dist = padded_dist.ravel()
    else:
        # Dummy array to satisfy type checker
        flat_dist = np.zeros(1, dtype=dist_map.dtype)

    # 3. Pre-calculate 1D Offsets for 26-connectivity
    offsets = np.zeros(26, dtype=np.int32)
    idx = 0
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets[idx] = dz * stride_z + dy * stride_y + dx
                idx += 1

    # 4. Use pre-allocated result buffers (passed as arguments)
    zone_count = 0

    # 5. Main Loop
    # Iterate over the flattened array.
    # Padding ensures we don't need to check bounds for neighbors.
    # Mask ensures we only process valid ROI voxels.

    for i in range(flat_size):
        # Skip background or already visited
        if flat_mask[i] == 0:
            continue

        # Found a new zone
        gl = flat_data[i]

        # Check grey level validity (1 to n_bins)
        if gl < 1 or gl > n_bins:
            flat_mask[i] = 0  # Mark as visited/invalid
            continue

        # Start DFS
        stack_ptr = 0
        stack[stack_ptr] = i
        stack_ptr += 1
        flat_mask[i] = 0  # Mark visited

        size = 0
        min_dist = flat_dist[i] if calc_gldzm else 0

        while stack_ptr > 0:
            stack_ptr -= 1
            curr_idx = stack[stack_ptr]
            size += 1

            if calc_gldzm:
                d = flat_dist[curr_idx]
                if d < min_dist:
                    min_dist = d

            # Check all 26 neighbors
            for k in range(26):
                neighbor_idx = curr_idx + offsets[k]

                # If mask is non-zero, it's a valid unvisited voxel
                if flat_mask[neighbor_idx] != 0:
                    if flat_data[neighbor_idx] == gl:
                        flat_mask[neighbor_idx] = 0  # Mark visited
                        stack[stack_ptr] = neighbor_idx
                        stack_ptr += 1

        # Store zone properties
        res_gl[zone_count] = gl
        if calc_glszm:
            res_size[zone_count] = size
        if calc_gldzm:
            res_dist[zone_count] = min_dist
        zone_count += 1

    # 6. Build Output Matrices

    # Build GLSZM
    glszm = np.zeros((n_bins, 1), dtype=np.uint32)
    if calc_glszm and zone_count > 0:
        max_sz = 0
        for i in range(zone_count):
            s = res_size[i]
            if s > max_sz:
                max_sz = s

        glszm = np.zeros((n_bins, max_sz), dtype=np.uint32)
        for i in range(zone_count):
            gl = res_gl[i]
            sz = res_size[i]
            glszm[gl - 1, sz - 1] += 1

    # Build GLDZM
    gldzm = np.zeros((n_bins, 1), dtype=np.uint32)
    if calc_gldzm and zone_count > 0:
        max_dist_val = 0
        for i in range(zone_count):
            d = int(res_dist[i])
            if d > max_dist_val:
                max_dist_val = d

        if max_dist_val == 0:
            max_dist_val = 1

        gldzm = np.zeros((n_bins, max_dist_val), dtype=np.uint32)
        for i in range(zone_count):
            gl = res_gl[i]
            d = int(res_dist[i])
            if d > 0:
                gldzm[gl - 1, d - 1] += 1

    return glszm, gldzm  # type: ignore[return-value]


def calculate_zone_features(
    data: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[np.floating[Any]],
    dist_map: npt.NDArray[np.floating[Any]],
    n_bins: int,
    calc_glszm: bool = True,
    calc_gldzm: bool = True,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    Wrapper for _calculate_zone_features_numba with buffer pooling.

    This function manages pre-allocated buffers to reduce memory allocation
    overhead for repeated calls (e.g., during batch processing).

    Args:
        data: 3D discretized image data.
        mask: 3D mask array (not modified - copied internally by JIT function).
        dist_map: 3D distance map for GLDZM.
        n_bins: Number of grey level bins.
        calc_glszm: Whether to calculate GLSZM.
        calc_gldzm: Whether to calculate GLDZM.

    Returns:
        Tuple of (glszm, gldzm) matrices.
    """
    # For zone features, the worst-case number of zones is bounded by ROI voxel count.
    # Sizing buffers to full image volume is extremely costly for sparse ROIs.
    max_zones = int(np.count_nonzero(mask))
    if max_zones < 1:
        max_zones = 1

    # Get pre-allocated buffers from pool
    pool = _ZoneBufferPool.get_instance()
    res_gl, res_size, res_dist, stack = pool.get_buffers(max_zones)

    return cast(
        tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]],
        _calculate_zone_features_numba(
            data,
            mask,
            dist_map,
            n_bins,
            res_gl,
            res_size,
            res_dist,
            stack,
            calc_glszm,
            calc_gldzm,
        ),
    )


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
        mask (npt.NDArray[np.floating[Any]]): The 3D binary mask array defining the ROI.
        n_bins (int): The number of grey levels.
        glszm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated GLSZM matrix.

    Returns:
        dict[str, float]: A dictionary of calculated GLSZM features.
            Example keys: 'small_zone_emphasis_P001', 'zone_percentage_P30P'.
    """
    if glszm_matrix is None:
        data_c, mask_c, _ = _maybe_crop_to_bbox(data, mask, None)
        if mask_c.dtype == np.uint8:
            mask_u8 = mask_c
        else:
            mask_u8 = (mask_c != 0).astype(np.uint8)

        # We need dist_map for the combined kernel signature, even if unused for GLSZM
        dummy_dist = np.zeros_like(data_c, dtype=np.int32)

        glszm, _ = calculate_zone_features(
            data_c, mask_u8, dummy_dist, n_bins, calc_glszm=True, calc_gldzm=False  # type: ignore[arg-type]
        )
    else:
        glszm = glszm_matrix

    N_zones = np.sum(glszm)
    if N_zones == 0:
        return {}

    P = glszm / N_zones

    n_g, n_s = glszm.shape
    i_idx, j_idx = np.indices((n_g, n_s))
    I = i_idx + 1  # noqa: E741
    J = j_idx + 1  # Zone size

    features = {}

    # Small Zone Emphasis (SZE) - P001
    features["small_zone_emphasis_P001"] = np.sum(P / (J**2))

    # Large Zone Emphasis (LZE) - 48P8
    features["large_zone_emphasis_48P8"] = np.sum(P * (J**2))

    # Grey Level Non-Uniformity (GLNU) - JNSA
    s_i = np.sum(glszm, axis=1)
    features["grey_level_non_uniformity_JNSA"] = np.sum(s_i**2) / N_zones

    # Normalised Grey Level Non-Uniformity (GLNN) - Y1RO
    features["normalised_grey_level_non_uniformity_Y1RO"] = np.sum(s_i**2) / (
        N_zones**2
    )

    # Zone Size Non-Uniformity (ZSNU) - 4JP3
    s_j = np.sum(glszm, axis=0)
    features["zone_size_non_uniformity_4JP3"] = np.sum(s_j**2) / N_zones

    # Normalised Zone Size Non-Uniformity (ZSNN) - VB3A
    features["normalised_zone_size_non_uniformity_VB3A"] = np.sum(s_j**2) / (N_zones**2)

    # Zone Percentage (ZP) - P30P
    n_voxels = np.sum(mask)
    features["zone_percentage_P30P"] = N_zones / n_voxels

    # Grey Level Variance (GLV) - BYLV
    mu_i = np.sum(I * P)
    features["grey_level_variance_BYLV"] = np.sum(((I - mu_i) ** 2) * P)

    # Zone Size Variance (ZSV) - 3NSA
    mu_j = np.sum(J * P)
    features["zone_size_variance_3NSA"] = np.sum(((J - mu_j) ** 2) * P)

    # Zone Size Entropy (ZSE) - GU8N
    mask_p = P > 0
    features["zone_size_entropy_GU8N"] = -np.sum(P[mask_p] * np.log2(P[mask_p]))

    # Low Grey Level Zone Emphasis (LGLZE) - XMSY
    features["low_grey_level_zone_emphasis_XMSY"] = np.sum(P / (I**2))

    # High Grey Level Zone Emphasis (HGLZE) - 5GN9
    features["high_grey_level_zone_emphasis_5GN9"] = np.sum(P * (I**2))

    # Small Zone Low Grey Level Emphasis (SZLGLE) - 5RAI
    features["small_zone_low_grey_level_emphasis_5RAI"] = np.sum(P / ((I**2) * (J**2)))

    # Small Zone High Grey Level Emphasis (SZHGLE) - HW1V
    features["small_zone_high_grey_level_emphasis_HW1V"] = np.sum(P * (I**2) / (J**2))

    # Large Zone Low Grey Level Emphasis (LZLGLE) - YH51
    features["large_zone_low_grey_level_emphasis_YH51"] = np.sum(P * (J**2) / (I**2))

    # Large Zone High Grey Level Emphasis (LZHGLE) - J17V
    features["large_zone_high_grey_level_emphasis_J17V"] = np.sum(P * (I**2) * (J**2))

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
        mask (npt.NDArray[np.floating[Any]]): The 3D binary mask array defining the ROI.
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
        # Calculate distance map
        # Use distance_mask if provided, else mask
        d_mask = distance_mask if distance_mask is not None else mask

        # Pad the mask with 0s to ensure the image border is treated as an edge.
        mask_bool = d_mask > 0
        mask_padded = np.pad(mask_bool, 1, mode="constant", constant_values=0)
        dist_map_padded = distance_transform_cdt(mask_padded, metric="taxicab").astype(
            np.int32
        )
        dist_map = dist_map_padded[1:-1, 1:-1, 1:-1]

        _, gldzm = calculate_zone_features(  # type: ignore[arg-type]
            data,
            mask,
            dist_map,
            n_bins,
            calc_glszm=False,
            calc_gldzm=True,
        )
    else:
        gldzm = gldzm_matrix

    N_zones = np.sum(gldzm)
    if N_zones == 0:
        return {}

    P = gldzm / N_zones

    n_g, n_d = gldzm.shape
    i_idx, j_idx = np.indices((n_g, n_d))
    I = i_idx + 1  # noqa: E741
    J = j_idx + 1  # Distance

    features = {}

    # Small Distance Emphasis (SDE) - 0GBI
    features["small_distance_emphasis_0GBI"] = np.sum(P / (J**2))

    # Large Distance Emphasis (LDE) - MB4I
    features["large_distance_emphasis_MB4I"] = np.sum(P * (J**2))

    # Grey Level Non-Uniformity (GLNU) - VFT7
    s_i = np.sum(gldzm, axis=1)
    features["grey_level_non_uniformity_VFT7"] = np.sum(s_i**2) / N_zones

    # Normalised Grey Level Non-Uniformity (GLNN) - 7HP3
    features["normalised_grey_level_non_uniformity_7HP3"] = np.sum(s_i**2) / (
        N_zones**2
    )

    # Zone Distance Non-Uniformity (ZDNU) - V294
    s_j = np.sum(gldzm, axis=0)
    features["zone_distance_non_uniformity_V294"] = np.sum(s_j**2) / N_zones

    # Normalised Zone Distance Non-Uniformity (ZDNN) - IATH
    features["normalised_zone_distance_non_uniformity_IATH"] = np.sum(s_j**2) / (
        N_zones**2
    )

    # Zone Percentage (ZP) - VIWW
    n_voxels = np.sum(mask)
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
    features["low_grey_level_zone_emphasis_S1RA"] = np.sum(P / (I**2))

    # High Grey Level Zone Emphasis (HGLZE) - K26C
    features["high_grey_level_zone_emphasis_K26C"] = np.sum(P * (I**2))

    # Small Distance Low Grey Level Emphasis (SDLGLE) - RUVG
    features["small_distance_low_grey_level_emphasis_RUVG"] = np.sum(
        P / ((I**2) * (J**2))
    )

    # Small Distance High Grey Level Emphasis (SDHGLE) - DKNJ
    features["small_distance_high_grey_level_emphasis_DKNJ"] = np.sum(
        P * (I**2) / (J**2)
    )

    # Large Distance Low Grey Level Emphasis (LDLGLE) - A7WM
    features["large_distance_low_grey_level_emphasis_A7WM"] = np.sum(
        P * (J**2) / (I**2)
    )

    # Large Distance High Grey Level Emphasis (LDHGLE) - KLTH
    features["large_distance_high_grey_level_emphasis_KLTH"] = np.sum(
        P * (I**2) * (J**2)
    )

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
        mask (npt.NDArray[np.floating[Any]]): The 3D binary mask array defining the ROI.
        n_bins (int): The number of grey levels.
        ngtdm_matrices (Optional[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]): Pre-calculated NGTDM matrices
            (sum of absolute differences `s`, and count `n`).

    Returns:
        dict[str, float]: A dictionary of calculated NGTDM features.
            Example keys: 'coarseness_QCDE', 'contrast_65HE', 'busyness_NQ30'.
    """
    if ngtdm_matrices is None:
        if n_bins <= 256:
            data_int = (data - 1).astype(np.uint8)
        elif n_bins <= 65536:
            data_int = (data - 1).astype(np.uint16)
        else:
            data_int = (data - 1).astype(np.int32)  # pragma: no cover

        # Determine n_threads for JIT call
        try:
            n_threads = int(numba.config.NUMBA_NUM_THREADS)
        except (ValueError, TypeError):
            n_threads = 1  # Fallback

        _, _, s, n, _ = _calculate_local_features_numba(
            data_int,
            mask,
            n_bins,
            calc_glcm=False,
            calc_glrlm=False,
            calc_ngtdm=True,
            calc_ngldm=False,
            offsets_26=OFFSETS_26,
            directions_13=DIRECTIONS_13,
            ngldm_alpha=0,
            n_threads=n_threads,
        )
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
        Pi, Pj = np.meshgrid(p_nz, p_nz, indexing="ij")
        Ii, Ij = np.meshgrid(I_nz, I_nz, indexing="ij")

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
    Pi, Pj = np.meshgrid(p_nz, p_nz, indexing="ij")
    Si, Sj = np.meshgrid(s_nz, s_nz, indexing="ij")
    Ii, Ij = np.meshgrid(I_nz, I_nz, indexing="ij")

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
        mask (npt.NDArray[np.floating[Any]]): The 3D binary mask array defining the ROI.
        n_bins (int): The number of grey levels.
        ngldm_matrix (Optional[npt.NDArray[np.floating[Any]]]): Pre-calculated NGLDM matrix.
        ngldm_alpha (int): The coarseness parameter α. Two grey levels are considered dependent
            if their absolute difference is ≤ α. Default is 0 (exact match, IBSI standard).

    Returns:
        dict[str, float]: A dictionary of calculated NGLDM features.
            Example keys: 'low_dependence_emphasis_SODN', 'dependence_count_entropy_FCBV'.
    """
    if ngldm_matrix is None:
        if n_bins <= 256:
            data_int = (data - 1).astype(np.uint8)
        elif n_bins <= 65536:
            data_int = (data - 1).astype(np.uint16)
        else:
            data_int = (data - 1).astype(np.int32)  # pragma: no cover

        # Determine n_threads for JIT call
        try:
            n_threads = int(numba.config.NUMBA_NUM_THREADS)
        except (ValueError, TypeError):
            n_threads = 1  # Fallback

        _, _, _, _, ngldm = _calculate_local_features_numba(
            data_int,
            mask,
            n_bins,
            calc_glcm=False,
            calc_glrlm=False,
            calc_ngtdm=False,
            calc_ngldm=True,
            offsets_26=OFFSETS_26,
            directions_13=DIRECTIONS_13,
            ngldm_alpha=ngldm_alpha,
            n_threads=n_threads,
        )
    else:
        ngldm = ngldm_matrix

    N_s = np.sum(ngldm)
    if N_s == 0:
        return {}

    P = ngldm / N_s

    n_g, n_d = ngldm.shape
    i_idx, j_idx = np.indices((n_g, n_d))
    I = i_idx + 1  # noqa: E741
    J = j_idx + 1  # Dependence count

    features = {}

    # Low Dependence Emphasis (LDE) - SODN
    features["low_dependence_emphasis_SODN"] = np.sum(P / (J**2))

    # High Dependence Emphasis (HDE) - IMOQ
    features["high_dependence_emphasis_IMOQ"] = np.sum(P * (J**2))

    # Low Grey Level Count Emphasis (LGCE) - TL9H
    features["low_grey_level_count_emphasis_TL9H"] = np.sum(P / (I**2))

    # High Grey Level Count Emphasis (HGCE) - OAE7
    features["high_grey_level_count_emphasis_OAE7"] = np.sum(P * (I**2))

    # Low Dependence Low Grey Level Emphasis (LDLGE) - EQ3F
    features["low_dependence_low_grey_level_emphasis_EQ3F"] = np.sum(
        P / ((I**2) * (J**2))
    )

    # Low Dependence High Grey Level Emphasis (LDHGE) - JA6D
    features["low_dependence_high_grey_level_emphasis_JA6D"] = np.sum(
        P * (I**2) / (J**2)
    )

    # High Dependence Low Grey Level Emphasis (HDLGE) - NBZI
    features["high_dependence_low_grey_level_emphasis_NBZI"] = np.sum(
        P * (J**2) / (I**2)
    )

    # High Dependence High Grey Level Emphasis (HDHGE) - 9QMG
    features["high_dependence_high_grey_level_emphasis_9QMG"] = np.sum(
        P * (I**2) * (J**2)
    )

    # Grey Level Non-Uniformity - FP8K
    s_i = np.sum(ngldm, axis=1)
    features["grey_level_non_uniformity_FP8K"] = np.sum(s_i**2) / N_s

    # Normalised Grey Level Non-Uniformity - 5SPA
    features["normalised_grey_level_non_uniformity_5SPA"] = np.sum(s_i**2) / (N_s**2)

    # Dependence Count Non-Uniformity - Z87G
    s_j = np.sum(ngldm, axis=0)
    features["dependence_count_non_uniformity_Z87G"] = np.sum(s_j**2) / N_s

    # Normalised Dependence Count Non-Uniformity - OKJI
    features["normalised_dependence_count_non_uniformity_OKJI"] = np.sum(s_j**2) / (
        N_s**2
    )

    # Dependence Count Percentage - 6XV8
    n_voxels = np.sum(mask)
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
        mask_array: Mask array (ROI).
        n_bins: Number of bins.
        distance_mask_array: Optional mask for GLDZM distance calculation.
                             If None, mask_array is used.
        ngldm_alpha: The coarseness parameter α for NGLDM. Two grey levels are considered
            dependent if their absolute difference is ≤ α. Default is 0 (IBSI standard).

    Returns:
        Dictionary of all texture features.
    """
    results = {}

    # Calculate all matrices once
    texture_matrices = calculate_all_texture_matrices(
        disc_array,
        mask_array,
        n_bins,
        distance_mask=distance_mask_array,
        ngldm_alpha=ngldm_alpha,
    )

    # GLCM
    results.update(
        calculate_glcm_features(
            disc_array, mask_array, n_bins, glcm_matrix=texture_matrices["glcm"]
        )
    )

    # GLRLM
    results.update(
        calculate_glrlm_features(
            disc_array, mask_array, n_bins, glrlm_matrix=texture_matrices["glrlm"]
        )
    )

    # GLSZM
    results.update(
        calculate_glszm_features(
            disc_array, mask_array, n_bins, glszm_matrix=texture_matrices["glszm"]
        )
    )

    # GLDZM
    results.update(
        calculate_gldzm_features(
            disc_array,
            mask_array,
            n_bins,
            gldzm_matrix=texture_matrices["gldzm"],
            distance_mask=(
                distance_mask_array if distance_mask_array is not None else mask_array
            ),
        )
    )

    # NGTDM
    results.update(
        calculate_ngtdm_features(
            disc_array,
            mask_array,
            n_bins,
            ngtdm_matrices=(texture_matrices["ngtdm_s"], texture_matrices["ngtdm_n"]),
        )
    )
    # NGLDM
    results.update(
        calculate_ngldm_features(
            disc_array, mask_array, n_bins, ngldm_matrix=texture_matrices["ngldm"]
        )
    )

    return results
