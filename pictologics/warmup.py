"""
JIT Warmup Module
=================

This module handles the eager compilation (warmup) of Numba-accelerated functions
upon package import. This ensures that the first call to these functions by the user
is fast, at the cost of slightly increased import time.

Behavior can be controlled via the environment variable:
    PICTOLOGICS_DISABLE_WARMUP=1  : Disables automatic warmup.
"""

from __future__ import annotations

import logging
import os
import warnings

import numba
import numpy as np

# Private imports to access Numba kernels directly
from .features import intensity, morphology, texture

logger = logging.getLogger(__name__)


def warmup_jit() -> None:
    """
    Trigger compilation of Numba-accelerated functions by running them
    with minimal dummy data.
    """
    if os.environ.get("PICTOLOGICS_DISABLE_WARMUP", "0") == "1":
        logger.info("Pictologics warmup disabled via environment variable.")
        return

    logger.info("Warming up Pictologics JIT functions... (this may take a moment)")

    # Suppress warnings during warmup (e.g. division by zero in dummy data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _warmup_texture()
            _warmup_intensity()
            _warmup_morphology()
            logger.info("Pictologics JIT warmup complete.")
        except Exception as e:
            # We don't want warmup failure to crash the program import
            logger.warning(f"Pictologics JIT warmup failed: {e}")


def _warmup_texture() -> None:
    """Warmup texture calculation functions."""
    # Shared dummy data
    shape = (4, 4, 4)
    n_bins = 5
    mask = np.ones(shape, dtype=np.uint8)
    # Use a predictable random state or just zeros/ones to avoid runtime variation
    base = np.zeros(shape, dtype=np.uint8)
    base[::2] = 1  # Add some variation

    # Get thread count safely
    try:
        n_threads = int(numba.get_num_threads())
    except Exception:
        n_threads = int(getattr(numba.config, "NUMBA_NUM_THREADS", 1))

    # _calculate_local_features_numba is specialized by dtype; compile common variants.
    # We loop over dtypes but reuse the same logic
    for dtype in (np.uint8, np.uint16, np.int32):
        data_int = base.astype(dtype, copy=False)
        texture._calculate_local_features_numba(
            data_int,
            mask,
            n_bins,
            calc_glcm=True,
            calc_glrlm=True,
            calc_ngtdm=True,
            calc_ngldm=True,
            offsets_26=texture.OFFSETS_26,
            directions_13=texture.DIRECTIONS_13,
            ngldm_alpha=0,
            n_threads=n_threads,
        )

    # Zone features warmup
    # Data must be 1-based for some logic if not careful, but the kernel usually takes 1..n_bins
    data_1based = (base.astype(np.int64) + 1).astype(np.int64, copy=False)
    dist_map = np.ones(shape, dtype=np.int32)
    max_zones = int(np.prod(shape))

    # Use the pool to get buffers
    pool = texture._ZoneBufferPool.get_instance()
    res_gl, res_size, res_dist, stack = pool.get_buffers(max_zones)

    texture._calculate_zone_features_numba(
        data_1based,
        mask.copy(),  # Copy because mask is modified in place
        dist_map,
        n_bins,
        res_gl,
        res_size,
        res_dist,
        stack,
        calc_glszm=True,
        calc_gldzm=True,
    )


def _warmup_intensity() -> None:
    """Warmup intensity feature functions."""
    # 1. First Order Statistics Helpers
    values = np.array([0.0, 1.0, 2.0, 10.0, 10.0], dtype=np.float64)
    mean_val = 4.6

    intensity._sum_sq_centered(values, mean_val)
    intensity._central_moments_2_3_4(values, mean_val)
    intensity._mean_abs_dev(values, mean_val)
    intensity._robust_mean_abs_dev(values, lower=0.0, upper=10.0)

    # 2. Spatial Features
    # Minimal 3-voxel structure
    x_idx = np.array([0, 1, 0], dtype=np.int64)
    y_idx = np.array([0, 0, 1], dtype=np.int64)
    z_idx = np.array([0, 0, 0], dtype=np.int64)
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    intensity._calculate_spatial_features_numba(
        x_idx,
        y_idx,
        z_idx,
        intensities,
        mean_int=2.0,
        sx=1.0,
        sy=1.0,
        sz=1.0,
    )

    # 3. Local Mean / Peaks
    # 5x5x5 volume
    data = np.zeros((5, 5, 5), dtype=np.float64)
    data[2, 2, 2] = 10.0
    # Two voxels in mask
    mask_indices = np.ascontiguousarray(
        np.array([[2, 2, 2], [2, 2, 3]], dtype=np.int32)
    )
    # Two offsets
    offsets = np.ascontiguousarray(np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32))

    roi_means = intensity._calculate_local_mean_numba(data, mask_indices, offsets)
    intensity._calculate_local_peaks_numba(data, mask_indices, roi_means)

    # Max mean at max intensity helper
    roi_data = np.array([9.0, 10.0], dtype=np.float64)
    roi_means_small = np.array([1.0, 2.0], dtype=np.float64)
    intensity._max_mean_at_max_intensity(roi_data, roi_means_small, max_val=10.0)


def _warmup_morphology() -> None:
    """Warmup morphology functions."""
    # 1. Mask Moments
    # 4x4x4 mask with a small block
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    # Intensity image for weighted moments
    img = np.zeros(mask.shape, dtype=np.float64)
    img[mask > 0] = 2.0

    morphology._accumulate_moments_from_mask_numba(mask)
    morphology._accumulate_intensity_weighted_moments_numba(mask, img)

    # 2. Point Cloud / Mesh Operations
    # Simple pyramid (5 verts)
    verts = np.ascontiguousarray(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
    )

    # OMBB
    center = np.ascontiguousarray(np.array([0.5, 0.5, 0.5], dtype=np.float64))
    evecs = np.ascontiguousarray(np.eye(3, dtype=np.float64))
    morphology._ombb_extents_numba(verts, center, evecs)
    morphology._max_pairwise_distance_numba(verts)

    tet_verts = verts[:4]  # First 4 verts form a tet
    tet_faces = np.ascontiguousarray(
        np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            dtype=np.int64,
        )
    )
    morphology._mesh_area_volume_numba(tet_verts, tet_faces)
    mvee_points = np.ascontiguousarray(
        np.concatenate([verts, [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]], axis=0)
    )
    morphology._mvee_khachiyan_numba(mvee_points, tol=0.1)
