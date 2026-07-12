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

import os
import warnings
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

# Private imports to access Numba kernels directly
from .features import _utils, intensity, morphology, texture


def warmup_jit() -> None:
    """
    Trigger compilation of Numba-accelerated functions by running them
    with minimal dummy data.
    """
    if os.environ.get("PICTOLOGICS_DISABLE_WARMUP", "0") == "1":
        return

    errors: list[str] = []

    # Suppress warnings during warmup (e.g. division by zero in dummy data).
    # Each group is warmed independently so one failure doesn't skip the rest.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, step in (
            ("_warmup_texture", _warmup_texture),
            ("_warmup_intensity", _warmup_intensity),
            ("_warmup_morphology", _warmup_morphology),
            ("_warmup_filters", _warmup_filters),
        ):
            try:
                step()
            except Exception as e:
                errors.append(f"{name}: {e}")

    # Warn about warmup failures outside the suppression context
    if errors:
        warnings.warn(
            f"Pictologics JIT warmup failed: {'; '.join(errors)}",
            RuntimeWarning,
            stacklevel=2,
        )


def _warmup_texture() -> None:
    """Warmup texture calculation functions."""
    # Shared dummy data
    shape = (4, 4, 4)
    n_bins = 5
    mask = np.ones(shape, dtype=np.uint8)

    # Bounding-box scan is specialized by mask dtype AND memory layout; cropped masks
    # (mask[bbox]) are non-contiguous views, so compile both the C-contiguous and the
    # strided ('A') signature for each common dtype.
    for bbox_dtype in (np.float64, np.uint8, np.bool_):
        bbox_mask = mask.astype(bbox_dtype)
        _utils._bbox_scan_numba(bbox_mask)
        _utils._bbox_scan_numba(bbox_mask[1:, 1:, 1:])

    # ROI min/max scan (GLCM Ng_eff): the pipeline passes the bbox-cropped discretised
    # image (int32 strided view) with a uint8 strided mask; float64 data covers direct
    # API use. Compile both layouts for each combination.
    data_f64 = np.ones(shape, dtype=np.float64)
    for mm_dtype in (np.float64, np.uint8):
        mm_mask = mask.astype(mm_dtype)
        _utils._roi_min_max_numba(data_f64, mm_mask)
        _utils._roi_min_max_numba(data_f64[1:, 1:, 1:], mm_mask[1:, 1:, 1:])
        _utils._roi_min_max_serial_numba(data_f64, mm_mask)
        _utils._roi_min_max_serial_numba(data_f64[1:, 1:, 1:], mm_mask[1:, 1:, 1:])
    data_i32 = np.ones(shape, dtype=np.int32)
    _utils._roi_min_max_numba(data_i32, mask)
    _utils._roi_min_max_numba(data_i32[1:, 1:, 1:], mask[1:, 1:, 1:])
    _utils._roi_min_max_serial_numba(data_i32, mask)
    _utils._roi_min_max_serial_numba(data_i32[1:, 1:, 1:], mask[1:, 1:, 1:])
    # Use a predictable random state or just zeros/ones to avoid runtime variation
    base = np.zeros(shape, dtype=np.uint8)
    base[::2] = 1  # Add some variation

    # Match the production callers in texture.py
    try:
        n_threads = int(numba.config.NUMBA_NUM_THREADS)
    except (ValueError, TypeError):
        n_threads = 1  # Fallback

    # _calculate_local_features_numba is specialized by dtype; the dispatch code casts
    # discretised data to uint8 (n_bins <= 256) or int32, never other integer types.
    for dtype in (np.uint8, np.int32):
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

    # GLDZM distance-transform kernel: mask_bool is always a fresh, C-contiguous bool
    # array (the `> 0` comparison that produces it always allocates a C-contiguous
    # output, regardless of the input mask's dtype or memory layout).
    texture._chamfer_distance_taxicab_numba(mask.astype(np.bool_))

    # Zone features warmup (GLSZM/GLDZM). Numba's lazy dispatch specializes on the
    # exact dtype AND layout of every array argument, so mirror the production calls
    # exactly: the discretised image (int32) is a strided bbox-cropped view in the
    # common case but C-contiguous for full-volume ROIs — compile both. The mask is
    # always a C-contiguous uint8 copy and the distance map is always an int32
    # strided view (real GLDZM maps and the GLSZM-only dummy alike). Grey levels are
    # 1-based in [1, n_bins]. The kernels pad-and-copy their inputs; nothing is
    # modified in place.
    zone_pad: npt.NDArray[Any] = np.zeros((5, 5, 5), dtype=np.int32)
    zone_pad[1:, 1:, 1:] = base + 1
    data_strided = zone_pad[1:, 1:, 1:]
    data_contig: npt.NDArray[Any] = np.ascontiguousarray(data_strided)
    dist_map: npt.NDArray[Any] = np.ones((5, 5, 5), dtype=np.int32)[1:, 1:, 1:]
    max_zones = int(np.prod(shape))

    # Use the pool to get buffers
    pool = texture._ZoneBufferPool.get_instance()
    res_gl, res_size, res_dist, stack = pool.get_buffers(max_zones)

    for zone_data in (data_strided, data_contig):
        texture._calculate_zone_features_serial_numba(
            zone_data,
            mask,
            dist_map,
            n_bins,
            res_gl,
            res_size,
            res_dist,
            stack,
            calc_glszm=True,
            calc_gldzm=True,
        )
        texture._calculate_zone_features_numba(
            zone_data,
            mask,
            dist_map,
            n_bins,
            res_gl,
            res_size,
            res_dist,
            stack,
            n_chunks=2,  # >1 so the cross-chunk merge path is compiled too
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

    # Discretised images are int32 (see discretise_image) and apply_mask preserves
    # dtype, so the histogram feature path calls these helpers with int32 arrays;
    # compile that specialization too.
    values_i32: npt.NDArray[Any] = np.array([0, 1, 2, 10, 10], dtype=np.int32)
    intensity._central_moments_2_3_4(values_i32, mean_val)
    intensity._mean_abs_dev(values_i32, mean_val)
    intensity._robust_mean_abs_dev(values_i32, lower=0.0, upper=10.0)

    # 2. Spatial Features
    # Minimal 3-voxel structure. int32 to match the production caller in
    # calculate_spatial_intensity_features, so the same specialization is compiled.
    x_idx = np.array([0, 1, 0], dtype=np.int32)
    y_idx = np.array([0, 0, 1], dtype=np.int32)
    z_idx = np.array([0, 0, 0], dtype=np.int32)
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
    mask_indices = np.ascontiguousarray(np.array([[2, 2, 2], [2, 2, 3]], dtype=np.int32))
    # Two offsets
    offsets = np.ascontiguousarray(np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32))

    roi_means = intensity._calculate_local_mean_numba(data, mask_indices, offsets)
    intensity._calculate_local_peaks_numba(data, mask_indices, roi_means)


def _warmup_morphology() -> None:
    """Warmup morphology functions."""
    # 1. Mask Moments
    # 4x4x4 mask with a small block
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    # Intensity image for weighted moments
    img = np.zeros(mask.shape, dtype=np.float64)
    img[mask > 0] = 2.0

    # Production scans bbox-cropped views (strided) in the common case and
    # C-contiguous arrays for full-volume ROIs; compile both layouts.
    morphology._accumulate_moments_from_mask_numba(mask)
    morphology._accumulate_moments_from_mask_numba(mask[1:, 1:, 1:])
    morphology._accumulate_intensity_weighted_moments_numba(mask, img)
    morphology._accumulate_intensity_weighted_moments_numba(mask[1:, 1:, 1:], img[1:, 1:, 1:])

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


def _warmup_filters() -> None:
    """Warmup filter and preprocessing operations."""
    # Import here to avoid circular dependencies
    from scipy.ndimage import affine_transform
    from scipy.signal import fftconvolve

    from . import preprocessing

    # 1. Preprocessing kernels (discretise / resegment / resample). The
    # dispatch code always feeds C-contiguous arrays (via ravel /
    # ascontiguousarray), so one layout per dtype combination suffices.
    flat = np.linspace(0.0, 10.0, 27, dtype=np.float64)
    binned = np.empty(flat.size, dtype=np.int32)
    preprocessing._discretise_fbn_numba(flat, 4.0, 0.0, 10.0, binned)
    preprocessing._discretise_fbs_numba(flat, 2.5, 0.0, binned)

    for m_dtype in (np.float64, np.uint8, np.bool_):
        m_flat = np.ones(flat.size, dtype=m_dtype)
        m_out = np.empty(flat.size, dtype=m_dtype)
        preprocessing._resegment_numba(flat, m_flat, 0.0, 5.0, m_out)

    src = np.ones((4, 4, 4), dtype=np.float64)
    scale = np.array([1.1, 1.1, 1.1])
    shift = np.zeros(3)
    out3 = np.empty((3, 3, 3), dtype=np.float64)
    preprocessing._resample_trilinear_numba(src, scale, shift, out3)
    for s_dtype in (np.float64, np.uint8, np.bool_):
        src_d = src.astype(s_dtype)
        out_d = np.empty((3, 3, 3), dtype=s_dtype)
        preprocessing._resample_nearest_numba(src_d, scale, shift, out_d)
    valid = np.ones((4, 4, 4), dtype=np.bool_)
    out_valid = np.empty((3, 3, 3), dtype=np.bool_)
    preprocessing._resample_trilinear_masked_numba(src, valid, scale, shift, 0.5, out3, out_valid)

    # 2. Warmup affine_transform (scipy fallback for cubic / exotic boundary modes)
    # Small 3D array
    dummy_img = np.ones((5, 5, 5), dtype=np.float32)
    matrix = np.array([1.1, 1.1, 1.1])  # Slight scaling
    offset = np.array([0.0, 0.0, 0.0])
    _ = affine_transform(dummy_img, matrix=matrix, offset=offset, output_shape=(6, 6, 6), order=1)

    # 3. Warmup FFT convolution (used in Gabor, Laws, etc.)
    dummy_2d = np.ones((8, 8), dtype=np.float32)
    kernel_2d = np.ones((3, 3), dtype=np.complex64)
    _ = fftconvolve(dummy_2d, kernel_2d, mode="same")

    # 4. Warmup 3D convolution
    dummy_3d = np.ones((8, 8, 8), dtype=np.float32)
    kernel_3d = np.ones((3, 3, 3), dtype=np.float32)
    _ = fftconvolve(dummy_3d, kernel_3d, mode="same")
