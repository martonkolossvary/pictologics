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

import math
import os
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

# Private imports to access Numba kernels directly
from .features import _utils, intensity, morphology, texture
from .features._mc_tables import EDGE_TABLE, TRIANGLE_COUNT


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
    mask: npt.NDArray[Any] = np.ones(shape, dtype=np.uint8)

    # The box scan, for each mask type that it reads: an integer or bool mask goes as the
    # integer type of its size (see _utils._nonzero_form). A column-order mask goes as its
    # row-order transpose and a strided one as a row-order copy, so row order only.
    mask_kinds = (np.float64, np.float32, np.uint8, np.uint16, np.int32, np.int64)
    for kind in mask_kinds:
        _utils._bbox_scan_numba(mask.astype(kind))
        _utils._bbox_scan_numba_serial(mask.astype(kind))
    # The parallel copy of an ROI region (float64, row order or strided)
    cube = np.ones((2, 2, 4))
    for source in (cube, cube[:, :, :2]):
        _utils._copy_numba(source, np.empty(source.shape))
    # The label boxes of run_rois: an integer label map as the integer type of its size
    edges = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    for kind in (np.uint8, np.uint16, np.int32, np.int64):
        tables = np.empty((1, 2, 6), dtype=np.int64)
        _utils._label_extents_numba(mask.astype(kind), edges, tables, np.empty(1, np.int64))

    # ROI min/max scan. The FBS bin count reads the discretised image (int32) with the mask
    # of the run (each type of the box scan). Discretise reads float64 data with a uint8
    # or float64 mask. The GLCM Ng_eff reads the box crop of the grey levels (int32 in the
    # pipeline; int64 or float64 in a direct call, see texture._level_form) with the
    # row-order texture ROI as uint8. Other layouts go as transposes or row-order copies
    # (see _utils._row_order_pair), so row order and the strided crop only.
    data_f64 = np.ones(shape, dtype=np.float64)
    data_i32 = np.ones(shape, dtype=np.int32)
    roi_u8 = np.ascontiguousarray(mask[1:, 1:, 1:])
    pairs: list[tuple[npt.NDArray[Any], npt.NDArray[Any]]] = [
        (data_i32, mask.astype(kind)) for kind in mask_kinds
    ]
    pairs += [(data_f64, mask), (data_f64, mask.astype(np.float64))]
    pairs += [(data_i32[1:, 1:, 1:], roi_u8)]
    for kind in (np.int64, np.float64):
        grey: npt.NDArray[Any] = np.ones(shape, dtype=kind)[1:, 1:, 1:]
        pairs += [(grey, roi_u8), (np.ascontiguousarray(grey), roi_u8)]
    for mm_data, mm_mask in pairs:
        _utils._roi_min_max_numba(mm_data, mm_mask)
        _utils._roi_min_max_serial_numba(mm_data, mm_mask)

    # The texture kernels. The grey-level volume build is specialized by the type and
    # layout of the grey levels: int32 in the pipeline (int64 and float64 too in a direct
    # call, see texture._level_form), a strided box crop or C-contiguous for an ROI that
    # fills the image; the ROI is a fresh bool array. The local and zone kernels read one
    # uint16 volume, so they compile once, with merged (compact) and per-direction tables
    # alike. Levels are 1-based in [1, n_bins].
    levels: npt.NDArray[Any] = np.zeros((5, 5, 5), dtype=np.int32)
    levels[1:, 1:, 1:] = 1
    levels[1::2, 1:, 1:] = 2  # some variation
    box = levels[1:, 1:, 1:]
    for data in (box, np.ascontiguousarray(box)):
        texture._texture_matrices(data, mask, n_bins, compact=True)
    # Small volumes take the serial volume kernel; large ones the parallel one
    for kind in (np.int32, np.int64, np.float64):
        crop = levels.astype(kind)[1:, 1:, 1:]
        for data in (crop, np.ascontiguousarray(crop)):
            vol = np.zeros(tuple(s + 2 for s in data.shape), dtype=np.uint16)
            counts = np.empty((data.shape[0], 2), dtype=np.int64)
            texture._texture_volume_numba(data, mask != 0, n_bins, vol, counts)
            texture._texture_volume_serial_numba(data, mask != 0, n_bins, vol, counts)
    texture.calculate_all_texture_matrices(box, mask, n_bins)
    # The GLCM feature sums: float64 probabilities and int64 grey levels
    glcm_p = np.full((2, 2), 0.25)
    glcm_levels = np.array([1, 2], dtype=np.int64)
    texture._glcm_sums_numba(glcm_p, glcm_levels, 2.0, np.zeros(3), np.zeros(5))
    texture._glcm_mu_sums_numba(glcm_p, glcm_levels, 1.5)
    # The NGTDM complexity: int64 levels, float64 probabilities and differences
    texture._ngtdm_complexity_numba(glcm_levels, np.full(2, 0.5), np.ones(2))
    # The parallel zeroing and sum of large thread tables (from 182 grey levels on)
    texture._zero_fill_numba(np.ones(3, dtype=np.uint32))
    texture._thread_sum_numba(np.zeros((2, 4), dtype=np.uint32), np.zeros(4, dtype=np.uint64))
    # The levels that occur, for the compact tables of many grey levels
    texture._levels_seen_numba(np.zeros((3, 3, 3), dtype=np.uint16), np.zeros((3, 2), np.bool_))
    # GLDZM distance-transform kernel: its input is always a fresh bool array from a
    # comparison, so a C-contiguous array, with the three planar flags.
    dist = texture._chamfer_distance_taxicab_numba(mask.astype(np.bool_), False, False, False)
    # The distance map of large masks, in threads
    texture._distance_map_parallel_numba(mask.astype(np.bool_), False, False, False)
    # The parallel zone kernels of large volumes, on this small one (with and without the
    # distance map)
    for distance in (dist, texture._NO_DISTANCE):
        vol, counts = texture._texture_volume(box, mask != 0, n_bins)
        gldzm = distance is dist
        texture._zone_matrices(vol, counts, distance, n_bins, True, gldzm, False, parallel=True)


def _warmup_intensity() -> None:
    """Warmup intensity feature functions."""
    # 1. First Order Statistics Helpers
    values = np.array([0.0, 1.0, 2.0, 10.0, 10.0], dtype=np.float64)
    mean_val = 4.6

    intensity._sum_sq_centered(values, mean_val)
    intensity._central_moments_2_3_4(values, mean_val)
    intensity._mean_abs_dev(values, mean_val)
    intensity._robust_mean_abs_dev(values, lower=0.0, upper=10.0)
    # The value counts of the IVH: int32 binned values (other integers go to int64)
    for dtype in (np.int32, np.int64):
        intensity._value_counts_numba(
            np.array([1, 2, 2], dtype=dtype), 0, np.zeros((1, 3), np.int64)
        )

    # The histogram features of integer values come from their bin counts, so these
    # helpers read float64 values only.

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

    # calculate_local_intensity_features passes a crop of the image: a strided view, or
    # the C-contiguous array itself when the crop covers all of it. Compile both layouts.
    # float32 data: the responses of the filters. The two-stage search reads the row
    # prefix sums (float64) of the same crops.
    rows = np.zeros(1, dtype=np.int64)
    data32 = data.astype(np.float32)
    for local_data in (data, data[1:, 1:, 1:], data32, data32[1:, 1:, 1:]):
        roi_means = intensity._calculate_local_mean_numba(local_data, mask_indices, offsets)
        intensity._calculate_local_peaks_numba(local_data, mask_indices, roi_means)
        prefix = np.empty(local_data.shape[:2] + (local_data.shape[2] + 1,), dtype=np.float64)
        intensity._row_prefix_sums_numba(local_data, prefix)
    intensity._approximate_local_means_numba(prefix, mask_indices, rows, rows, rows)
    # Order statistics of large ROIs: the radix select kernels
    bits = values.view(np.uint64)
    intensity._bucket_counts_numba(bits, 48, np.uint64(0), np.zeros((1, 1 << 16), dtype=np.int64))
    intensity._bucket_values_numba(
        bits,
        values,
        48,
        np.uint64(0),
        np.ones(1 << 16, dtype=np.bool_),
        np.array([bits.size]),
        np.empty(bits.size),
    )
    # The linear select: the sample check, the serial kernel, the count and the copy in
    # threads
    order = np.array([0, 1, 2, 2, 3, 4], dtype=np.int64)
    intensity._crowded_numba(values, 0.0, 409.6, order, np.zeros(4096, dtype=np.int64), 2, 1)
    intensity._linear_select_numba(
        values, 0.0, 409.6, order, np.zeros(4096, dtype=np.int64), np.empty(6, dtype=np.uint64)
    )
    intensity._linear_counts_numba(values, 0.0, 409.6, np.empty((1, 4096), dtype=np.int64))
    intensity._linear_keys_numba(
        values,
        0.0,
        409.6,
        np.full(4096, -1, dtype=np.int64),
        np.zeros((1, 1), dtype=np.int64),
        np.empty(1, dtype=np.uint64),
    )


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
    # C-contiguous arrays for full-volume ROIs; compile both layouts. Each parallel kernel
    # of a small ROI has a serial twin with the same signatures.
    for moments in (
        morphology._accumulate_moments_from_mask_numba,
        morphology._accumulate_moments_from_mask_numba_serial,
    ):
        moments(mask)
        moments(mask[1:, 1:, 1:])
    img32 = img.astype(np.float32)  # float32 images: the responses of the filters
    # A mask of another type becomes a row-order uint8 copy of the crop; the image crop
    # stays a strided view
    row_mask = np.ascontiguousarray(mask[1:, 1:, 1:])
    for weighted in (
        morphology._accumulate_intensity_weighted_moments_numba,
        morphology._accumulate_intensity_weighted_moments_numba_serial,
    ):
        weighted(mask, img)
        weighted(mask[1:, 1:, 1:], img[1:, 1:, 1:])
        weighted(mask, img32)
        weighted(mask[1:, 1:, 1:], img32[1:, 1:, 1:])
        weighted(row_mask, img[1:, 1:, 1:])
        weighted(row_mask, img32[1:, 1:, 1:])
        # An image of another type becomes a row-order float64 copy of the crop
        weighted(mask[1:, 1:, 1:], np.ascontiguousarray(img[1:, 1:, 1:]))

    # Marching cubes (the mask with its zero border), also the parallel form of large volumes
    for parallel in (False, True):
        morphology._mesh(np.pad(mask, 1), np.zeros(3), np.ones(3), parallel)
    padded = np.pad(mask, 1)
    for counts in (morphology._mc_counts_numba, morphology._mc_counts_numba_serial):
        counts(
            padded,
            EDGE_TABLE,
            TRIANGLE_COUNT,
            np.empty(padded.shape[0] - 1, dtype=np.int64),
            np.empty(padded.shape[0] - 1, dtype=np.int64),
        )
    morphology._column_stats_numba(np.ones((4, 3), dtype=np.float64))

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
    # np.linalg.eigh gives its eigenvectors in column order
    evecs = np.asfortranarray(np.eye(3, dtype=np.float64))
    morphology._ombb_extents_numba(verts, center, evecs)
    morphology._ombb_extents_numba_serial(verts, center, evecs)
    morphology._max_pairwise_distance_numba(verts)
    # The serial twin of the morphology worker thread (see pipeline._MorphologyAhead)
    morphology._max_pairwise_distance_serial_numba(verts)
    morphology._hull_candidates_numba(verts, np.ones(3, dtype=np.float64))
    found, _, triangles = morphology._exact_hull_numba(np.rint(2.0 * verts).astype(np.int64))
    morphology._hull_area_volume_numba(verts, triangles)

    tet_verts = verts[:4]  # First 4 verts form a tet
    tet_faces = np.ascontiguousarray(
        np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            dtype=np.int64,
        )
    )
    morphology._mesh_area_volume_numba(tet_verts, tet_faces)
    morphology._mesh_area_volume_numba_serial(tet_verts, tet_faces)
    mvee_points = np.ascontiguousarray(
        np.concatenate([verts, [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]], axis=0)
    )
    morphology._mvee_khachiyan_numba(mvee_points)  # production omits tol


def _warmup_filters() -> None:
    """Warmup filter, preprocessing and loader operations."""
    # Import here to avoid circular dependencies
    from . import loader, preprocessing
    from .filters import base as filter_base

    # 0. Loader: the column-order to row-order copy for NIfTI, DICOM and SEG data, and
    # the float64 copy with rescale of stored DICOM pixels and NIfTI data.
    for dtype in loader._ROW_ORDER_DTYPES:
        col = np.asfortranarray(np.ones((4, 4, 4), dtype=dtype))
        loader._to_row_order_numba(col, np.empty((4, 4, 4), dtype=dtype))
    # The column-order copy of save_image
    for kind in loader._COLUMN_ORDER_DTYPES:
        row = np.ones((4, 4, 4), dtype=kind)
        loader._to_row_order_numba(row, np.empty((4, 4, 4), dtype=kind, order="F"))
    ones, flags = np.ones(4), np.ones(4, dtype=np.bool_)
    # Every stored type of the fused NIfTI load (int8, int32 and uint32 too), and float32
    stored_types: list[Any] = list(
        dict.fromkeys((*loader._ROW_ORDER_DTYPES, *loader._NIFTI_FUSED_DTYPES))
    )
    for stored in stored_types:
        col = np.asfortranarray(np.ones((4, 4, 4), dtype=stored))
        loader._to_float_row_order_numba(col, ones, ones, flags, flags, np.empty((4, 4, 4)), 0)

    # 1. Preprocessing kernels (discretise / resegment / resample). The
    # dispatch code always feeds C-contiguous arrays (via ravel /
    # ascontiguousarray), so one layout per dtype combination suffices.
    flat = np.linspace(0.0, 10.0, 27, dtype=np.float64)
    binned = np.empty(flat.size, dtype=np.int32)
    preprocessing._discretise_fbn_numba(flat, 4.0, 0.0, 10.0, binned)
    preprocessing._discretise_fbs_numba(flat, 2.5, 0.0, binned)
    preprocessing._discretise_cutoffs_numba(flat, np.array([2.0, 5.0]), True, binned)
    preprocessing._nonfinite_blocks_numba(flat, np.zeros(1, dtype=np.uint8))

    for m_dtype in (np.float64, np.uint8, np.bool_):
        m_flat = np.ones(flat.size, dtype=m_dtype)
        m_out = np.empty(flat.size, dtype=m_dtype)
        preprocessing._resegment_numba(flat, m_flat, 0.0, 5.0, m_out)
        preprocessing._roi_values_numba(flat, m_flat)
        preprocessing._filter_outliers_numba(flat, m_flat, 0.0, 5.0, m_out)
        preprocessing._sentinel_counts_numba(flat, m_flat, np.array([0.0, -1000.0]))
    preprocessing._sentinel_counts_numba(  # no ROI mask
        flat, np.empty(0, dtype=np.uint8), np.array([0.0, -1000.0])
    )

    src = np.ones((4, 4, 4), dtype=np.float64)
    scale = np.array([1.1, 1.1, 1.1])
    shift = np.zeros(3)
    start = np.zeros(3, dtype=np.int64)  # the first voxel of the computed region
    out3 = np.empty((3, 3, 3), dtype=np.float64)
    out_u8 = np.empty((3, 3, 3), dtype=np.uint8)
    valid = np.ones((4, 4, 4), dtype=np.bool_)
    out_valid = np.empty((3, 3, 3), dtype=np.bool_)
    for linear, nearest, masked in (
        (
            preprocessing._resample_trilinear_numba,
            preprocessing._resample_nearest_numba,
            preprocessing._resample_trilinear_masked_numba,
        ),
        (
            preprocessing._resample_trilinear_numba_serial,
            preprocessing._resample_nearest_numba_serial,
            preprocessing._resample_trilinear_masked_numba_serial,
        ),
    ):
        linear(src, scale, shift, start, False, math.nan, out3)
        linear(src, scale, shift, start, False, 0.5, out_u8)
        linear(src.astype(np.uint8), scale, shift, start, True, math.nan, out_u8)  # uint8 input
        for s_dtype in (np.float64, np.uint8, np.bool_):
            out_d = np.empty((3, 3, 3), dtype=s_dtype)
            nearest(src.astype(s_dtype), scale, shift, start, out_d)
        masked(src, valid, scale, shift, start, 0.5, out3, out_valid)

    # 2. The LoG transfer of the FFT path of large float64 images
    filter_base._times_log_transfer(
        np.ones((2, 2, 2), dtype=np.complex128),
        np.ones((2, 2)),
        np.ones((2, 2)),
        np.ones(2),
        np.ones(2),
    )
