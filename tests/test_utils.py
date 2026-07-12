"""Tests for pictologics.features._utils (internal shared array utilities)."""

import numpy as np
import pytest

from pictologics.features._utils import (
    compute_nonzero_bbox,
    merge_bboxes,
    roi_min_max,
)


class TestComputeNonzeroBbox:
    """Tests for compute_nonzero_bbox (numpy path for small masks, numba for large)."""

    def test_small_mask_numpy_path(self):
        mask = np.zeros((5, 5, 5), dtype=int)
        mask[1:4, 1:4, 1:4] = 1
        assert compute_nonzero_bbox(mask) == (slice(1, 4), slice(1, 4), slice(1, 4))

    def test_small_mask_empty(self):
        assert compute_nonzero_bbox(np.zeros((5, 5, 5), dtype=int)) is None

    def test_non_3d_raises(self):
        with pytest.raises(ValueError, match="Expected a 3D mask"):
            compute_nonzero_bbox(np.zeros((5, 5)))

    def test_large_mask_numba_path(self):
        # >= 2^20 voxels routes through _bbox_scan_numba; a small corner ROI leaves
        # most rows empty, exercising the empty-row `continue`.
        mask = np.zeros((256, 64, 64), dtype=bool)
        mask[5:9, 3:7, 2:6] = True
        assert compute_nonzero_bbox(mask) == (slice(5, 9), slice(3, 7), slice(2, 6))

    def test_large_mask_numba_empty(self):
        # Empty mask on the numba path exercises the nz.size == 0 early return.
        assert compute_nonzero_bbox(np.zeros((256, 64, 64), dtype=bool)) is None


class TestRoiMinMax:
    """Tests for roi_min_max (serial kernel for small data, parallel for large)."""

    def test_serial_path(self):
        data = np.arange(27, dtype=float).reshape(3, 3, 3)
        mask = np.zeros((3, 3, 3), dtype=int)
        mask[0, 0, 0] = 1  # value 0
        mask[1, 1, 1] = 1  # value 13
        assert roi_min_max(data, mask) == (0.0, 13.0)

    def test_empty_mask_returns_none(self):
        assert roi_min_max(np.zeros((3, 3, 3)), np.zeros((3, 3, 3), dtype=int)) is None

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="two 3D arrays of equal shape"):
            roi_min_max(np.zeros((3, 3, 3)), np.zeros((3, 3, 4)))

    def test_large_data_parallel_path(self):
        # >= 2^19 voxels routes through the parallel kernel.
        data = np.random.rand(128, 64, 64)
        mask = np.zeros((128, 64, 64), dtype=np.uint8)
        mask[10:20, 5:15, 5:15] = 1
        lo, hi = roi_min_max(data, mask)
        roi = data[mask > 0]
        assert lo == pytest.approx(roi.min())
        assert hi == pytest.approx(roi.max())


class TestMergeBboxes:
    """Tests for merge_bboxes (per-axis union of two nonzero bounding boxes)."""

    def test_merge_two(self):
        a = (slice(1, 3), slice(2, 5), slice(0, 4))
        b = (slice(2, 6), slice(1, 3), slice(3, 7))
        assert merge_bboxes(a, b) == (slice(1, 6), slice(1, 5), slice(0, 7))

    def test_a_none_returns_b(self):
        b = (slice(1, 3), slice(1, 3), slice(1, 3))
        assert merge_bboxes(None, b) is b

    def test_b_none_returns_a(self):
        a = (slice(1, 3), slice(1, 3), slice(1, 3))
        assert merge_bboxes(a, None) is a
