"""Tests for pictologics.features._utils (internal shared array utilities)."""

from unittest.mock import patch

import numpy as np
import pytest

from pictologics.features import _utils
from pictologics.features._utils import (
    _nonzero_form,
    compute_nonzero_bbox,
    merge_bboxes,
    roi_min_max,
)


@pytest.mark.parametrize(
    ("kind", "form"),
    [
        (np.bool_, np.uint8),
        (np.int8, np.uint8),
        (np.uint8, np.uint8),
        (np.int16, np.uint16),
        (np.dtype(">i2"), np.uint16),  # big-endian, as some files give it
        (np.uint16, np.uint16),
        (np.int32, np.int32),
        (np.uint32, np.int32),
        (np.int64, np.int64),
        (np.uint64, np.int64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_the_scans_read_a_mask_as_the_integer_type_of_its_size(kind, form):
    # The same bits, so the same nonzero voxels, with no copy: the box and range scans
    # compile for six mask types (see warmup.py)
    mask = np.array([[[0, 1, 2, -1]]]).astype(kind)
    view = _nonzero_form(mask)
    assert view.dtype == form and np.shares_memory(view, mask)
    assert ((view != 0) == (mask != 0)).all()


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

    def test_large_masks_of_other_orders_give_the_same_box(self):
        # The scan reads a column-order mask as its row-order transpose and a strided mask
        # as a row-order copy, both as the integer type of its size: the box is the same.
        mask = np.zeros((256, 64, 64), dtype=np.int16)
        mask[5:9, 3:7, 2:6] = -3
        wide = np.zeros((256, 64, 128), dtype=np.int16)
        wide[:, :, ::2] = mask
        with patch.object(_utils, "_bbox_scan_numba", wraps=_utils._bbox_scan_numba) as scan:
            for form in (np.asfortranarray(mask), wide[:, :, ::2]):
                assert compute_nonzero_bbox(form) == (slice(5, 9), slice(3, 7), slice(2, 6))
            assert compute_nonzero_bbox(np.asfortranarray(np.zeros_like(mask))) is None
        shapes = [(64, 64, 256), (256, 64, 64), (64, 64, 256)]
        assert [call.args[0].shape for call in scan.call_args_list] == shapes
        for call in scan.call_args_list:
            assert call.args[0].dtype == np.uint16 and call.args[0].flags.c_contiguous


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

    def test_non_finite_values_are_skipped(self):
        # NaN and infinite values are no ROI intensities, in both kernels; an ROI with no
        # finite value gives None.
        for shape in ((3, 3, 3), (128, 64, 64)):
            data = np.full(shape, np.nan)
            mask = np.zeros(shape, dtype=np.uint8)
            mask[:2, :2, :2] = 1
            assert roi_min_max(data, mask) is None
            data[0, 0, :2] = (np.inf, -np.inf)
            data[1, 1, 1] = 4.0
            data[0, 1, 0] = -2.0
            assert roi_min_max(data, mask) == (-2.0, 4.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="two 3D arrays of equal shape"):
            roi_min_max(np.zeros((3, 3, 3)), np.zeros((3, 3, 4)))

    def test_masks_of_each_type_give_one_range(self):
        # The kernels read an integer or bool mask as the integer type of its size
        data = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
        mask = np.zeros((3, 3, 3))
        mask[0, 0, 0] = mask[1, 1, 1] = 1
        kinds = (np.bool_, np.int8, np.int16, np.uint32, np.uint64, np.float32)
        serial = _utils._roi_min_max_serial_numba
        with patch.object(_utils, "_roi_min_max_serial_numba", wraps=serial) as scan:
            for kind in kinds:
                assert roi_min_max(data, mask.astype(kind)) == (0.0, 13.0)
        forms = [np.uint8, np.uint8, np.uint16, np.int32, np.int64, np.float32]
        assert [call.args[1].dtype for call in scan.call_args_list] == forms

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


def test_a_float16_mask_scans_as_float32() -> None:
    # The scans read a float16 mask as a float32 copy: the same nonzero voxels, and a
    # kernel type that the import warms
    from pictologics.features._utils import _nonzero_form, roi_min_max

    mask = np.zeros((4, 5, 6), dtype=np.float16)
    mask[1, 2, 3] = 0.5
    mask[2, 3, 4] = -1.0
    form = _nonzero_form(mask)
    assert form.dtype == np.float32 and np.array_equal(form != 0, mask != 0)
    data = np.arange(120, dtype=np.float64).reshape(4, 5, 6)
    assert roi_min_max(data, mask) == roi_min_max(data, mask.astype(np.float64))


def test_the_range_scan_reads_the_layouts_of_the_import() -> None:
    # Two column-order arrays go as their transposes, other layouts as row-order copies;
    # the strided grey-level crop with a row-order uint8 ROI stays as it is
    from pictologics.features import _utils

    seen: list[tuple[bool, bool, str]] = []
    real = _utils._roi_min_max_serial_numba

    def spy(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seen.append((bool(data.flags.c_contiguous), bool(mask.flags.c_contiguous), mask.dtype.name))
        return real(data, mask)

    rng = np.random.default_rng(1)
    data = rng.normal(size=(6, 7, 8)).astype(np.int32)
    mask = (rng.random((6, 7, 8)) > 0.4).astype(np.float64)
    expected = _utils.roi_min_max(data, mask)
    cases = [
        (np.asfortranarray(data), np.asfortranarray(mask)),  # both column order
        (np.asfortranarray(data), mask),  # column-order data
        (data, np.asfortranarray(mask)),  # column-order mask
        (data, mask[:, ::2].repeat(2, axis=1)[:, :7]),  # a strided mask
        (np.pad(data, 1)[1:-1, 1:-1, 1:-1], mask),  # strided data with a float64 mask
        (np.pad(data, 1)[1:-1, 1:-1, 1:-1], mask.astype(np.uint8)),  # the texture crop
    ]
    with patch.object(_utils, "_roi_min_max_serial_numba", spy):
        for d, m in cases:
            assert _utils.roi_min_max(d, m) == expected
    assert [s[1] for s in seen] == [True] * 6  # the mask is in row order
    assert [s[0] for s in seen] == [True, True, True, True, True, False]  # the crop stays
    assert seen[-1][2] == "uint8"


def test_a_serial_twin_is_the_kernel_code_with_parallel_off() -> None:
    # The twin is a copy of the Python code of the kernel under its own name, compiled with
    # the options of the kernel but parallel=False; with the compiler off, it is the kernel.
    def kernel(values: np.ndarray, factor: int = 2) -> float:
        return float(values.sum() * factor)

    dispatcher = type(
        "Dispatcher",
        (),
        {
            "py_func": kernel,
            "targetoptions": {"nopython": True, "parallel": True, "fastmath": True},
        },
    )()
    with patch.object(_utils, "jit", lambda **options: lambda code: (options, code)):
        options, twin = _utils.serial_twin(dispatcher)
    assert options == {"nopython": True, "parallel": False, "fastmath": True, "cache": True}
    assert twin.__name__ == "kernel_serial" and twin.__qualname__.endswith(".kernel_serial")
    assert twin.__module__ == kernel.__module__ and twin is not kernel
    assert twin(np.arange(4.0)) == kernel(np.arange(4.0)) == 12.0
    assert _utils.serial_twin(kernel) is kernel  # the compiler is off


def test_sized_picks_the_twin_for_small_work_or_one_thread() -> None:
    assert _utils.sized("kernel", "twin", 99, 100) == "twin"
    assert _utils.sized("kernel", "twin", 100, 100) == "kernel"
    with patch.object(_utils, "get_num_threads", return_value=1):
        assert _utils.sized("kernel", "twin", 10**9, 100) == "twin"
