from __future__ import annotations

# ruff: noqa: E402
import os
import warnings

# Suppress "NumPy module was reloaded" warning which can happen in test setups
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pictologics.loader import Image
from pictologics.preprocessing import (
    COMMON_SENTINEL_VALUES,
    apply_mask,
    create_source_mask_from_sentinel,
    detect_sentinel_value,
    discretise_image,
    extract_roi,
    filter_outliers,
    grow_mask,
    keep_largest_component,
    normalise_image,
    resample_image,
    resegment_mask,
    round_intensities,
)


@pytest.fixture
def mock_image() -> Image:
    """A simple 5x5x5 numeric gradient image."""
    shape = (5, 5, 5)
    array = np.zeros(shape, dtype=float)
    for z in range(5):
        for y in range(5):
            for x in range(5):
                array[z, y, x] = x + y + z
    return Image(
        array=array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="CT",
    )


@pytest.fixture
def mock_mask() -> Image:
    """A 3x3x3 ROI centered in the 5x5x5 volume."""
    shape = (5, 5, 5)
    array = np.zeros(shape, dtype=np.uint8)
    array[1:4, 1:4, 1:4] = 1
    return Image(
        array=array,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        modality="mask",
    )


def test_resample_image_linear(mock_image: Image) -> None:
    # Resample to 2x spacing (downsample)
    new_spacing = (2.0, 2.0, 2.0)
    resampled = resample_image(mock_image, new_spacing, interpolation="linear")

    # Expected shape: ceil(5 * 1.0 / 2.0) = 3
    expected_shape = (3, 3, 3)
    assert resampled.array.shape == expected_shape
    assert resampled.spacing == new_spacing

    # Check origin shift
    # Shift = 0 for grid aligned centers?
    # extent_orig = (4,4,4), extent_new=(4,4,4) -> shift=0
    assert np.allclose(resampled.origin, mock_image.origin)


def test_resample_image_origin_shift_uses_direction() -> None:
    direction = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    image = Image(
        array=np.arange(9, dtype=float).reshape((3, 3, 1)),
        spacing=(2.0, 2.0, 1.0),
        origin=(10.0, 20.0, 30.0),
        direction=direction,
        modality="CT",
    )

    resampled = resample_image(image, (1.0, 1.0, 1.0), interpolation="nearest")

    origin_shift = np.array([-0.5, -0.5, 0.0])
    expected_origin = np.array(image.origin) + direction @ origin_shift
    assert resampled.array.shape == (6, 6, 1)
    assert np.allclose(resampled.origin, expected_origin)
    assert not np.allclose(resampled.origin, np.array(image.origin) + origin_shift)


def test_resample_image_nearest(mock_image: Image) -> None:
    new_spacing = (0.5, 0.5, 0.5)
    resampled = resample_image(mock_image, new_spacing, interpolation="nearest")
    # Expected shape: ceil(5 * 1.0 / 0.5) = 10
    assert resampled.array.shape == (10, 10, 10)


def test_resample_image_cubic(mock_image: Image) -> None:
    new_spacing = (1.5, 1.5, 1.5)
    resampled = resample_image(mock_image, new_spacing, interpolation="cubic")
    assert resampled.spacing == new_spacing


def test_resample_image_errors(mock_image: Image) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        resample_image(mock_image, (-1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="Unknown interpolation method"):
        resample_image(mock_image, (1.0, 1.0, 1.0), interpolation="unknown")


def test_resample_mask_threshold(mock_mask: Image) -> None:
    # Resample mask with thresholding
    new_spacing = (2.0, 2.0, 2.0)
    resampled_mask = resample_image(
        mock_mask, new_spacing, interpolation="linear", mask_threshold=0.5
    )
    # Check boolean-like behavior (0 or 1)
    unique = np.unique(resampled_mask.array)
    assert np.all(np.isin(unique, [0, 1]))
    assert resampled_mask.array.dtype == np.uint8


def test_resample_round_intensities(mock_image: Image) -> None:
    new_spacing = (1.2, 1.2, 1.2)
    resampled = resample_image(
        mock_image, new_spacing, interpolation="linear", round_intensities=True
    )
    assert np.all(resampled.array == np.round(resampled.array))


def test_resample_image_source_mask_geometry_mismatch_raises(
    mock_image: Image,
) -> None:
    shifted_source_mask = Image(
        array=np.ones(mock_image.array.shape, dtype=np.uint8),
        spacing=mock_image.spacing,
        origin=(10.0, 0.0, 0.0),
        direction=mock_image.direction,
        modality="SOURCE_MASK",
    )

    with pytest.raises(ValueError, match="Origin mismatch"):
        resample_image(
            mock_image,
            (1.0, 1.0, 1.0),
            source_mask=shifted_source_mask,
        )


def test_resample_image_source_mask_array_shape_mismatch_raises(
    mock_image: Image,
) -> None:
    with pytest.raises(ValueError, match="Source mask shape"):
        resample_image(
            mock_image,
            (1.0, 1.0, 1.0),
            source_mask=np.ones((2, 2, 2), dtype=bool),
        )


# --- Discretisation Tests ---


def test_discretise_image_fbn(mock_image: Image) -> None:
    # FBN with 5 bins
    disc_img = discretise_image(mock_image, method="FBN", n_bins=5)
    assert isinstance(disc_img, Image)
    assert np.min(disc_img.array) == 1
    assert np.max(disc_img.array) == 5
    assert disc_img.array.shape == mock_image.array.shape


def test_discretise_empty_image() -> None:
    # Empty image (all NaNs or shape 0)
    empty_arr = np.array([])
    disc = discretise_image(empty_arr, method="FBN", n_bins=5)
    assert isinstance(disc, np.ndarray)
    assert disc.size == 0

    # Image object with NaNs
    shape = (5, 5, 5)
    nan_img = Image(np.full(shape, np.nan), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    disc_nan = discretise_image(nan_img, method="FBN", n_bins=5)
    assert np.all(disc_nan.array == 0)


def test_discretise_image_fbn_explicit_range(mock_image: Image) -> None:
    # FBN with explicit min/max
    disc = discretise_image(mock_image, method="FBN", n_bins=5, min_val=0.0, max_val=10.0)
    assert disc.array.shape == mock_image.array.shape


def test_discretise_image_fbs(mock_image: Image) -> None:
    # FBS with bin width 2.0
    # Values range from 0 to 12. min=0.
    # bins: [0, 2) -> 1, [2, 4) -> 2, ...
    disc_img = discretise_image(mock_image, method="FBS", bin_width=2.0)
    arr = disc_img.array  # type: ignore
    assert np.min(arr) >= 1
    # Check specific value logic: val=3 -> floor((3-0)/2)+1 = floor(1.5)+1 = 2
    # mock_image(2,2,2) = 6 -> floor(6/2)+1 = 4
    # But wait, mock_image gradient depends on indexing.
    # z=0, y=0, x=3 -> val=3.
    pass


def test_discretise_image_fixed_cutoffs(mock_image: Image) -> None:
    cutoffs = [2.0, 5.0, 8.0]
    disc = discretise_image(mock_image, method="FIXED_CUTOFFS", cutoffs=cutoffs)
    # digitize returns 0 for values < cutoffs[0]
    arr = disc.array  # type: ignore
    assert np.all(arr >= 0)


def test_discretise_image_roi(mock_image: Image, mock_mask: Image) -> None:
    # Discretise only using ROI for min/max
    disc = discretise_image(mock_image, method="FBN", n_bins=5, roi_mask=mock_mask)
    assert disc.array.shape == mock_image.array.shape


def test_discretise_numpy_input() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    disc = discretise_image(arr, method="FBN", n_bins=3)
    assert isinstance(disc, np.ndarray)
    assert np.array_equal(disc, [1, 2, 3])


def test_discretise_errors(mock_image: Image) -> None:
    with pytest.raises(ValueError, match="Unknown discretisation method"):
        discretise_image(mock_image, method="UNKNOWN")

    with pytest.raises(ValueError, match="n_bins required for FBN"):
        discretise_image(mock_image, method="FBN")  # Missing n_bins

    with pytest.raises(ValueError, match="n_bins must be positive"):
        discretise_image(mock_image, method="FBN", n_bins=-1)

    with pytest.raises(ValueError, match="bin_width required for FBS"):
        discretise_image(mock_image, method="FBS")  # Missing bin_width

    with pytest.raises(ValueError, match="bin_width must be positive"):
        discretise_image(mock_image, method="FBS", bin_width=-1.0)

    with pytest.raises(ValueError, match="cutoffs required"):
        discretise_image(mock_image, method="FIXED_CUTOFFS")  # Missing cutoffs

    # Shape mismatch
    bad_mask = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="Shape mismatch"):
        discretise_image(mock_image, method="FBN", n_bins=5, roi_mask=bad_mask)

    shifted_mask = Image(
        array=np.ones(mock_image.array.shape, dtype=np.uint8),
        spacing=mock_image.spacing,
        origin=(5.0, 0.0, 0.0),
        direction=mock_image.direction,
    )
    with pytest.raises(ValueError, match="Origin mismatch"):
        discretise_image(mock_image, method="FBN", n_bins=5, roi_mask=shifted_mask)


def test_discretise_empty_roi(mock_image: Image) -> None:
    empty_mask = np.zeros(mock_image.array.shape)
    # Fallback to global min/max
    disc = discretise_image(mock_image, method="FBN", n_bins=5, roi_mask=empty_mask)
    assert disc.array.shape == mock_image.array.shape


def test_discretise_flat_region() -> None:
    flat_img = np.ones((5, 5, 5))
    disc = discretise_image(flat_img, method="FBN", n_bins=5)
    assert np.all(disc == 1)


# --- apply_mask Tests ---


def test_apply_mask_simple(mock_image: Image, mock_mask: Image) -> None:
    values = apply_mask(mock_image, mock_mask)
    # Mask has 3x3x3 = 27 voxels
    assert values.size == 27


def test_apply_mask_none_values(mock_image: Image, mock_mask: Image) -> None:
    # Explicit None uses all nonzero mask values.
    values = apply_mask(mock_image, mock_mask, mask_values=None)
    assert values.size == 27


def test_apply_mask_treats_nonzero_labels_as_roi(mock_image: Image) -> None:
    mask_array = np.zeros(mock_image.array.shape, dtype=np.uint8)
    mask_array[1:3, 1:3, 1:3] = 2
    mask_array[3, 3, 3] = 5
    label_mask = Image(mask_array, mock_image.spacing, mock_image.origin, mock_image.direction)

    values = apply_mask(mock_image, label_mask)
    assert values.size == 9

    values_label_2 = apply_mask(mock_image, label_mask, mask_values=2)
    assert values_label_2.size == 8


def test_apply_mask_errors(mock_image: Image) -> None:
    # Shape mismatch
    with pytest.raises(ValueError):
        apply_mask(mock_image, np.zeros((2, 2, 2)))

    shifted_mask = Image(
        array=np.ones(mock_image.array.shape, dtype=np.uint8),
        spacing=mock_image.spacing,
        origin=(5.0, 0.0, 0.0),
        direction=mock_image.direction,
    )
    with pytest.raises(ValueError, match="Origin mismatch"):
        apply_mask(mock_image, shifted_mask)

    # Empty result
    empty_mask = np.zeros(mock_image.array.shape)
    values_empty = apply_mask(mock_image, empty_mask)
    assert values_empty.size == 0


def test_apply_mask_gathers_a_large_array_in_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    # With 4 threads or more, a large row-order 3D array takes the parallel gather: the
    # values of image[mask != 0] in the same order and type, in a new array.
    from pictologics import preprocessing

    monkeypatch.setattr(preprocessing, "_KERNEL_MIN_SIZE", 0)
    monkeypatch.setattr(preprocessing, "get_num_threads", lambda: 4)
    rng = np.random.default_rng(15)
    image = rng.normal(0.0, 10.0, (5, 6, 7))
    image[0, 0, :3] = (np.nan, np.inf, -0.0)
    levels = rng.integers(-50, 50, image.shape).astype(np.int32)
    masks = (
        rng.integers(0, 3, image.shape).astype(np.uint8),
        rng.random(image.shape) < 0.5,
        rng.choice(np.array([0.0, 1.0, np.nan, -0.0, 0.25]), image.shape),  # NaN and 0.25: ROI
    )
    with patch.object(
        preprocessing, "_box_values_numba", wraps=preprocessing._box_values_numba
    ) as spy:
        for img in (image, levels):
            for mask in masks:
                got = apply_mask(img, mask)
                ref = img[mask != 0]
                assert got.dtype == ref.dtype and got.tobytes() == ref.tobytes()
                assert not np.shares_memory(got, img)
            empty = apply_mask(img, np.zeros(img.shape, dtype=np.uint8))
            assert empty.dtype == img.dtype and empty.size == 0
    assert spy.call_count == 8


def test_apply_mask_takes_numpy_where_the_threaded_gather_does_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Other types, another order, 2D arrays, mask values, fewer than 4 threads and a small
    # array take numpy's gather, with the same values.
    from pictologics import preprocessing
    from pictologics.preprocessing import _threaded_roi_values

    monkeypatch.setattr(preprocessing, "_KERNEL_MIN_SIZE", 0)
    monkeypatch.setattr(preprocessing, "get_num_threads", lambda: 4)
    rng = np.random.default_rng(16)
    image = rng.normal(0.0, 10.0, (5, 6, 7))
    mask = (rng.random(image.shape) < 0.5).astype(np.uint8)
    cases = (
        (image.astype(np.float32), mask),
        (image, mask.astype(np.int16)),
        (np.asfortranarray(image), mask),
        (image[0], mask[0]),
    )
    with patch.object(
        preprocessing, "_box_values_numba", wraps=preprocessing._box_values_numba
    ) as spy:
        for img, m in cases:
            assert apply_mask(img, m).tobytes() == img[m != 0].tobytes()
        assert apply_mask(image, mask, mask_values=[1]).tobytes() == image[mask == 1].tobytes()
        monkeypatch.setattr(preprocessing, "get_num_threads", lambda: 3)
        assert apply_mask(image, mask).tobytes() == image[mask != 0].tobytes()
        monkeypatch.setattr(preprocessing, "get_num_threads", lambda: 4)
        monkeypatch.setattr(preprocessing, "_KERNEL_MIN_SIZE", 1 << 20)
        assert apply_mask(image, mask).tobytes() == image[mask != 0].tobytes()
        assert spy.call_count == 0
        monkeypatch.setattr(preprocessing, "_KERNEL_MIN_SIZE", 0)
        assert _threaded_roi_values(image[0], mask[0]) is None
        assert _threaded_roi_values(image, mask[:4]) is None
        box = (slice(2, 4), slice(1, 5), slice(0, 3))
        got = _threaded_roi_values(image, mask, box)
        assert got is not None and got.tobytes() == image[box][mask[box] != 0].tobytes()
        assert spy.call_count == 1


# --- extract_roi Tests ---


def test_extract_roi(mock_image: Image, mock_mask: Image) -> None:
    roi_img = extract_roi(mock_image, mock_mask)
    # Voxels outside mask should be NaN
    assert np.isnan(roi_img.array[0, 0, 0])
    # Voxels inside mask should be original values
    assert roi_img.array[2, 2, 2] == mock_image.array[2, 2, 2]

    # Error
    with pytest.raises(ValueError):
        extract_roi(mock_image, Image(np.zeros((2, 2, 2)), (1, 1, 1), (0, 0, 0)))

    shifted_mask = Image(
        np.ones(mock_image.array.shape, dtype=np.uint8),
        mock_image.spacing,
        (5.0, 0.0, 0.0),
        mock_image.direction,
    )
    with pytest.raises(ValueError, match="Origin mismatch"):
        extract_roi(mock_image, shifted_mask)


def test_extract_roi_none_values(mock_image: Image, mock_mask: Image) -> None:
    roi_img = extract_roi(mock_image, mock_mask, mask_values=None)
    assert roi_img.array[2, 2, 2] == mock_image.array[2, 2, 2]


def test_extract_roi_treats_nonzero_labels_as_roi(mock_image: Image) -> None:
    mask_array = np.zeros(mock_image.array.shape, dtype=np.uint8)
    mask_array[1:3, 1:3, 1:3] = 2
    mask_array[3, 3, 3] = 5
    label_mask = Image(mask_array, mock_image.spacing, mock_image.origin, mock_image.direction)

    roi_img = extract_roi(mock_image, label_mask)
    assert not np.isnan(roi_img.array[1, 1, 1])
    assert not np.isnan(roi_img.array[3, 3, 3])
    assert np.isnan(roi_img.array[0, 0, 0])

    roi_label_2 = extract_roi(mock_image, label_mask, mask_values=2)
    assert not np.isnan(roi_label_2.array[1, 1, 1])
    assert np.isnan(roi_label_2.array[3, 3, 3])


# --- resegment_mask Tests ---


def test_resegment_mask_defaults(mock_image: Image, mock_mask: Image) -> None:
    # No range specified, should return copy of mask
    new_mask = resegment_mask(mock_image, mock_mask)
    assert np.array_equal(new_mask.array, mock_mask.array)


def test_resegment_mask_logic(mock_image: Image, mock_mask: Image) -> None:
    # Exclude values < 5
    new_mask = resegment_mask(mock_image, mock_mask, range_min=5.0)

    # Original values: x+y+z
    # (2,2,2) -> 6 (>=5) -> Keep
    # (1,1,1) -> 3 (<5) -> Remove
    assert new_mask.array[2, 2, 2] == 1
    assert new_mask.array[1, 1, 1] == 0

    # Max range
    new_mask_max = resegment_mask(mock_image, mock_mask, range_max=5.0)
    assert new_mask_max.array[2, 2, 2] == 0  # 6 > 5
    assert new_mask_max.array[1, 1, 1] == 1  # 3 <= 5

    with pytest.raises(ValueError):
        resegment_mask(mock_image, Image(np.zeros((2, 2, 2)), (1, 1, 1), (0, 0, 0)))

    shifted_mask = Image(
        np.ones(mock_image.array.shape, dtype=np.uint8),
        mock_image.spacing,
        (5.0, 0.0, 0.0),
        mock_image.direction,
    )
    with pytest.raises(ValueError, match="Origin mismatch"):
        resegment_mask(mock_image, shifted_mask)


def test_resegment_removes_nan_voxels() -> None:
    # A NaN intensity is in no range: its voxel leaves the mask on the numpy path and in
    # the kernel, also with one bound or none.
    arr = _f64((4, 4, 4))
    arr[1, 1, 1] = np.nan
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    ones = np.ones((4, 4, 4), dtype=np.uint8)
    # A C mask takes the kernel; a small column-order mask (no shared order) the numpy chain
    for mask_arr in (ones, np.asfortranarray(ones)):
        mask = Image(mask_arr, (1, 1, 1), (0, 0, 0))
        for bounds in ((None, None), (0.0, None), (None, 100.0)):
            out = resegment_mask(img, mask, *bounds).array
            assert out[1, 1, 1] == 0 and out.sum() == 63


# --- filter_outliers Tests ---


def test_filter_outliers(mock_image: Image, mock_mask: Image) -> None:
    # Modify image to have an outlier
    arr = mock_image.array.copy()
    arr[2, 2, 2] = 1000.0  # Outlier
    outlier_img = Image(arr, mock_image.spacing, mock_image.origin)

    filtered_mask = filter_outliers(outlier_img, mock_mask, sigma=1.0)

    assert filtered_mask.array[2, 2, 2] == 0  # Removed
    assert filtered_mask.array[1, 1, 1] == 1  # Kept


def test_filter_outliers_float_mask(mock_image: Image) -> None:
    # Create float mask
    mask_arr = np.zeros(mock_image.array.shape, dtype=float)
    mask_arr[1:4, 1:4, 1:4] = 1.0
    mask = Image(mask_arr, mock_image.spacing, mock_image.origin)

    filtered = filter_outliers(mock_image, mask)
    # Mask dtype is preserved; outlier voxels are zeroed in place
    assert filtered.array.dtype == mask_arr.dtype
    assert np.all(filtered.array[mask_arr == 0] == 0)


def test_filter_outliers_bool_mask(mock_image: Image) -> None:
    # Create boolean mask
    mask_arr = np.zeros(mock_image.array.shape, dtype=bool)
    mask_arr[1:4, 1:4, 1:4] = True
    mask = Image(mask_arr, mock_image.spacing, mock_image.origin)

    # Image with outlier
    arr = mock_image.array.copy()
    arr[2, 2, 2] = 1000.0
    outlier_img = Image(arr, mock_image.spacing, mock_image.origin)

    filtered = filter_outliers(outlier_img, mask)
    # Check that it returns boolean mask or uint8?
    # The implementation returns boolean if input is boolean?
    # Let's check implementation behavior:
    # if new_mask_array.dtype == bool:
    #     new_mask_array = new_mask_array & valid_mask
    # return Image(..., array=new_mask_array, ...)
    # So it should remain boolean (or at least valid_mask is boolean).

    assert filtered.array.dtype == bool
    # Outlier at 2,2,2 should be removed (False)
    assert not filtered.array[2, 2, 2]
    # Normal value at 1,1,1 should be kept (True)
    assert filtered.array[1, 1, 1]


def test_filter_outliers_empty(mock_image: Image) -> None:
    empty = Image(np.zeros(mock_image.array.shape), mock_image.spacing, mock_image.origin)
    res = filter_outliers(mock_image, empty)
    assert np.sum(res.array) == 0


# --- Other Utilities ---


def test_round_intensities() -> None:
    img_arr = np.array([[[1.2, 1.8, 2.5]]])
    img = Image(img_arr, (1, 1, 1), (0, 0, 0))
    rounded = round_intensities(img)
    # 2.5 rounds to 2.0 (nearest even)
    assert np.allclose(rounded.array, [[[1.0, 2.0, 2.0]]])


def test_round_intensities_in_threads_gives_the_bits_of_np_round() -> None:
    # From _KERNEL_MIN_SIZE voxels on, with more than one thread, a float64 image in row
    # or column order is rounded by _rint_numba: the same bits as np.round, the same
    # memory order. Else (one thread, other types, no shared order) np.round.
    from pictologics import preprocessing

    values = np.array(
        [0.5, 1.5, 2.5, -0.5, -2.5, -0.4, -0.0, 0.0, 0.49999999999999994, 2.4999999999999996,
         4503599627370495.5, 1e300, -1e300, 5e-324, -5e-324, np.nan, np.inf, -np.inf,
         7.5, -7.5, 1023.5, -1000.5, 3.3, -3.7]
    ).reshape(2, 3, 4)  # fmt: skip
    cases = [
        (values, 2, 1),
        (np.asfortranarray(values), 2, 1),
        (values, 1, 0),  # one thread
        (np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 0.5, 2, 0),  # another type
        (np.asfortranarray(values)[:, :, ::2], 2, 0),  # no shared order
        (values[:1, :1, :7], 2, 0),  # below the size
    ]
    for array, threads, calls in cases:
        with (
            patch.object(preprocessing, "_KERNEL_MIN_SIZE", 8),
            patch.object(preprocessing, "get_num_threads", return_value=threads),
            patch.object(preprocessing, "_rint_numba", wraps=preprocessing._rint_numba) as kernel,
        ):
            out = round_intensities(Image(array, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))).array
        expected = np.round(array)
        assert kernel.call_count == calls
        assert out.dtype == expected.dtype
        assert out.flags.f_contiguous == expected.flags.f_contiguous
        assert out.flags.c_contiguous == expected.flags.c_contiguous
        assert out.tobytes(order="A") == expected.tobytes(order="A")
        assert not np.shares_memory(out, array)


def test_keep_largest_component(mock_image: Image) -> None:
    mask_arr = np.zeros(mock_image.array.shape, dtype=np.uint8)
    # Component 1 (size 2)
    mask_arr[0, 0, 0] = 1
    mask_arr[0, 0, 1] = 1
    # Component 2 (size 1)
    mask_arr[4, 4, 4] = 1

    mask = Image(mask_arr, mock_image.spacing, mock_image.origin)

    largest = keep_largest_component(mask)
    assert largest.array[0, 0, 0] == 1
    assert largest.array[4, 4, 4] == 0

    # Run again on single component
    again = keep_largest_component(largest)
    assert np.array_equal(again.array, largest.array)


def test_keep_largest_component_2d() -> None:
    # A non-3D mask takes the full-array path (no bounding-box crop): a full mask stays
    # full, and of two components the larger one stays.
    out = keep_largest_component(Image(np.ones((4, 4), dtype=np.uint8), (1.0, 1.0), (0.0, 0.0)))
    assert out.array.shape == (4, 4)
    assert out.array.all()
    two = np.zeros((6, 8), dtype=np.uint8)
    two[0:2, 0:2] = 1
    two[3:6, 4:8] = 1
    out = keep_largest_component(Image(two, (1.0, 1.0), (0.0, 0.0)))
    assert int(out.array.sum()) == 12 and out.array[0, 0] == 0


def test_keep_largest_component_full_box() -> None:
    # A mask that fills its bounding box is one component: no labeling
    arr = np.zeros((8, 9, 7), dtype=np.int16)
    arr[1:6, 2:8, 1:5] = np.arange(1, 121).reshape(5, 6, 4)
    with patch("pictologics.preprocessing.label", side_effect=AssertionError("label ran")):
        out = keep_largest_component(Image(arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))).array
    assert out.dtype == np.uint8
    assert_array_equal(out, (arr != 0).astype(np.uint8))


def test_keep_largest_component_one_component_with_hole() -> None:
    # One component that does not fill its box (a hole; NaN counts as nonzero)
    arr = np.zeros((7, 7, 7))
    arr[1:6, 1:6, 1:6] = 1.0
    arr[3, 3, 3] = 0.0
    arr[1, 1, 1] = np.nan
    out = keep_largest_component(Image(arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))).array
    assert_array_equal(out, (arr != 0).astype(np.uint8))


def test_extract_roi_integer_image() -> None:
    # An integer image is upcast to float (NaN-capable) before masking.
    img = Image(np.arange(27, dtype=np.int32).reshape(3, 3, 3), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mask = Image(np.ones((3, 3, 3), dtype=np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    assert extract_roi(img, mask).array.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.bool_])
def test_extract_roi_matches_copy_and_fill(dtype: type) -> None:
    # The ROI voxels keep their bits and the other voxels become NaN, for each memory order.
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 50.0, (6, 5, 4)).astype(dtype)
    roi = rng.random((6, 5, 4)) > 0.5
    floating = np.issubdtype(values.dtype, np.floating)
    expected = values.astype(values.dtype if floating else np.float64)
    expected[~roi] = np.nan
    for array in (values, np.asfortranarray(values)):
        image = Image(array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        mask = Image(roi.astype(np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        out = extract_roi(image, mask).array
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected, equal_nan=True)
        assert not np.shares_memory(out, array)


# ---------------------------------------------------------------------------
# Sentinel detection & source-mask creation
# (merged from the former test_preprocessing_coverage.py)
# ---------------------------------------------------------------------------


def test_detect_sentinel_value_basic() -> None:
    arr = np.full((10, 10, 10), -2048.0, dtype=np.float32)
    arr[2:8, 2:8, 2:8] = 100.0
    assert detect_sentinel_value(Image(arr, (1, 1, 1), (0, 0, 0))) == -2048.0


def test_detect_sentinel_value_minus_3024() -> None:
    """-3024 HU (outside the CT reconstruction FOV) is a recognised sentinel."""
    assert -3024.0 in COMMON_SENTINEL_VALUES
    arr = np.full((10, 10, 10), -3024.0, dtype=np.float32)
    arr[2:8, 2:8, 2:8] = 100.0
    assert detect_sentinel_value(Image(arr, (1, 1, 1), (0, 0, 0))) == -3024.0


def test_detect_sentinel_value_none() -> None:
    # All-zero image: 0.0 is a default candidate and dominates -> detected.
    zeros = Image(np.zeros((10, 10, 10), np.float32), (1, 1, 1), (0, 0, 0))
    assert detect_sentinel_value(zeros) == 0.0
    # Noise with no candidate value present -> None.
    noise = np.random.rand(10, 10, 10).astype(np.float32) + 100.0
    assert detect_sentinel_value(Image(noise, (1, 1, 1), (0, 0, 0))) is None


def test_detect_sentinel_with_roi() -> None:
    shape = (20, 20, 20)
    roi = np.zeros(shape, dtype=np.uint8)
    roi[5:15, 5:15, 5:15] = 1
    roi_img = Image(roi, (1, 1, 1), (0, 0, 0))
    arr = np.full(shape, -1024.0, dtype=np.float32)
    arr[5:15, 5:15, 5:15] = 50.0
    assert detect_sentinel_value(Image(arr, (1, 1, 1), (0, 0, 0)), roi_mask=roi_img) == -1024.0

    # A candidate that lives inside the ROI is not treated as a background sentinel.
    arr2 = np.zeros(shape, dtype=np.float32)
    arr2[5:15, 5:15, 5:15] = -1024.0
    assert (
        detect_sentinel_value(
            Image(arr2, (1, 1, 1), (0, 0, 0)),
            candidate_values=(-1024.0,),
            roi_mask=roi_img,
        )
        is None
    )


def test_detect_sentinel_below_threshold() -> None:
    # A candidate occupying < 5% of the image must not be detected.
    arr = np.full((10, 10, 10), 100.0, dtype=np.float32)
    arr.flat[:30] = -1024.0  # 3% of 1000 voxels
    assert detect_sentinel_value(Image(arr, (1, 1, 1), (0, 0, 0))) is None


def test_detect_sentinel_highest_proportion_wins() -> None:
    # When two candidates both exceed the threshold, the larger fraction wins.
    arr = np.full((10, 10, 10), 100.0, dtype=np.float32)
    arr.flat[:100] = -1024.0  # 10%
    arr.flat[100:400] = 0.0  # 30%
    assert detect_sentinel_value(Image(arr, (1, 1, 1), (0, 0, 0))) == 0.0


def test_create_source_mask_from_sentinel() -> None:
    img = Image(np.array([-2048.0, 100.0, -2048.0], dtype=np.float32), (1, 1, 1), (0, 0, 0))
    mask = create_source_mask_from_sentinel(img, -2048.0)
    assert mask.modality == "SOURCE_MASK"
    assert_array_equal(mask.array, [0, 1, 0])  # 0 where sentinel, 1 where valid

    img_tol = Image(np.array([-2048.1, -2047.9, 100.0], dtype=np.float32), (1, 1, 1), (0, 0, 0))
    mask_tol = create_source_mask_from_sentinel(img_tol, -2048.0, tolerance=0.5)
    assert_array_equal(mask_tol.array, [0, 0, 1])
    # A float64 image in row or column order (the kernel): uint8, the order of the image
    arr = np.where(np.random.default_rng(4).random((5, 6, 7)) < 0.3, -2048.0, 40.0)
    for array in (arr, np.asfortranarray(arr)):
        out = create_source_mask_from_sentinel(Image(array, (1, 1, 1), (0, 0, 0)), -2048.0).array
        assert out.dtype == np.uint8 and out.flags.f_contiguous == array.flags.f_contiguous
        assert_array_equal(out, (array != -2048.0).astype(np.uint8))
        tol = create_source_mask_from_sentinel(Image(array, (1, 1, 1), (0, 0, 0)), -2048.0, 0.5)
        assert tol.array.dtype == np.uint8
        assert_array_equal(tol.array, out)


def test_not_sentinel_matches_numpy() -> None:
    # _not_sentinel gives array != value as bool in the order of the array: the kernel for
    # a float64 array in row or column order, numpy for a strided view and other types.
    from pictologics.preprocessing import _not_sentinel

    arr = np.random.default_rng(6).normal(40.0, 20.0, (6, 7, 8))
    arr[0], arr[:, 3], arr[2, 2, 2], arr[4, 5, 6] = -2048.0, -2048.0, np.nan, -0.0
    for array in (arr, np.asfortranarray(arr), arr[::2], arr.astype(np.float32)):
        for value in (-2048.0, 0.0, np.nan):
            got = _not_sentinel(array, value)
            assert got.dtype == np.bool_
            assert_array_equal(got, array != value)
            assert got.flags.c_contiguous == (array != value).flags.c_contiguous
            assert got.flags.f_contiguous == (array != value).flags.f_contiguous


def test_resample_with_source_mask() -> None:
    # 3D column [10, sentinel, 30] with the centre flagged invalid.
    arr = np.array([[[10.0]], [[-1000.0]], [[30.0]]], dtype=np.float32)
    img = Image(arr, (1.0, 1.0, 1.0), (0, 0, 0))
    src = np.array([[[1]], [[0]], [[1]]], dtype=np.uint8)
    img_masked = img.with_source_mask(Image(src, img.spacing, img.origin))

    # Default weight_threshold=0.5: gap voxels are zeroed and flagged invalid;
    # the sentinel must not leak into any valid output voxel.
    r = resample_image(img_masked, new_spacing=(0.5, 1.0, 1.0), interpolation="linear")
    data, valid = r.array.flatten(), r.source_mask.flatten()
    assert np.all(data[valid] > 0)
    assert np.all(data[valid] < 40)
    assert np.all(data[~valid] == 0)
    assert not np.all(valid)

    # A permissive threshold restores gap-filling via normalized convolution.
    rf = resample_image(
        img_masked, new_spacing=(0.5, 1.0, 1.0), interpolation="linear", weight_threshold=0.01
    )
    df = rf.array.flatten()
    assert np.all(df > 0)
    assert np.all(df < 40)
    assert np.all(rf.source_mask)


# ---------------------------------------------------------------------------
# Numba kernel paths (float64, size >= a kernel's limit, for example
# _KERNEL_MIN_SIZE for arrays that share no memory order; the discretise and resegment
# kernels take every row- or column-order float64 array). The limits are patched small
# so tiny arrays exercise the single-pass kernels; the optimization work proved these
# bit-identical to the numpy fallback.
# ---------------------------------------------------------------------------


def _f64(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)


def test_discretise_fbn_kernel_clamps() -> None:
    arr = _f64((4, 4, 4))
    arr[0, 0, 0] = np.nan  # NaN maps to bin 0
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    # Explicit range narrower than the data: values below it clamp to bin 1,
    # values above clamp to n_bins.
    out = discretise_image(img, method="FBN", n_bins=8, min_val=20.0, max_val=40.0)
    assert out.array[0, 0, 0] == 0
    assert out.array.min() >= 0
    assert out.array.max() <= 8


def test_discretise_fbs_kernel_clamps() -> None:
    # NaN and +inf map to bin 0 (no cast of infinity to an integer), -inf to bin 1, on
    # both paths: the kernel (row order) and the numpy chain (a layout of neither order).
    # The default minimum is the smallest finite ROI value.
    arr = _f64((4, 4, 4))
    arr[0, 0, :3] = (np.nan, np.inf, -np.inf)
    for layout in (arr, np.swapaxes(arr, 0, 1)):
        img = Image(layout, (1, 1, 1), (0, 0, 0))
        out = discretise_image(img, method="FBS", bin_width=5.0, min_val=20.0, max_val=40.0)
        assert out.array[0, 0, :3].tolist() == [0, 0, 1]
        out = discretise_image(img, method="FBS", bin_width=5.0, roi_mask=np.ones((4, 4, 4)))
        assert out.array[0, 0, 3] == 1 and out.array[3, 3, 3] == 13  # (63 - 3) / 5 + 1


def test_discretise_kernel_keeps_array_order() -> None:
    # The kernels give the numpy chain's bins and keep the layout of row- and column-order
    # input. The DICOM loader's layout (neither) needs a copy: below _KERNEL_MIN_SIZE it
    # keeps the numpy chain, from it the kernels take a copy.
    rng = np.random.default_rng(5)
    row_order = rng.normal(0.0, 50.0, (5, 6, 7))
    row_order[1, 2, 3] = np.nan
    dicom_layout = np.swapaxes(row_order, 0, 1)
    for method, kw in (("FBN", {"n_bins": 8}), ("FBS", {"bin_width": 10.0})):
        with (
            patch("pictologics.preprocessing._discretise_fbn_numba", side_effect=AssertionError),
            patch(
                "pictologics.preprocessing._discretise_fbn_numba_serial",
                side_effect=AssertionError,
            ),
            patch("pictologics.preprocessing._discretise_fbs_numba", side_effect=AssertionError),
            patch(
                "pictologics.preprocessing._discretise_fbs_numba_serial",
                side_effect=AssertionError,
            ),
        ):
            ref = np.swapaxes(discretise_image(dicom_layout, method, **kw), 0, 1)
        for arr in (row_order, np.asfortranarray(row_order)):
            out = discretise_image(arr, method, **kw)
            assert_array_equal(out, ref)
            assert out.flags.f_contiguous == arr.flags.f_contiguous
        with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
            out = discretise_image(dicom_layout, method, **kw)
        assert_array_equal(np.swapaxes(out, 0, 1), ref)


def test_discretise_cutoffs_kernel_matches_digitize() -> None:
    # The cutoff kernel gives the bins of digitize (+1, NaN to 0) for increasing,
    # decreasing, repeated and no cutoffs, in row and column order. The DICOM layout
    # (neither order) takes digitize. Cutoffs that go up and down keep the error of
    # digitize.
    rng = np.random.default_rng(6)
    arr = rng.normal(0.0, 50.0, (5, 6, 7))
    arr[1, 2, 3] = np.nan
    arr[0, 0, :3] = (-20.0, 0.0, 20.0)  # values on the cutoffs
    for cutoffs in ([-20.0, 0.0, 20.0], [20.0, 0.0, -20.0], [0.0, 0.0, 20.0], []):
        expected = np.digitize(arr, cutoffs) + 1
        expected[np.isnan(arr)] = 0
        for layout in (arr, np.asfortranarray(arr), np.swapaxes(arr, 0, 1)):
            out = discretise_image(layout, "FIXED_CUTOFFS", cutoffs=cutoffs)
            assert out.dtype == np.int32
            assert_array_equal(out if layout.flags.forc else np.swapaxes(out, 0, 1), expected)
    with pytest.raises(ValueError, match="monotonically"):
        discretise_image(arr, "FIXED_CUTOFFS", cutoffs=[0.0, 20.0, 10.0])


def test_discretise_roi_search_only_for_missing_bounds() -> None:
    # Default bounds come from the ROI values: one fused pass for a float64 or uint8
    # row-order mask, a gather for other masks. An all-NaN or empty ROI falls back to the
    # whole image. Given bounds skip the ROI search.
    rng = np.random.default_rng(9)
    arr = rng.normal(0.0, 50.0, (6, 7, 8))
    arr[2, 3, 4] = np.nan
    mask = np.zeros(arr.shape)
    mask[1:4, 2:6, 3:7] = rng.random((3, 4, 4)) < 0.7
    values = arr[mask > 0]
    values = values[~np.isnan(values)]
    expected = discretise_image(arr, "FBN", n_bins=8, min_val=values.min(), max_val=values.max())
    for m in (mask, mask.astype(np.uint8), mask > 0, np.asfortranarray(mask)):
        assert_array_equal(discretise_image(arr, "FBN", roi_mask=m, n_bins=8), expected)
    all_nan = arr.copy()
    all_nan[mask > 0] = np.nan
    for data, m in ((all_nan, mask), (arr, np.zeros(arr.shape))):
        lo, hi = np.nanmin(data), np.nanmax(data)
        ref = discretise_image(data, "FBN", n_bins=8, min_val=lo, max_val=hi)
        assert_array_equal(discretise_image(data, "FBN", roi_mask=m, n_bins=8), ref)

    class NoSearch(np.ndarray):
        def __ne__(self, other: object) -> np.ndarray:  # type: ignore[override]
            raise AssertionError("the ROI search ran")

    no_gather = (mask > 0).view(NoSearch)  # a bool mask takes the gather
    with patch(
        "pictologics.preprocessing.roi_min_max",
        side_effect=AssertionError("the ROI search ran"),
    ):
        discretise_image(arr, "FBS", roi_mask=no_gather, bin_width=10.0, min_val=-100.0)
        discretise_image(arr, "FBN", roi_mask=mask, n_bins=8, min_val=-9.0, max_val=9.0)
        for m in (mask, no_gather):
            with pytest.raises(AssertionError, match="the ROI search ran"):
                discretise_image(arr, "FBN", roi_mask=m, n_bins=8)


def test_discretise_integer_input() -> None:
    # Integer input takes the out-of-place (promoting) division branch.
    img = Image(np.arange(27, dtype=np.int32).reshape(3, 3, 3), (1, 1, 1), (0, 0, 0))
    assert discretise_image(img, method="FBN", n_bins=4).array.max() <= 4
    assert discretise_image(img, method="FBS", bin_width=3.0).array.min() >= 1


def test_discretise_empty_image_returns_image() -> None:
    img = Image(np.zeros((0,), dtype=np.float64), (1, 1, 1), (0, 0, 0))
    out = discretise_image(img, method="FBN", n_bins=4)
    assert isinstance(out, Image)
    assert out.array.size == 0


def test_resample_nearest_kernel() -> None:
    img = Image(np.random.rand(8, 8, 8).astype(np.float64), (1, 1, 1), (0, 0, 0))
    assert resample_image(img, (0.7, 0.7, 0.7), interpolation="nearest").array.ndim == 3
    assert resample_image(img, (1.3, 1.3, 1.3), interpolation="nearest").array.ndim == 3


def test_resample_linear_kernel_and_all_valid_source() -> None:
    img = Image(np.random.rand(8, 8, 8).astype(np.float64), (1, 1, 1), (0, 0, 0))
    resample_image(img, (0.7, 0.7, 0.7), interpolation="linear")
    # An all-valid source mask collapses to "no mask", but the resampled image
    # still carries an all-valid source mask.
    all_valid = img.with_source_mask(Image(np.ones((8, 8, 8), np.uint8), img.spacing, img.origin))
    r = resample_image(all_valid, (0.7, 0.7, 0.7), interpolation="linear")
    assert r.source_mask is not None
    assert bool(r.source_mask.all())


def test_small_resamples_take_the_kernels() -> None:
    # A small 3D resample takes the kernels (no scipy call), with the bits of scipy, the
    # threshold rule and the source-mask resample of scipy
    from scipy.ndimage import affine_transform

    from pictologics import preprocessing as module

    rng = np.random.default_rng(22)
    img = Image(rng.normal(0.0, 50.0, (3, 4, 5)), (1.0, 1.3, 0.8), (0.0, 0.0, 0.0))
    mask = Image((rng.random((3, 4, 5)) < 0.5).astype(np.uint8), img.spacing, img.origin)
    valid = rng.random((3, 4, 5)) < 0.7
    new_spacing = (0.6, 0.9, 1.1)
    shape, scale, shift, _ = module._output_grid(img.array.shape, img.spacing, new_spacing)

    def scipy_out(arr: np.ndarray, order: int) -> np.ndarray:
        return affine_transform(
            arr, np.array(scale), np.array(shift), output_shape=shape, order=order, mode="nearest"
        )

    with (
        patch.object(module, "affine_transform") as scipy,
        patch.object(module, "_resample_with_source_mask") as masked_scipy,
    ):
        linear = resample_image(img, new_spacing).array
        nearest = resample_image(mask, new_spacing, "nearest").array
        thr = resample_image(mask, new_spacing, mask_threshold=0.5).array
        masked = resample_image(img, new_spacing, source_mask=valid)
    assert scipy.call_count == 0 and masked_scipy.call_count == 0
    assert linear.tobytes() == scipy_out(img.array, 1).tobytes()
    assert nearest.tobytes() == scipy_out(mask.array, 0).tobytes()
    expected = (scipy_out(mask.array.astype(np.float64), 1) + 0.5 >= 1.0).astype(np.uint8)
    assert_array_equal(thr, expected)
    ref, ref_valid = module._resample_with_source_mask(
        img.array, valid, np.array(scale), np.array(shift), shape, 1, "nearest", 0.5
    )
    assert masked.array.tobytes() == ref.tobytes()
    assert masked.source_mask is not None and np.array_equal(masked.source_mask, ref_valid)


def test_resample_kernels_match_scipy() -> None:
    # The kernels give scipy's output bit for bit at every size: -0.0 values, rounding,
    # mask thresholds, uint8 input, the source-mask path, and nearest mode.
    from scipy.ndimage import affine_transform

    from pictologics.preprocessing import _resample_with_source_mask

    rng = np.random.default_rng(6)
    spacing, new_spacing = (3.27, 5.0, 0.8), (2.2, 4.0, 2.0)
    image = np.round(rng.normal(0.0, 3.0, (7, 6, 9)))  # has -0.0 values
    mask = (rng.random((7, 6, 9)) < 0.5).astype(np.uint8)
    source = rng.random((7, 6, 9)) < 0.7
    uint8_img = rng.integers(0, 256, image.shape).astype(np.uint8)
    shape = (11, 8, 4)
    matrix = np.array(new_spacing) / np.array(spacing)
    offset = (np.array(image.shape) - 1) / 2.0 - matrix * (np.array(shape) - 1) / 2.0

    def scipy_out(arr: np.ndarray, order: int = 1) -> np.ndarray:
        return affine_transform(
            arr, matrix=matrix, offset=offset, output_shape=shape, order=order, mode="nearest"
        )

    def same(a: np.ndarray, b: np.ndarray) -> bool:
        return a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))

    img = Image(image, spacing, (0, 0, 0))
    masked = img.with_source_mask(Image(source.astype(np.uint8), spacing, (0, 0, 0)))
    ref, ref_valid = _resample_with_source_mask(
        image, source, matrix, offset, shape, 1, "nearest", 0.5
    )
    assert same(resample_image(img, new_spacing).array, scipy_out(image))
    rounded = resample_image(img, new_spacing, round_intensities=True).array
    assert same(rounded, np.round(scipy_out(image)))
    for arr in (mask, mask.astype(np.float64)):
        out = resample_image(Image(arr, spacing, (0, 0, 0)), new_spacing, mask_threshold=0.5)
        assert same(out.array, (scipy_out(arr) + 0.5 >= 1.0).astype(np.uint8))
    uint8_out = resample_image(Image(uint8_img, spacing, (0, 0, 0)), new_spacing).array
    assert same(uint8_out, scipy_out(uint8_img))
    nearest = resample_image(img, new_spacing, interpolation="nearest").array
    assert same(nearest, scipy_out(image, order=0))
    out = resample_image(masked, new_spacing)
    assert same(out.array, ref)
    assert_array_equal(out.source_mask, ref_valid)

    # The cubic kernels (no size gate) give scipy's bits where scipy's C code fuses a * b + c
    # (FMA), as the arm64 wheels do: a probe of one line shows it. Else they agree to 1e-9.
    from scipy.ndimage import spline_filter1d

    from pictologics import preprocessing as module

    line = rng.normal(0.0, 1.0, 8)
    probe = line.reshape(1, 1, 8).copy()
    z = module._SPLINE_POLE
    module._prefilter_axis2_numba(probe, z, (1.0 - z) * (1.0 - 1.0 / z), z**8)
    fuses = probe.tobytes() == spline_filter1d(line, 3, mode="nearest").tobytes()
    rule = "bitwise, scipy fuses" if fuses else "rtol 1e-9, scipy does not fuse"

    def close(a: np.ndarray, b: np.ndarray) -> bool:
        return same(a, b) if fuses else a.dtype == b.dtype and np.allclose(a, b, 1e-9, 0.0)

    cubic = resample_image(img, new_spacing, "cubic").array
    assert close(cubic, scipy_out(image, order=3)), rule
    rounded = resample_image(img, new_spacing, "cubic", round_intensities=True).array
    assert close(rounded, np.round(scipy_out(image, order=3))), rule
    float_mask = mask.astype(np.float64)
    out = resample_image(
        Image(float_mask, spacing, (0, 0, 0)), new_spacing, "cubic", mask_threshold=0.5
    )
    assert same(out.array, (scipy_out(float_mask, order=3) + 0.5 >= 1.0).astype(np.uint8)), rule
    # With a source mask: the values under the same rule and the same validity, also for
    # the weight threshold 1.0 and for a mask with no valid voxel
    for valid, threshold in ((source, 0.5), (source, 1.0), (np.zeros_like(source), 0.5)):
        ref, ref_valid = _resample_with_source_mask(
            image, valid, matrix, offset, shape, 3, "nearest", threshold
        )
        out = resample_image(
            img, new_spacing, "cubic", source_mask=valid, weight_threshold=threshold
        )
        assert close(out.array, ref), rule
        assert_array_equal(out.source_mask, ref_valid)
    assert not out.source_mask.any() and not out.array.any()
    # A float32 image and another boundary stay on scipy, also with a source mask
    with (
        patch.object(module, "affine_transform", wraps=affine_transform) as scipy,
        patch.object(
            module, "_resample_with_source_mask", wraps=_resample_with_source_mask
        ) as masked_scipy,
    ):
        resample_image(Image(image.astype(np.float32), spacing, (0, 0, 0)), new_spacing, "cubic")
        resample_image(img, new_spacing, "cubic", boundary_mode="constant")
        float32 = Image(image.astype(np.float32), spacing, (0, 0, 0))
        resample_image(float32, new_spacing, "cubic", source_mask=source)
    assert scipy.call_count == 4 and masked_scipy.call_count == 1


def test_resample_cubic_region_is_the_cut_of_the_whole_grid() -> None:
    # A region of the cubic grid has the bits of the whole grid's cut: the coefficients come
    # once and scipy never runs. Also a region of one voxel and a region at the far border,
    # and with a source mask (the same values and validity).
    from pictologics import preprocessing as module

    rng = np.random.default_rng(22)
    img = Image(rng.normal(0.0, 100.0, (7, 9, 8)), (1.3, 0.9, 2.1), (5.0, -3.0, 1.0))
    valid = rng.random(img.array.shape) < 0.7
    spacing = (0.7, 0.8, 1.1)
    whole = resample_image(img, spacing, "cubic").array
    whole_masked = resample_image(img, spacing, "cubic", source_mask=valid)
    depth, height, width = whole.shape
    regions = [
        (slice(2, depth - 1), slice(0, 5), slice(3, width)),
        (slice(4, 5), slice(6, 7), slice(0, 1)),
        (slice(depth - 2, depth), slice(height - 3, height), slice(width - 1, width)),
    ]
    for region in regions:
        with (
            patch.object(module, "_spline_coefficients", wraps=module._spline_coefficients) as coef,
            patch.object(
                module, "_spline_coefficients_masked", wraps=module._spline_coefficients_masked
            ) as coef_masked,
            patch.object(module, "affine_transform") as scipy,
        ):
            part = resample_image(img, spacing, "cubic", region=region).array
            masked = resample_image(img, spacing, "cubic", source_mask=valid, region=region)
        assert coef.call_count == 1 and coef_masked.call_count == 1 and scipy.call_count == 0
        assert part.shape == whole[region].shape
        assert part.tobytes() == np.ascontiguousarray(whole[region]).tobytes()
        assert masked.array.tobytes() == np.ascontiguousarray(whole_masked.array[region]).tobytes()
        assert_array_equal(masked.source_mask, whole_masked.source_mask[region])


def test_resample_copies_a_column_order_image_with_the_tiled_kernel() -> None:
    # A large column-order image (and a bool source mask, as its uint8 bytes) takes the tiled
    # copy of the loaders before the linear, nearest, cubic and masked kernels: the bits of
    # the row-order run. A strided view and a row-order image do not take it.
    from pictologics import loader
    from pictologics.preprocessing import _row_order_copy

    rng = np.random.default_rng(14)
    values = rng.normal(0.0, 50.0, (6, 7, 8))
    labels = (rng.random(values.shape) < 0.5).astype(np.uint8)
    valid = rng.random(values.shape) < 0.8
    spacing, new_spacing = (1.0, 1.1, 1.2), (0.7, 0.8, 0.9)

    def resample(arr: np.ndarray, **kw: Any) -> Any:
        return resample_image(Image(arr, spacing, (0, 0, 0)), new_spacing, **kw)

    cases = [
        (values, {"interpolation": "linear"}, 1),
        (values, {"interpolation": "nearest"}, 1),
        (labels, {"interpolation": "nearest"}, 1),
        (labels, {"interpolation": "linear", "mask_threshold": 0.5}, 1),
        (values, {"interpolation": "linear", "source_mask": valid}, 2),
        (values, {"interpolation": "cubic"}, 1),
        (values, {"interpolation": "cubic", "source_mask": valid}, 2),
    ]
    for arr, kw, copies in cases:
        expected = resample(arr, **kw)
        column_kw = {**kw, "source_mask": np.asfortranarray(valid)} if "source_mask" in kw else kw
        with (
            patch.object(loader, "_ROW_ORDER_MIN_SIZE", 8),
            patch.object(loader, "_to_row_order_numba", wraps=loader._to_row_order_numba) as tiled,
        ):
            got = resample(np.asfortranarray(arr), **column_kw)
            assert tiled.call_count == copies
            strided = resample(np.repeat(arr, 2, axis=2)[:, :, ::2], **kw)
            row = resample(arr, **kw)
            assert tiled.call_count == copies
        for out in (got, strided, row):
            assert out.array.tobytes() == expected.array.tobytes()
        if "source_mask" in kw:
            assert_array_equal(got.source_mask, expected.source_mask)
    # The helper: a row-order bool array stays (a view), a small column-order one is copied
    row_bool = np.zeros((3, 4, 5), dtype=bool)
    assert np.shares_memory(_row_order_copy(row_bool), row_bool)
    small = np.asfortranarray(valid)
    copied = _row_order_copy(small)
    assert copied.flags.c_contiguous and copied.dtype == np.bool_
    assert_array_equal(copied, small)
    assert _row_order_copy(values) is values


def test_resample_masked_linear_kernel() -> None:
    # A partial source mask + float64 + linear routes to the fused masked kernel.
    img = Image(np.random.rand(8, 8, 8).astype(np.float64), (1, 1, 1), (0, 0, 0))
    src = np.ones((8, 8, 8), dtype=np.uint8)
    src[3:5, 3:5, 3:5] = 0
    masked = img.with_source_mask(Image(src, img.spacing, img.origin))
    r = resample_image(masked, (0.7, 0.7, 0.7), interpolation="linear")
    assert r.array.ndim == 3
    assert r.source_mask is not None


def test_filter_outliers_kernels_match_numpy() -> None:
    # The kernels give the numpy path's ROI values (same order) and mask, bit for bit:
    # label values, NaN intensities, and -0.0 outside the ROI. An image and a mask with no
    # shared memory order take the numpy path; a shared order takes the serial kernel below
    # the gate (a column-order pair gives a column-order mask), the parallel kernel above
    # it, and the serial kernel above it at one thread.
    from pictologics import preprocessing
    from pictologics.features import _utils
    from pictologics.preprocessing import _roi_values_numba

    rng = np.random.default_rng(10)
    image = rng.normal(0.0, 10.0, (6, 7, 8))
    image[0, :3, 0] = np.nan
    image[2, 3, 4] = 80.0  # an outlier
    labels = np.where(rng.random(image.shape) < 0.6, 2.0, -0.0)
    img = Image(image, (1, 1, 1), (0, 0, 0))
    f_img = Image(np.asfortranarray(image), (1, 1, 1), (0, 0, 0))
    serial, parallel = (
        preprocessing._filter_outliers_serial_numba,
        preprocessing._filter_outliers_numba,
    )
    for mask_arr in (labels, (labels != 0).astype(np.uint8), labels != 0):
        np.testing.assert_array_equal(
            _roi_values_numba(image.ravel(), mask_arr.ravel()), image[mask_arr != 0]
        )
        mask = Image(mask_arr, (1, 1, 1), (0, 0, 0))
        f_mask = Image(np.asfortranarray(mask_arr), (1, 1, 1), (0, 0, 0))
        with (
            patch.object(preprocessing, "_filter_outliers_serial_numba", wraps=serial) as s,
            patch.object(preprocessing, "_filter_outliers_numba", wraps=parallel) as par,
        ):
            ref = filter_outliers(f_img, mask, 2.0).array  # numpy: no shared order
            assert s.call_count == par.call_count == 0
            outs = [
                filter_outliers(img, mask, 2.0).array,
                filter_outliers(f_img, f_mask, 2.0).array,
            ]
            assert outs[1].flags.f_contiguous and s.call_count == 2
            with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
                with patch.object(_utils, "get_num_threads", return_value=2):
                    outs.append(filter_outliers(img, mask, 2.0).array)
                assert par.call_count == 1
                with patch.object(_utils, "get_num_threads", return_value=1):
                    outs.append(filter_outliers(img, mask, 2.0).array)
                assert s.call_count == 3 and par.call_count == 1
        for out in outs:
            assert out.dtype == ref.dtype
            assert out.tobytes() == ref.tobytes()
        assert int(np.count_nonzero(ref)) < int(np.count_nonzero(mask_arr))


def test_filter_outliers_numpy_path_for_other_types() -> None:
    # A uint16 mask and a float32 image take the numpy path (the kernels compile for a
    # float64 image and a float64, uint8 or bool mask only), with the same rule.
    from pictologics import preprocessing

    rng = np.random.default_rng(13)
    image = rng.normal(0.0, 10.0, (6, 7, 8))
    image[2, 3, 4] = 80.0  # an outlier
    labels = np.where(rng.random(image.shape) < 0.6, 3, 0)
    pairs = ((image, labels.astype(np.uint16)), (image.astype(np.float32), labels.astype(np.uint8)))
    for values, mask_arr in pairs:
        roi = mask_arr != 0
        mean, std = np.mean(values[roi]), np.std(values[roi], ddof=0)
        expected = mask_arr.copy()
        expected[roi & ((values < mean - 2.0 * std) | (values > mean + 2.0 * std))] = 0
        with (
            patch.object(preprocessing, "_filter_outliers_serial_numba") as s,
            patch.object(preprocessing, "_filter_outliers_numba") as par,
        ):
            out = filter_outliers(
                Image(values, (1, 1, 1), (0, 0, 0)), Image(mask_arr, (1, 1, 1), (0, 0, 0)), 2.0
            ).array
        assert s.call_count == par.call_count == 0
        assert out.dtype == mask_arr.dtype
        assert_array_equal(out, expected)
        assert np.count_nonzero(out) < np.count_nonzero(mask_arr)


def test_detect_sentinel_one_pass_matches_numpy() -> None:
    # The one-pass count gives the per-candidate loop's answer, with and without an ROI
    # mask, also for a repeated candidate.
    from pictologics.preprocessing import _sentinel_counts_numba

    rng = np.random.default_rng(11)
    arr = rng.normal(0.0, 100.0, (8, 8, 8))
    arr[:, :, :3] = -2048.0
    arr[:2, :, 3:] = -1000.0
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    roi = np.zeros(arr.shape)
    roi[:, :, 5:] = 1.0  # -2048 lies outside it; -1000 mostly inside
    candidates = (-1000.0, -2048.0, -1000.0)
    counts, inside = _sentinel_counts_numba(arr.ravel(), roi.ravel(), np.array(candidates))
    for k, value in enumerate(candidates):
        assert counts[k] == np.count_nonzero(arr == value)
        assert inside[k] == np.count_nonzero((arr == value) & (roi > 0))
    for roi_arr in (None, roi, roi.astype(np.uint8), roi > 0):
        roi_img = None if roi_arr is None else Image(roi_arr, (1, 1, 1), (0, 0, 0))
        ref = detect_sentinel_value(img, candidates, roi_mask=roi_img)
        with patch("pictologics.preprocessing._SENTINEL_KERNEL_MIN_SIZE", 8):
            assert detect_sentinel_value(img, candidates, roi_mask=roi_img) == ref
        assert ref == -2048.0
    # The kernel for exactly 6 candidates (repeated, NaN and both zeros among them)
    from pictologics.preprocessing import _sentinel_counts6_numba

    six = (-1000.0, -2048.0, -1000.0, np.nan, 0.0, -0.0)
    arr[0, 0, :4] = (0.0, -0.0, 0.0, np.nan)
    for roi_flat in (
        roi.ravel(),
        roi.astype(np.uint8).ravel(),
        (roi > 0).ravel(),
        np.empty(0, np.uint8),
    ):
        counts, inside = _sentinel_counts6_numba(arr.ravel(), roi_flat, np.array(six))
        for k, value in enumerate(six):
            assert counts[k] == np.count_nonzero(arr == value)
            expected = np.count_nonzero((arr == value) & (roi > 0)) if roi_flat.size else 0
            assert inside[k] == expected


def test_detect_sentinel_takes_the_six_candidate_kernel() -> None:
    # The default candidates (6) take the 6-candidate kernel, with the result of the numpy
    # loop; _sentinel_search also gives the count of the found value (0 when none).
    from pictologics import preprocessing as pp

    arr = np.random.default_rng(12).normal(40.0, 20.0, (6, 7, 8))
    arr[:, :, :3] = -2048.0
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    expected = (-2048.0, int(np.count_nonzero(arr == -2048.0)))
    assert pp._sentinel_search(img) == expected
    with (
        patch.object(pp, "_SENTINEL_KERNEL_MIN_SIZE", 8),
        patch.object(pp, "_sentinel_counts6_numba", wraps=pp._sentinel_counts6_numba) as spy,
    ):
        assert detect_sentinel_value(img) == -2048.0
        assert pp._sentinel_search(img) == expected
    assert spy.call_count == 2
    noise = Image(np.random.default_rng(13).normal(0.0, 1.0, (6, 7, 8)), (1, 1, 1), (0, 0, 0))
    assert pp._sentinel_search(noise) == (None, 0)


def test_detect_sentinel_kernel_pairs_every_layout() -> None:
    # The kernel pairs each voxel with its own ROI value: in a shared column order with
    # no copy, and for an image and an ROI in different orders with a copy. An int32 ROI
    # keeps the numpy loop.
    from pictologics import preprocessing as pp

    rng = np.random.default_rng(12)
    arr = np.where(rng.random((6, 7, 8)) < 0.5, -1000.0, 5.0)
    roi = (rng.random(arr.shape) < 0.5).astype(np.uint8)
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    kernel = pp._sentinel_counts_numba

    def spy(*args: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seen.append(kernel(*args))
        return seen[-1]

    def image(a: np.ndarray) -> Image:
        return Image(a, (1, 1, 1), (0, 0, 0))

    layouts = [
        (np.asfortranarray(arr), np.asfortranarray(roi)),
        (arr, np.asfortranarray(roi)),
        (arr, roi.astype(np.int32)),
    ]
    with patch.multiple(
        pp, _SENTINEL_KERNEL_MIN_SIZE=8, _KERNEL_MIN_SIZE=8, _sentinel_counts_numba=spy
    ):
        for a, r in layouts:
            pp.detect_sentinel_value(image(a), (-1000.0,), roi_mask=image(r))
    assert len(seen) == 2
    for counts, inside in seen:
        assert counts[0] == np.count_nonzero(arr == -1000.0)
        assert inside[0] == np.count_nonzero((arr == -1000.0) & (roi > 0))


def test_resegment_kernel() -> None:
    img = Image(_f64((4, 4, 4)), (1, 1, 1), (0, 0, 0))
    mask = Image(np.ones((4, 4, 4), dtype=np.uint8), (1, 1, 1), (0, 0, 0))
    with patch("pictologics.preprocessing._RESEGMENT_PARALLEL_MIN", 8):
        out = resegment_mask(img, mask, range_min=5.0, range_max=50.0)
    assert out.array.shape == (4, 4, 4)


def test_resegment_kernel_keeps_array_order() -> None:
    # The kernel gives the numpy chain's mask bit for bit (label values, -0.0, NaN
    # intensities) for row, column and mixed order. A shared column order stays column
    # order; an image and a mask in different orders need a copy.
    rng = np.random.default_rng(13)
    image = rng.normal(0.0, 50.0, (5, 6, 7))
    image[1, 2, 3] = np.nan
    labels = np.where(rng.random(image.shape) < 0.7, 3.0, -0.0)

    def image_of(a: np.ndarray) -> Image:
        return Image(a, (1, 1, 1), (0, 0, 0))

    for mask_arr in (labels, (labels != 0).astype(np.uint8), labels != 0):
        ref = resegment_mask(image_of(image), image_of(mask_arr), -40.0, 60.0).array
        for img_arr, m_arr in (
            (image, mask_arr),
            (np.asfortranarray(image), np.asfortranarray(mask_arr)),
            (image, np.asfortranarray(mask_arr)),
        ):
            with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
                out = resegment_mask(image_of(img_arr), image_of(m_arr), -40.0, 60.0).array
            assert out.dtype == ref.dtype
            bits = np.ascontiguousarray(out).view(np.uint8)
            assert np.array_equal(bits, np.ascontiguousarray(ref).view(np.uint8))
            both_f = img_arr.flags.f_contiguous and m_arr.flags.f_contiguous
            assert out.flags.f_contiguous == both_f


def test_resegment_kernel_keeps_the_bits_of_each_mask_type() -> None:
    # The kernel reads each mask as the integer type of its size, so it gives the bits
    # of the numpy chain for each mask type: negative labels, -0.0, NaN and infinite
    # values, and a big-endian mask. A complex mask takes the numpy chain.
    rng = np.random.default_rng(21)
    image = rng.normal(0.0, 50.0, (4, 5, 6))
    image[0, 0, :3] = (np.nan, np.inf, -np.inf)
    labels = rng.integers(-3, 4, image.shape)
    floats = rng.choice(np.array([0.0, -0.0, 3.0, -2.0, np.nan, np.inf]), image.shape)
    kinds = (np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
             np.uint64, np.float16, np.float32, np.float64, ">f8", np.complex128)  # fmt: skip
    for kind in kinds:
        mask = (floats if np.dtype(kind).kind in "fc" else labels).astype(kind)
        ref = mask.copy()
        ref[~((image >= -40.0) & (image <= 60.0))] = 0
        out = resegment_mask(
            Image(image, (1, 1, 1), (0, 0, 0)), Image(mask, (1, 1, 1), (0, 0, 0)), -40.0, 60.0
        ).array
        assert out.dtype == ref.dtype
        assert out.tobytes() == ref.tobytes()


def test_a_linear_resample_rounds_in_the_kernel() -> None:
    # The linear kernel rounds as np.round does (half to even, -0.0 kept, NaN and inf
    # kept), with no np.round(out=) pass on its path; the cubic path and the masked linear
    # path keep that pass. A uint8 image gives the same bytes with and without rounding.
    from pictologics import preprocessing

    rng = np.random.default_rng(29)
    values = np.arange(-6.0, 6.0) + 0.5
    tail = [-0.3, -0.0, 0.3, 2.0**52 + 1.0, -7.5, 1e300, -2.5, 3.5]
    ties = np.concatenate([values, tail]).reshape(4, 5, 1) * np.ones((1, 1, 3))
    special = rng.normal(0.0, 10.0, (4, 5, 3))
    special[1, 2, 1], special[2, 3, 0], special[3, 0, 2] = np.nan, np.inf, -np.inf
    real_round = np.round
    passes: list[bool] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        passes.append("out" in kwargs)
        return real_round(*args, **kwargs)

    for grid in (ties, special):
        img = Image(grid, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        valid = np.isfinite(grid) & (np.arange(grid.size).reshape(grid.shape) != 0)
        for spacing in ((1.0, 1.0, 1.0), (0.7, 0.9, 0.6)):
            # inf times a zero weight gives NaN, as in scipy; the kernels warn only as Python
            with np.errstate(invalid="ignore"):
                plain = resample_image(img, spacing, "linear").array
            with np.errstate(invalid="ignore"), patch.object(preprocessing.np, "round", spy):
                passes.clear()
                out = resample_image(img, spacing, "linear", round_intensities=True).array
                assert not any(passes)
                resample_image(img, spacing, "cubic", round_intensities=True)
                assert sum(passes) == 1
                passes.clear()
                resample_image(img, spacing, round_intensities=True, source_mask=valid)
                assert sum(passes) == 1
            assert out.tobytes() == real_round(plain).tobytes()
    unit = (1.0, 1.0, 1.0)
    out = resample_image(Image(ties, unit, (0.0, 0.0, 0.0)), unit, round_intensities=True).array
    assert np.signbit(out[2, 2, 0]) and out[2, 2, 0] == 0.0  # -0.3 gives -0.0
    assert out[0, 0, 0] == -6.0 and out[0, 1, 0] == -4.0  # -5.5 and -4.5: half to even
    u8 = Image(rng.integers(0, 256, (4, 5, 6)).astype(np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    rounded = resample_image(u8, (0.7, 0.7, 0.7), round_intensities=True).array
    assert rounded.tobytes() == resample_image(u8, (0.7, 0.7, 0.7)).array.tobytes()
    value = np.empty((1, 1, 1))
    src = np.zeros((2, 2, 3))
    src[0, 0, 0], src[0, 0, 1] = 0.5, 0.5 - 2.0**-53
    scale, shift, start = np.ones(3), np.array([0.0, 0.0, 0.5]), np.zeros(3, dtype=np.int64)
    preprocessing._resample_trilinear_numba(src, scale, shift, start, False, np.nan, True, value)
    assert value[0, 0, 0] == 0.0  # 0.5 - 2^-54 rounds down


def test_the_trilinear_kernel_keeps_a_tie_at_the_mask_threshold() -> None:
    # Two neighbours of a float64 mask, 0.5 and 0.5 - 2^-53, with the weights 0.5/0.5 give
    # 0.5 - 2^-54 exactly. The rounded threshold rule keeps that voxel (the rule of uint8
    # masks); the old float64 rule, value >= 0.5, dropped it.
    from pictologics import preprocessing

    src = np.zeros((2, 2, 3))
    src[0, 0, 0] = 0.5
    src[0, 0, 1] = 0.5 - 2.0**-53
    scale, shift, start = np.ones(3), np.array([0.0, 0.0, 0.5]), np.zeros(3, dtype=np.int64)
    value = np.empty((1, 1, 1))
    kernel = preprocessing._resample_trilinear_numba
    kernel(src, scale, shift, start, False, float("nan"), False, value)
    assert value[0, 0, 0] == 0.5 - 2.0**-54 and not value[0, 0, 0] >= 0.5
    for unsigned in (False, True):
        kept = np.empty((1, 1, 1), dtype=np.uint8)
        kernel(src, scale, shift, start, unsigned, 0.5, False, kept)
        assert kept[0, 0, 0] == 1


def test_resample_mask_threshold_every_type() -> None:
    # A mask threshold applies to the interpolated value, on the kernel paths and on
    # scipy's path (linear and cubic). Every mask type (float64, uint8, bool) rounds it at
    # the threshold, which at 0.5 is the rounding a uint8 mask always had. Before, a
    # float64 mask compared the value itself, a uint8 mask ignored the threshold and a bool
    # mask kept only voxels with every neighbour inside.
    rng = np.random.default_rng(21)
    labels = rng.random((12, 11, 10)) < 0.5
    spacing = (1.7, 1.6, 1.5)

    def resample(
        arr: np.ndarray, how: str, thr: float | None, boundary: str = "nearest"
    ) -> np.ndarray:
        img = Image(arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        return resample_image(img, spacing, how, boundary_mode=boundary, mask_threshold=thr).array

    for how in ("linear", "cubic"):
        for boundary in ("nearest", "reflect"):  # the kernels, then scipy for every type
            values = resample(labels.astype(np.float64), how, None, boundary)
            for thr in (0.3, 0.5, 0.7):
                rounded = (values + (1.0 - thr) >= 1.0).astype(np.uint8)
                for arr in (labels.astype(np.float64), labels.astype(np.uint8), labels):
                    out = resample(arr, how, thr, boundary)
                    assert out.dtype == np.uint8
                    assert_array_equal(out, rounded)
                if how == "linear" and thr == 0.5:  # the old uint8 result
                    old = resample(labels.astype(np.uint8), how, None, boundary)
                    assert_array_equal(rounded, (old >= 0.5).astype(np.uint8))
    # A nearest resample compares the value itself (a probability mask of 0.2 and 0.8)
    probability = labels * 0.6 + 0.2
    nearest = resample(probability, "nearest", None)
    assert_array_equal(resample(probability, "nearest", 0.5), (nearest >= 0.5).astype(np.uint8))


def test_resample_grid_stays_for_each_input() -> None:
    # The grid of positive spacings stays for each input: spacings as ints, floats or numpy
    # floats are one input, with one grid. Other spacings (here -0.0 and NaN) never stay.
    from pictologics.preprocessing import _grid, _kept_grid, _output_grid

    _kept_grid.cache_clear()
    shape = (9, 11, 13)
    first = _output_grid(shape, (1.3, 0.9, 2.1), (1, 1, 1))
    assert first == _grid(shape, (1.3, 0.9, 2.1), (1.0, 1.0, 1.0))
    assert _output_grid(shape, (1.3, 0.9, np.float64(2.1)), (1.0, 1.0, 1.0)) is first
    assert _kept_grid.cache_info().hits == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for spacing in ((1.3, -0.0, 2.1), (1.3, np.nan, 2.1)):
            _output_grid(shape, spacing, (1.0, 1.0, 1.0))
    assert _kept_grid.cache_info().currsize == 1


def test_resample_region_matches_the_whole_grid() -> None:
    # A region of the new grid has the values of the whole grid bit for bit: with the
    # kernels (linear, nearest, cubic, uint8 masks with a threshold, a source mask, rounding)
    # and with the cut of the whole grid (a float32 cubic). Its origin is the region's first
    # voxel.
    from pictologics.loader import _direction_matrix

    rng = np.random.default_rng(21)
    arr = rng.normal(0.0, 100.0, (9, 11, 13))
    img = Image(arr, (1.3, 0.9, 2.1), (5.0, -3.0, 1.0))
    mask = Image((rng.random(arr.shape) < 0.4).astype(np.uint8), img.spacing, img.origin)
    source = rng.random(arr.shape) < 0.9
    spacing = (0.7, 0.8, 1.1)
    depth, height, width = resample_image(img, spacing).array.shape
    region = (slice(2, depth - 1), slice(0, 5), slice(3, width))
    cases = [
        (img, {"interpolation": "linear"}),
        (img, {"interpolation": "nearest"}),
        (img, {"interpolation": "cubic"}),
        (Image(arr.astype(np.float32), img.spacing, img.origin), {"interpolation": "cubic"}),
        (img, {"interpolation": "linear", "round_intensities": True}),
        (img, {"interpolation": "linear", "source_mask": source}),
        (mask, {"interpolation": "linear", "mask_threshold": 0.5}),
        (mask, {"interpolation": "nearest"}),
    ]
    for image, kw in cases:
        whole = resample_image(image, spacing, **kw)
        part = resample_image(image, spacing, region=region, **kw)
        assert part.array.dtype == whole.array.dtype
        assert np.array_equal(
            np.ascontiguousarray(part.array).view(np.uint8),
            np.ascontiguousarray(whole.array[region]).view(np.uint8),
        )
        if "source_mask" in kw:
            assert np.array_equal(part.source_mask, whole.source_mask[region])
        step = np.array([r.start for r in region]) * np.array(spacing)
        expected = np.array(whole.origin) + _direction_matrix(whole.direction) @ step
        assert np.allclose(part.origin, expected)


def test_roi_region_holds_every_resampled_roi_voxel() -> None:
    # The region from the input ROI box holds every nonzero voxel of the resampled mask
    # (nearest, and linear with a threshold), for random grids; a margin grows it.
    from pictologics.features._utils import compute_nonzero_bbox
    from pictologics.preprocessing import _roi_region

    rng = np.random.default_rng(4)
    for _ in range(40):
        shape = tuple(int(n) for n in rng.integers(4, 14, 3))
        spacing = tuple(float(s) for s in rng.uniform(0.5, 2.5, 3))
        new_spacing = tuple(float(s) for s in rng.uniform(0.4, 2.5, 3))
        m = np.zeros(shape, dtype=np.uint8)
        lo = [int(rng.integers(0, n)) for n in shape]
        hi = [int(rng.integers(a + 1, n + 1)) for a, n in zip(lo, shape, strict=True)]
        m[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 1
        box = compute_nonzero_bbox(m)
        region = _roi_region(box, shape, spacing, new_spacing, (0, 0, 0))
        for interpolation, threshold in (("nearest", None), ("linear", 0.5)):
            out = resample_image(
                Image(m, spacing, (0.0, 0.0, 0.0)),
                new_spacing,
                interpolation=interpolation,
                mask_threshold=threshold,
            ).array
            outside = np.ones(out.shape, dtype=bool)
            outside[region] = False
            assert not out[outside].any()
        grown = _roi_region(box, shape, spacing, new_spacing, (2, 2, 2))
        assert all(
            g.start == max(r.start - 2, 0) and g.stop <= r.stop + 2
            for g, r in zip(grown, region, strict=True)
        )


def test_all_finite() -> None:
    # One parallel pass for large contiguous float64 arrays (row or column order) on enough
    # threads, numpy for small or other arrays or on too few threads; a NaN or an infinite
    # value anywhere makes it False.
    from pictologics import preprocessing
    from pictologics.preprocessing import _all_finite

    arr = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    # The parallel pass, numpy for a small array, numpy on too few threads
    for minimum, threads in ((0, 1), (1 << 21, 1), (0, 1 << 10)):
        with (
            patch.object(preprocessing, "_FINITE_PARALLEL_MIN", minimum),
            patch.object(preprocessing, "_FINITE_PARALLEL_THREADS", threads),
        ):
            # New arrays: the check keeps its result for each array (see the next test)
            for layout in (arr.copy(), np.asfortranarray(arr), arr[:, ::2], arr.astype(np.float32)):
                assert _all_finite(layout)
            for bad in (np.nan, np.inf, -np.inf):
                broken = arr.copy()
                broken[2, 3, 4] = bad
                assert not _all_finite(broken)
                assert not _all_finite(broken.astype(np.float32))


def test_all_finite_runs_the_kernel_only_on_enough_threads() -> None:
    # Below 4 threads the kernel is slower than np.sum at every size: numpy answers
    from pictologics import preprocessing

    arr = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    with (
        patch.object(preprocessing, "_FINITE_PARALLEL_MIN", 0),
        patch.object(
            preprocessing, "_nonfinite_blocks_numba", wraps=preprocessing._nonfinite_blocks_numba
        ) as kernel,
    ):
        for threads, calls in ((3, 0), (4, 1)):
            with patch.object(preprocessing, "get_num_threads", return_value=threads):
                assert preprocessing._all_finite(arr.copy())  # a new array: no kept result
            assert kernel.call_count == calls
        kernel.reset_mock()  # the mock keeps its arguments, a view of the array


def test_all_finite_gives_no_numpy_warning() -> None:
    # The numpy branch of the finite check gives no warning for a sum that overflows or
    # adds +inf to -inf; the answer stays False.
    from pictologics.preprocessing import _all_finite

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _all_finite(np.full(10, 1e308))
        assert not _all_finite(np.array([1.0, np.inf, -np.inf]))
        assert not _all_finite(np.full(10, 3e38, dtype=np.float32))


def test_all_finite_reads_each_array_once() -> None:
    # The finite check keeps its result while the array lives: a second check of the same
    # array (the next run on the same image) does not read it again; another array is read,
    # and the result goes when its array goes.
    import gc

    from pictologics import preprocessing

    arr = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    broken = arr.copy()
    broken[1, 2, 3] = np.nan
    with (
        patch.object(preprocessing, "_FINITE_PARALLEL_MIN", 0),
        patch.object(preprocessing, "_FINITE_PARALLEL_THREADS", 1),
        patch.object(
            preprocessing, "_nonfinite_blocks_numba", wraps=preprocessing._nonfinite_blocks_numba
        ) as kernel,
    ):
        assert preprocessing._all_finite(arr) and preprocessing._all_finite(arr)
        assert kernel.call_count == 1
        assert not preprocessing._all_finite(broken) and not preprocessing._all_finite(broken)
        assert kernel.call_count == 2
    key = id(broken)
    assert preprocessing._FINITE_RESULTS[key] is False
    kernel.reset_mock()  # the mock keeps its arguments, a view of the array
    del broken
    gc.collect()
    assert key not in preprocessing._FINITE_RESULTS


def _grown_by_definition(
    inside: np.ndarray, spacing: tuple[float, float, float], x: float
) -> np.ndarray:
    """The mask `inside` grown by x mm (shrunk by -x mm), voxel by voxel: the distance from
    each voxel center to the nearest center in (out of) the mask, with one layer of voxels
    outside the mask past each image edge."""
    padded = np.pad(inside, 1)
    centers = np.indices(padded.shape).reshape(3, -1).T * np.asarray(spacing)
    targets = centers[padded.ravel() == (x > 0)]
    nearest = np.sqrt(((centers[:, None, :] - targets[None]) ** 2).sum(axis=2)).min(axis=1)
    nearest = nearest.reshape(padded.shape)[1:-1, 1:-1, 1:-1]
    if x > 0:
        return inside | (nearest <= x + 1e-6)
    return inside & (nearest > -x + 1e-6) if x < 0 else inside


def test_grow_mask_matches_the_distance_definition() -> None:
    # Grow, shrink and rings of a random mask on voxels of three sizes, against the
    # distances voxel by voxel (also at the image edge, where outside voxels follow).
    rng = np.random.default_rng(3)
    inside = rng.random((6, 7, 8)) < 0.35
    inside[0, 3, 3] = inside[5, 6, 7] = True  # mask voxels at the edge
    spacing = (0.5, 0.7, 0.9)
    mask = Image(inside.astype(np.uint8) * 3, spacing, (1.0, 2.0, 3.0))
    for to_mm, from_mm in (
        (1.2, None), (0.5, None), (2.0, 0.6), (0.0, -0.8), (-0.4, None), (-0.6, -1.5),
        (1.0, -0.7), (0.0, None),
    ):  # fmt: skip
        expected = _grown_by_definition(inside, spacing, to_mm)
        if from_mm is not None:
            expected &= ~_grown_by_definition(inside, spacing, from_mm)
        grown = grow_mask(mask, to_mm, from_mm)
        assert grown.array.dtype == np.uint8
        assert_array_equal(grown.array, expected)
        assert (grown.spacing, grown.origin) == (mask.spacing, mask.origin)


def test_grow_mask_counts_whole_steps_at_the_limit() -> None:
    # 3 steps of 0.4 mm are 1.2 mm (1.2000000000000002 in floating point): a grow by
    # 1.2 mm adds 3 voxels along the 0.4 mm axes and 1 along the 0.8 mm axis.
    array = np.zeros((9, 9, 9), dtype=np.uint8)
    array[4, 4, 4] = 1
    grown = grow_mask(Image(array, (0.4, 0.4, 0.8), (0.0, 0.0, 0.0)), 1.2).array
    assert np.flatnonzero(grown[:, 4, 4]).tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert np.flatnonzero(grown[4, 4, :]).tolist() == [3, 4, 5]
    # and a shrink by 0.4 mm removes the voxels next to the outside
    cube = np.ones((5, 5, 5), dtype=np.uint8)
    core = grow_mask(Image(cube, (0.4, 0.4, 0.4), (0.0, 0.0, 0.0)), -0.4).array
    assert core.sum() == 27 and core[1:4, 1:4, 1:4].all()


def test_grow_mask_of_an_empty_mask_and_bad_distances() -> None:
    empty = Image(np.zeros((4, 4, 4), dtype=np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    assert not grow_mask(empty, 2.0).array.any()
    for to_mm, from_mm, message in (
        ("3", None, "to_mm must be a finite number"),
        (True, None, "to_mm must be a finite number"),
        (np.inf, None, "to_mm must be a finite number"),
        (2.0, np.nan, "from_mm must be a finite number"),
        (2.0, 2.0, r"from_mm \(2.0\) must be below to_mm \(2.0\)"),
    ):
        with pytest.raises(ValueError, match=message):
            grow_mask(empty, to_mm, from_mm)
    assert grow_mask(empty, np.int64(1), np.float32(0.5)).array.dtype == np.uint8


def test_squared_distances_are_the_least_sum(scipy_fuses: bool) -> None:
    # The distance passes give the least ((d0 s0)^2 + (d1 s1)^2) + (d2 s2)^2 over the True
    # voxels (the sum order of scipy), bit for bit, and inf with no True voxel. Their roots
    # are scipy's distance transform where the scipy build does not fuse a multiply-add.
    # With labels, each voxel gets the label of a True voxel at that least sum.
    from scipy.ndimage import distance_transform_edt

    from pictologics.preprocessing import _squared_distances

    rng = np.random.default_rng(31)
    shapes = ((1, 5, 7), (6, 1, 4), (9, 8, 7))
    arrays = [rng.random(shape) < density for shape in shapes for density in (0.05, 0.4)]
    plane = rng.random((5, 6, 7)) < 0.3
    plane[2] = False  # a plane with no True voxel
    arrays += [plane, np.zeros((3, 4, 5), dtype=bool)]
    for spacing in ((0.7, 0.7, 1.25), (0.33, 1.7, 0.91), (1.0, 1.0, 1.0)):
        for feature in arrays:
            points = np.argwhere(feature)
            voxels = np.indices(feature.shape).reshape(3, -1).T
            parts = [
                ((voxels[:, a, None] - points[None, :, a]) * spacing[a]) ** 2 for a in range(3)
            ]
            sums = (parts[0] + parts[1]) + parts[2]
            expected = sums.min(axis=1, initial=np.inf).reshape(feature.shape)
            lab = np.zeros(feature.shape, dtype=np.int32)
            lab[feature] = np.arange(1, len(points) + 1)
            got = _squared_distances(feature, spacing, lab)
            assert got.tobytes() == expected.tobytes()
            assert (
                _squared_distances(np.asfortranarray(feature), spacing).tobytes() == got.tobytes()
            )
            if not feature.any():
                assert not lab.any()
                continue
            nearest = sums[np.arange(len(voxels)), lab.ravel() - 1].reshape(feature.shape)
            assert nearest.tobytes() == expected.tobytes()
            # scipy's values within 4e-16 where scipy fuses; elsewhere the same values (the
            # distances are finite and not negative, so the same bits)
            scipy_distances = distance_transform_edt(~feature, sampling=spacing)
            rtol = 4e-16 if scipy_fuses else 0.0
            np.testing.assert_allclose(np.sqrt(got), scipy_distances, rtol=rtol, atol=0.0)


def test_nearest_roi_map_carries_the_nearest_label() -> None:
    # Each map voxel gets the label of an ROI with a voxel at the least plain-arithmetic
    # distance (the sum order of scipy) from it. A column-order uint16 copy gives the same
    # map, and the int32 row-order map of the caller does not change (no alias).
    from pictologics.preprocessing import _nearest_roi_map

    z, y, x = np.indices((24, 24, 24))
    labels = np.zeros((24, 24, 24), dtype=np.int32)
    for label, (cz, cy, cx) in enumerate(((8, 8, 12), (14, 9, 12), (11, 15, 12)), start=1):
        labels[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= 9] = label
    original = labels.copy()
    roi = np.argwhere(labels != 0)
    for spacing in ((0.8, 0.8, 0.8), (2.0, 0.9, 0.9)):
        nearest = _nearest_roi_map(Image(labels, spacing, (0.0, 0.0, 0.0)), 3.0)
        assert nearest is not None and nearest.array.dtype == np.int32
        assert_array_equal(labels, original)
        start = np.rint(np.asarray(nearest.origin) / spacing).astype(int)
        voxels = np.argwhere(np.ones(nearest.array.shape, dtype=bool)) + start
        parts = [((voxels[:, a, None] - roi[None, :, a]) * spacing[a]) ** 2 for a in range(3)]
        sums = (parts[0] + parts[1]) + parts[2]
        own = labels[tuple(roi.T)][None, :] == nearest.array.reshape(-1, 1)
        assert_array_equal(np.where(own, sums, np.inf).min(axis=1), sums.min(axis=1))
        column = Image(np.asfortranarray(labels.astype(np.uint16)), spacing, (0.0, 0.0, 0.0))
        again = _nearest_roi_map(column, 3.0)
        assert again is not None and again.array.tobytes() == nearest.array.tobytes()


def test_nearest_roi_map_and_part() -> None:
    # Each voxel near the ROIs gets the label of its nearest ROI voxel; the added voxels
    # of a mask on another grid (other spacing, origin and axis directions) keep the
    # label of the map voxel nearest to their centers.
    from pictologics.preprocessing import _nearest_roi_map, _nearest_roi_part

    assert _nearest_roi_map(Image(np.zeros((3, 3, 3)), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), 1) is None
    labels = np.zeros((16, 16, 16), dtype=np.int64)
    labels[6:9, 5:8, 6:10] = 1
    labels[6:9, 8:11, 6:10] = 2
    direction = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    label_map = Image(labels, (0.5, 0.5, 0.5), (4.0, -1.0, 2.0), direction=direction)
    nearest = _nearest_roi_map(label_map, 1.0)
    assert nearest is not None and nearest.array.shape == (11, 14, 12)  # 1 mm and 2 voxels more

    def world(image: Image, index: np.ndarray) -> np.ndarray:
        axes = np.asarray(image.direction, dtype=float).reshape(3, 3) * np.asarray(image.spacing)
        return np.asarray(image.origin) + index @ axes.T

    roi = np.argwhere(labels != 0)
    map_index = np.argwhere(np.ones(nearest.array.shape, dtype=bool))
    gaps = np.linalg.norm(world(nearest, map_index)[:, None] - world(label_map, roi)[None], axis=2)
    closest = labels[tuple(roi[gaps.argmin(axis=1)].T)]
    assert_array_equal(nearest.array.ravel(), closest)
    # A grid of 0.7 mm in other axis directions: ROI 1 grown by 1.5 mm
    grid = Image(
        np.zeros((12, 12, 12), np.uint8), (0.7, 0.7, 0.7), (0.5, -6.5, 2.0), direction=np.eye(3)
    )
    centers = world(grid, np.argwhere(np.ones((12, 12, 12), dtype=bool)))
    to_roi1 = np.linalg.norm(
        centers[:, None] - world(label_map, np.argwhere(labels == 1))[None], axis=2
    )
    mask = Image(
        (to_roi1.min(axis=1) < 0.5).reshape(12, 12, 12).astype(np.uint8),
        grid.spacing,
        grid.origin,
        direction=grid.direction,
    )
    grown = grow_mask(mask, 1.5)
    added = np.argwhere((grown.array != 0) & (mask.array == 0))
    map_centers = world(nearest, map_index)
    owner = nearest.array.ravel()[
        np.linalg.norm(world(grid, added)[:, None] - map_centers[None], axis=2).argmin(axis=1)
    ]
    assert 0 < (owner == 1).sum() < len(added)  # both ROIs own some of the added voxels
    kept = _nearest_roi_part(grow_mask(mask, 1.5), mask, nearest, 1).array
    assert_array_equal(kept[tuple(added.T)], (owner == 1).astype(np.uint8))
    assert (kept[mask.array != 0] == 1).all()


def test_normalise_image_by_its_region() -> None:
    # z-score and percentile maps from the statistics of a mask, the whole image or a
    # value range (NaN voxels left out); the map changes every voxel.
    rng = np.random.default_rng(2)
    values = rng.normal(300.0, 40.0, (8, 9, 10))
    values[0, 0, 0] = np.nan
    image = Image(values, (1.0, 1.0, 2.0), (0.0, 0.0, 0.0))
    inside = np.zeros(values.shape, dtype=np.uint8)
    inside[2:6, 2:6, 2:6] = 5
    mask = Image(inside, image.spacing, image.origin)
    finite = values[np.isfinite(values)]
    z = normalise_image(image, "zscore")
    assert np.isnan(z.array[0, 0, 0]) and z.spacing == image.spacing
    assert np.allclose(z.array[1:], ((values - finite.mean()) / finite.std())[1:])
    roi = normalise_image(image, "zscore", mask=mask).array[inside != 0]
    assert abs(roi.mean()) < 1e-12 and abs(roi.std() - 1.0) < 1e-12
    low, high = np.percentile(finite, (5.0, 95.0))
    ranged = normalise_image(image, "percentile", percentiles=(5.0, 95.0))
    assert np.allclose(ranged.array[1:], ((values - low) / (high - low))[1:])
    middle = finite[(finite >= 280.0) & (finite <= 320.0)]
    banded = normalise_image(
        image, "percentile", percentiles=(0, 100), range_min=280.0, range_max=320.0
    )
    assert np.allclose(banded.array[1:], ((values - middle.min()) / np.ptp(middle))[1:])
    for kwargs, message in (
        ({"method": "minmax"}, "method must be one of"),
        ({"method": "percentile", "percentiles": (50, 50)}, "percentiles must be two numbers"),
        ({"method": "percentile", "percentiles": [1]}, "percentiles must be two numbers"),
        ({"method": "zscore", "range_min": "0"}, "range_min must be a number"),
        (
            {"method": "zscore", "range_min": 5.0, "range_max": 1.0},
            r"range_min \(5.0\) must not be above",
        ),
        ({"method": "zscore", "range_min": 1e9}, "holds no voxel"),
    ):
        with pytest.raises(ValueError, match=message):
            normalise_image(image, **kwargs)
    with pytest.raises(ValueError, match="do not spread"):
        normalise_image(Image(np.full((3, 3, 3), 7.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), "zscore")
    with pytest.raises(ValueError, match="Dimension mismatch between mask"):
        normalise_image(image, "zscore", mask=Image(inside[:4], image.spacing, image.origin))


def test_normalise_image_reads_a_finite_row_order_image_with_no_copy() -> None:
    # A whole finite image in row order gives its statistics with no gathered copy, and the
    # map runs in one pass in the order of the image (the kernel for float64 in row or
    # column order). The values and the layout are those of the numpy code.
    from pictologics import preprocessing

    values = np.random.default_rng(14).normal(300.0, 40.0, (9, 10, 11))
    with_nan = values.copy()
    with_nan[4, 5, 6] = np.nan
    inputs = (
        values,
        np.asfortranarray(values),
        values[:, ::2],
        values.astype(np.float32),
        np.round(values).astype(np.int16),
        with_nan,
    )
    for arr in inputs:
        for method in ("zscore", "percentile"):
            kept = arr[np.isfinite(arr)]
            if method == "zscore":
                center, scale = float(kept.mean()), float(kept.std())
            else:
                low, high = np.percentile(kept, (1.0, 99.0))
                center, scale = float(low), float(high - low)
            ref = np.subtract(arr, center, dtype=np.float64)
            ref /= scale
            for gate in (0, 1 << 30):
                with (
                    patch.object(preprocessing, "_NORMALISE_PARALLEL_MIN", gate),
                    patch.object(
                        preprocessing, "_all_finite", wraps=preprocessing._all_finite
                    ) as spy,
                ):
                    out = normalise_image(Image(arr, (1, 1, 1), (0, 0, 0)), method).array
                assert out.tobytes() == ref.tobytes()
                assert out.flags.f_contiguous == ref.flags.f_contiguous
                assert spy.call_count == int(arr.flags.c_contiguous)


def test_the_resample_kernels_take_their_serial_twins_below_their_gates() -> None:
    # A small output takes the serial twin of each resample kernel; from the gate on, with
    # more than one thread, the kernel runs. Both give the same arrays.
    from contextlib import ExitStack

    from pictologics import preprocessing as module

    names = [
        "_resample_trilinear_numba",
        "_resample_nearest_numba",
        "_resample_trilinear_masked_numba",
        "_resample_tricubic_numba",
        "_pad_edge_numba",
        "_prefilter_axis0_numba",
        "_prefilter_axis1_numba",
        "_prefilter_axis2_numba",
        "_resample_tricubic_masked_numba",
        "_pad_edge_masked_numba",
    ]
    gates = [
        "_LINEAR_PARALLEL_MIN",
        "_NEAREST_PARALLEL_MIN",
        "_MASKED_PARALLEL_MIN",
        "_CUBIC_PARALLEL_MIN",
        "_PREFILTER_PARALLEL_MIN",
    ]
    rng = np.random.default_rng(5)
    image = Image(rng.normal(40.0, 10.0, (20, 21, 22)), (0.7, 0.7, 1.25), (0.0, 0.0, 0.0))
    mask = Image(
        (rng.random(image.array.shape) > 0.5).astype(np.uint8), image.spacing, image.origin
    )
    valid = rng.random(image.array.shape) > 0.1

    def run(gate: int) -> tuple[list[np.ndarray], dict[str, int]]:
        with ExitStack() as stack:
            spies = {
                name: stack.enter_context(patch.object(module, name, wraps=getattr(module, name)))
                for base in names
                for name in (base, base + "_serial")
            }
            for name in gates:
                stack.enter_context(patch.object(module, name, getattr(module, name) * gate))
            out = [
                resample_image(image, (1.0, 1.0, 1.0)).array,
                resample_image(mask, (1.0, 1.0, 1.0), interpolation="nearest").array,
                resample_image(image, (1.0, 1.0, 1.0), source_mask=valid).array,
                resample_image(image, (1.0, 1.0, 1.0), interpolation="cubic").array,
                resample_image(image, (1.0, 1.0, 1.0), "cubic", source_mask=valid).array,
            ]
        return out, {name: spy.call_count for name, spy in spies.items()}

    # The prefilter runs on the plain coefficients and on the two masked ones
    expected = {name: 3 if name.startswith("_prefilter") else 1 for name in names}
    small, calls = run(1 << 10)  # every gate far above the work
    assert all(calls[name] == 0 and calls[name + "_serial"] == expected[name] for name in names)
    large, calls = run(0)  # every gate at 0: the parallel kernels
    assert all(calls[name] == expected[name] and calls[name + "_serial"] == 0 for name in names)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(small, large, strict=True))
