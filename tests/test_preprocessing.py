from __future__ import annotations

# ruff: noqa: E402
import os
import warnings

# Suppress "NumPy module was reloaded" warning which can happen in test setups
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

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
    mask = Image(np.ones((4, 4, 4), dtype=np.uint8), (1, 1, 1), (0, 0, 0))
    for limit in (8, 1 << 30):
        with patch("pictologics.preprocessing._RESEGMENT_KERNEL_MIN_SIZE", limit):
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
    # A non-3D mask takes the full-array path (no bounding-box crop).
    out = keep_largest_component(Image(np.ones((4, 4), dtype=np.uint8), (1.0, 1.0), (0.0, 0.0)))
    assert out.array.shape == (4, 4)


def test_extract_roi_integer_image() -> None:
    # An integer image is upcast to float (NaN-capable) before masking.
    img = Image(np.arange(27, dtype=np.int32).reshape(3, 3, 3), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    mask = Image(np.ones((3, 3, 3), dtype=np.uint8), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    assert extract_roi(img, mask).array.dtype == np.float64


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
# _DISCRETISE_KERNEL_MIN_SIZE, or _KERNEL_MIN_SIZE for arrays that share no memory
# order). The limits are patched small so tiny arrays exercise the single-pass
# kernels; the optimization work proved these bit-identical to the numpy fallback.
# ---------------------------------------------------------------------------


def _f64(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)


def test_discretise_fbn_kernel_clamps() -> None:
    arr = _f64((4, 4, 4))
    arr[0, 0, 0] = np.nan  # NaN maps to bin 0
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    # Explicit range narrower than the data: values below it clamp to bin 1,
    # values above clamp to n_bins.
    with patch("pictologics.preprocessing._DISCRETISE_KERNEL_MIN_SIZE", 8):
        out = discretise_image(img, method="FBN", n_bins=8, min_val=20.0, max_val=40.0)
    assert out.array[0, 0, 0] == 0
    assert out.array.min() >= 0
    assert out.array.max() <= 8


def test_discretise_fbs_kernel_clamps() -> None:
    # NaN and +inf map to bin 0 (no cast of infinity to an integer), -inf to bin 1, on
    # both paths. The default minimum is the smallest finite ROI value.
    arr = _f64((4, 4, 4))
    arr[0, 0, :3] = (np.nan, np.inf, -np.inf)
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    for limit in (8, 1 << 30):
        with patch("pictologics.preprocessing._DISCRETISE_KERNEL_MIN_SIZE", limit):
            out = discretise_image(img, method="FBS", bin_width=5.0, min_val=20.0, max_val=40.0)
            assert out.array[0, 0, :3].tolist() == [0, 0, 1]
            out = discretise_image(img, method="FBS", bin_width=5.0, roi_mask=np.ones((4, 4, 4)))
            assert out.array[0, 0, 3] == 1 and out.array[3, 3, 3] == 13  # (63 - 3) / 5 + 1


def test_discretise_kernel_keeps_array_order() -> None:
    # The kernels give the numpy chain's bins and layout, for row- and column-order input.
    # The DICOM loader's layout (neither) needs a copy: below _KERNEL_MIN_SIZE it keeps
    # the numpy chain.
    rng = np.random.default_rng(5)
    row_order = rng.normal(0.0, 50.0, (5, 6, 7))
    row_order[1, 2, 3] = np.nan
    dicom_layout = np.swapaxes(row_order, 0, 1)
    for arr in (row_order, np.asfortranarray(row_order), dicom_layout):
        for method, kw in (("FBN", {"n_bins": 8}), ("FBS", {"bin_width": 10.0})):
            ref = discretise_image(arr, method, **kw)
            with (
                patch("pictologics.preprocessing._DISCRETISE_KERNEL_MIN_SIZE", 8),
                patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8),
            ):
                out = discretise_image(arr, method, **kw)
            assert_array_equal(out, ref)
            assert out.flags.f_contiguous == ref.flags.f_contiguous == arr.flags.f_contiguous
    with (
        patch("pictologics.preprocessing._DISCRETISE_KERNEL_MIN_SIZE", 8),
        patch("pictologics.preprocessing._discretise_fbs_numba", side_effect=AssertionError),
    ):
        discretise_image(dicom_layout, "FBS", bin_width=10.0)


def test_discretise_cutoffs_kernel_matches_digitize() -> None:
    # The cutoff kernel gives the bins of digitize (+1, NaN to 0) for increasing,
    # decreasing, repeated and no cutoffs, in row and column order. Cutoffs that go up
    # and down keep the error of digitize.
    rng = np.random.default_rng(6)
    arr = rng.normal(0.0, 50.0, (5, 6, 7))
    arr[1, 2, 3] = np.nan
    arr[0, 0, :3] = (-20.0, 0.0, 20.0)  # values on the cutoffs
    for cutoffs in ([-20.0, 0.0, 20.0], [20.0, 0.0, -20.0], [0.0, 0.0, 20.0], []):
        expected = np.digitize(arr, cutoffs) + 1
        expected[np.isnan(arr)] = 0
        for layout in (arr, np.asfortranarray(arr)):
            with patch("pictologics.preprocessing._DISCRETISE_KERNEL_MIN_SIZE", 8):
                out = discretise_image(layout, "FIXED_CUTOFFS", cutoffs=cutoffs)
            assert out.dtype == np.int32
            assert_array_equal(out, expected)
    with (
        patch("pictologics.preprocessing._DISCRETISE_KERNEL_MIN_SIZE", 8),
        pytest.raises(ValueError, match="monotonically"),
    ):
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
    with patch("pictologics.preprocessing._NEAREST_KERNEL_MIN_SIZE", 1):
        assert resample_image(img, (0.7, 0.7, 0.7), interpolation="nearest").array.ndim == 3
        assert resample_image(img, (1.3, 1.3, 1.3), interpolation="nearest").array.ndim == 3


def test_resample_linear_kernel_and_all_valid_source() -> None:
    img = Image(np.random.rand(8, 8, 8).astype(np.float64), (1, 1, 1), (0, 0, 0))
    with patch("pictologics.preprocessing._LINEAR_KERNEL_MIN_SIZE", 1):
        resample_image(img, (0.7, 0.7, 0.7), interpolation="linear")
        # An all-valid source mask collapses to "no mask", but the resampled image
        # still carries an all-valid source mask.
        all_valid = img.with_source_mask(
            Image(np.ones((8, 8, 8), np.uint8), img.spacing, img.origin)
        )
        r = resample_image(all_valid, (0.7, 0.7, 0.7), interpolation="linear")
    assert r.source_mask is not None
    assert bool(r.source_mask.all())


def test_resample_kernels_match_scipy() -> None:
    # The kernels give scipy's output bit for bit: -0.0 values, rounding, mask
    # thresholds, uint8 input, the source-mask path, and nearest mode. Small outputs
    # go to scipy itself; gates of 1 force the kernels.
    from unittest.mock import patch as patch_attrs

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
    for gate in (1, 1 << 20):
        gates = {f"_{name}_KERNEL_MIN_SIZE": gate for name in ("LINEAR", "NEAREST", "MASKED")}
        with patch_attrs.multiple("pictologics.preprocessing", **gates):
            assert same(resample_image(img, new_spacing).array, scipy_out(image))
            rounded = resample_image(img, new_spacing, round_intensities=True).array
            assert same(rounded, np.round(scipy_out(image)))
            for arr in (mask, mask.astype(np.float64)):
                out = resample_image(
                    Image(arr, spacing, (0, 0, 0)), new_spacing, mask_threshold=0.5
                )
                assert same(out.array, (scipy_out(arr) >= 0.5).astype(np.uint8))
            uint8_out = resample_image(Image(uint8_img, spacing, (0, 0, 0)), new_spacing).array
            assert same(uint8_out, scipy_out(uint8_img))
            nearest = resample_image(img, new_spacing, interpolation="nearest").array
            assert same(nearest, scipy_out(image, order=0))
            out = resample_image(masked, new_spacing)
            assert same(out.array, ref)
            assert_array_equal(out.source_mask, ref_valid)


def test_resample_masked_linear_kernel() -> None:
    # A partial source mask + float64 + linear routes to the fused masked kernel.
    img = Image(np.random.rand(8, 8, 8).astype(np.float64), (1, 1, 1), (0, 0, 0))
    src = np.ones((8, 8, 8), dtype=np.uint8)
    src[3:5, 3:5, 3:5] = 0
    masked = img.with_source_mask(Image(src, img.spacing, img.origin))
    with patch("pictologics.preprocessing._MASKED_KERNEL_MIN_SIZE", 1):
        r = resample_image(masked, (0.7, 0.7, 0.7), interpolation="linear")
    assert r.array.ndim == 3
    assert r.source_mask is not None


def test_filter_outliers_kernels_match_numpy() -> None:
    # The two parallel passes give the numpy path's ROI values (same order) and mask,
    # bit for bit: label values, NaN intensities, and -0.0 outside the ROI.
    from pictologics.preprocessing import _roi_values_numba

    rng = np.random.default_rng(10)
    image = rng.normal(0.0, 10.0, (6, 7, 8))
    image[0, :3, 0] = np.nan
    image[2, 3, 4] = 80.0  # an outlier
    labels = np.where(rng.random(image.shape) < 0.6, 2.0, -0.0)
    img = Image(image, (1, 1, 1), (0, 0, 0))
    for mask_arr in (labels, (labels != 0).astype(np.uint8), labels != 0):
        np.testing.assert_array_equal(
            _roi_values_numba(image.ravel(), mask_arr.ravel()), image[mask_arr != 0]
        )
        mask = Image(mask_arr, (1, 1, 1), (0, 0, 0))
        ref = filter_outliers(img, mask, 2.0).array
        with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
            out = filter_outliers(img, mask, 2.0).array
        assert out.dtype == ref.dtype
        assert np.array_equal(out.view(np.uint8), ref.view(np.uint8))
        assert int(np.count_nonzero(out)) < int(np.count_nonzero(mask_arr))


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
    with patch("pictologics.preprocessing._RESEGMENT_KERNEL_MIN_SIZE", 8):
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
            with patch.multiple(
                "pictologics.preprocessing", _RESEGMENT_KERNEL_MIN_SIZE=8, _KERNEL_MIN_SIZE=8
            ):
                out = resegment_mask(image_of(img_arr), image_of(m_arr), -40.0, 60.0).array
            assert out.dtype == ref.dtype
            bits = np.ascontiguousarray(out).view(np.uint8)
            assert np.array_equal(bits, np.ascontiguousarray(ref).view(np.uint8))
            both_f = img_arr.flags.f_contiguous and m_arr.flags.f_contiguous
            assert out.flags.f_contiguous == both_f


def test_resample_mask_threshold_every_type() -> None:
    # A mask threshold applies to the interpolated value, on the kernel path and on
    # scipy's path (linear and cubic). A float64 mask compares the value itself; a uint8
    # or bool mask rounds it at the threshold, which at 0.5 is the rounding a uint8 mask
    # always had. Before, a uint8 mask ignored the threshold and a bool mask kept only
    # voxels with every neighbour inside.
    rng = np.random.default_rng(21)
    labels = rng.random((12, 11, 10)) < 0.5
    spacing = (1.7, 1.6, 1.5)

    def resample(arr: np.ndarray, how: str, thr: float | None) -> np.ndarray:
        img = Image(arr, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        return resample_image(img, spacing, how, mask_threshold=thr).array

    for how in ("linear", "cubic"):
        for limit in (1, 1 << 30):  # the kernel path (linear only), then scipy's path
            with patch("pictologics.preprocessing._LINEAR_KERNEL_MIN_SIZE", limit):
                values = resample(labels.astype(np.float64), how, None)
                for thr in (0.3, 0.5, 0.7):
                    ref = resample(labels.astype(np.float64), how, thr)
                    assert_array_equal(ref, (values >= thr).astype(np.uint8))
                    rounded = (values + (1.0 - thr) >= 1.0).astype(np.uint8)
                    for arr in (labels.astype(np.uint8), labels):
                        out = resample(arr, how, thr)
                        assert out.dtype == np.uint8
                        assert_array_equal(out, rounded)
                    if how == "linear" and thr == 0.5:  # the old uint8 result
                        old = resample(labels.astype(np.uint8), how, None)
                        assert_array_equal(rounded, (old >= 0.5).astype(np.uint8))


def test_resample_region_matches_the_whole_grid() -> None:
    # A region of the new grid has the values of the whole grid bit for bit: with the
    # kernels (linear, nearest, uint8 masks with a threshold, a source mask, rounding) and
    # with the cut of the whole grid (cubic). Its origin is the region's first voxel.
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
    # One parallel pass for large contiguous float64 arrays (row or column order), numpy
    # for small or other arrays; a NaN or an infinite value anywhere makes it False.
    from pictologics.preprocessing import _all_finite

    arr = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    for minimum in (0, 1 << 19):  # the parallel pass, then numpy for a small array
        with patch("pictologics.preprocessing._FINITE_PARALLEL_MIN", minimum):
            for layout in (arr, np.asfortranarray(arr), arr[:, ::2], arr.astype(np.float32)):
                assert _all_finite(layout)
            for bad in (np.nan, np.inf, -np.inf):
                broken = arr.copy()
                broken[2, 3, 4] = bad
                assert not _all_finite(broken)
                assert not _all_finite(broken.astype(np.float32))


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
