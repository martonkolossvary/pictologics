from __future__ import annotations

import os
import warnings

# Suppress "NumPy module was reloaded" warning which can happen in test setups
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import pytest
import numpy as np

from pictologics.loader import Image
from pictologics.preprocessing import (
    resample_image,
    discretise_image,
    apply_mask,
    extract_roi,
    resegment_mask,
    filter_outliers,
    round_intensities,
    keep_largest_component,
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
    disc = discretise_image(
        mock_image, method="FBN", n_bins=5, min_val=0.0, max_val=10.0
    )
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
    # Explicitly pass None for mask_values (defaults to 1)
    values = apply_mask(mock_image, mock_mask, mask_values=None)
    assert values.size == 27


def test_apply_mask_errors(mock_image: Image) -> None:
    # Shape mismatch
    with pytest.raises(ValueError):
        apply_mask(mock_image, np.zeros((2, 2, 2)))

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


def test_extract_roi_none_values(mock_image: Image, mock_mask: Image) -> None:
    roi_img = extract_roi(mock_image, mock_mask, mask_values=None)
    assert roi_img.array[2, 2, 2] == mock_image.array[2, 2, 2]


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
    assert filtered.array.dtype == np.uint8


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
    empty = Image(
        np.zeros(mock_image.array.shape), mock_image.spacing, mock_image.origin
    )
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
