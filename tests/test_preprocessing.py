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
# Numba kernel paths (float64, size >= _KERNEL_MIN_SIZE). _KERNEL_MIN_SIZE is
# patched small so tiny arrays exercise the single-pass kernels; the optimization
# work proved these bit-identical to the numpy fallback.
# ---------------------------------------------------------------------------


def _f64(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)


def test_discretise_fbn_kernel_clamps() -> None:
    arr = _f64((4, 4, 4))
    arr[0, 0, 0] = np.nan  # NaN maps to bin 0
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    # Explicit range narrower than the data: values below it clamp to bin 1,
    # values above clamp to n_bins.
    with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
        out = discretise_image(img, method="FBN", n_bins=8, min_val=20.0, max_val=40.0)
    assert out.array[0, 0, 0] == 0
    assert out.array.min() >= 0
    assert out.array.max() <= 8


def test_discretise_fbs_kernel_clamps() -> None:
    arr = _f64((4, 4, 4))
    arr[0, 0, 0] = np.nan
    img = Image(arr, (1, 1, 1), (0, 0, 0))
    with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
        out = discretise_image(img, method="FBS", bin_width=5.0, min_val=20.0, max_val=40.0)
    assert out.array[0, 0, 0] == 0


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
    with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
        assert resample_image(img, (0.7, 0.7, 0.7), interpolation="nearest").array.ndim == 3
        assert resample_image(img, (1.3, 1.3, 1.3), interpolation="nearest").array.ndim == 3


def test_resample_linear_kernel_and_all_valid_source() -> None:
    img = Image(np.random.rand(8, 8, 8).astype(np.float64), (1, 1, 1), (0, 0, 0))
    with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
        resample_image(img, (0.7, 0.7, 0.7), interpolation="linear")
        # An all-valid source mask collapses to "no mask", but the resampled image
        # still carries an all-valid source mask.
        all_valid = img.with_source_mask(
            Image(np.ones((8, 8, 8), np.uint8), img.spacing, img.origin)
        )
        r = resample_image(all_valid, (0.7, 0.7, 0.7), interpolation="linear")
    assert r.source_mask is not None
    assert bool(r.source_mask.all())


def test_resample_masked_linear_kernel() -> None:
    # A partial source mask + float64 + linear routes to the fused masked kernel.
    img = Image(np.random.rand(8, 8, 8).astype(np.float64), (1, 1, 1), (0, 0, 0))
    src = np.ones((8, 8, 8), dtype=np.uint8)
    src[3:5, 3:5, 3:5] = 0
    masked = img.with_source_mask(Image(src, img.spacing, img.origin))
    with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
        r = resample_image(masked, (0.7, 0.7, 0.7), interpolation="linear")
    assert r.array.ndim == 3
    assert r.source_mask is not None


def test_resegment_kernel() -> None:
    img = Image(_f64((4, 4, 4)), (1, 1, 1), (0, 0, 0))
    mask = Image(np.ones((4, 4, 4), dtype=np.uint8), (1, 1, 1), (0, 0, 0))
    with patch("pictologics.preprocessing._KERNEL_MIN_SIZE", 8):
        out = resegment_mask(img, mask, range_min=5.0, range_max=50.0)
    assert out.array.shape == (4, 4, 4)
