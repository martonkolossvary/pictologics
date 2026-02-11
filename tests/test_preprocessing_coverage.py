import numpy as np
from numpy.testing import assert_array_equal

from pictologics.loader import Image
from pictologics.preprocessing import (
    create_source_mask_from_sentinel,
    detect_sentinel_value,
    resample_image,
)


class TestPreprocessingCoverage:
    def test_detect_sentinel_value_basic(self):
        # Create image with background -2048
        shape = (10, 10, 10)
        arr = np.full(shape, -2048.0, dtype=np.float32)
        # Add some "tissue"
        arr[2:8, 2:8, 2:8] = 100.0

        img = Image(arr, (1, 1, 1), (0, 0, 0))

        # Should detect -2048
        sentinel = detect_sentinel_value(img)
        assert sentinel == -2048.0

    def test_detect_sentinel_value_none(self):
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        img = Image(arr, (1, 1, 1), (0, 0, 0))
        # 0.0 is a common sentinel, but if it's mostly 0.0 it should be detected?
        # Default candidate values are (-2048.0, -1024.0, -1000.0, 0.0, -32768.0)
        # So 0.0 will be detected if > 1%.
        sentinel = detect_sentinel_value(img)
        assert sentinel == 0.0

        # Test with random noise (no sentinel)
        arr_noise = np.random.rand(10, 10, 10).astype(np.float32) + 100.0
        img_noise = Image(arr_noise, (1, 1, 1), (0, 0, 0))
        sentinel = detect_sentinel_value(img_noise)
        assert sentinel is None

    def test_detect_sentinel_with_roi(self):
        # Sentinel -1024
        shape = (20, 20, 20)
        arr = np.full(shape, -1024.0, dtype=np.float32)

        # ROI in center
        roi = np.zeros(shape, dtype=np.uint8)
        roi[5:15, 5:15, 5:15] = 1
        roi_img = Image(roi, (1, 1, 1), (0, 0, 0))

        # "Tissue" inside ROI
        arr[5:15, 5:15, 5:15] = 50.0

        img = Image(arr, (1, 1, 1), (0, 0, 0))

        # Sentinel is outside ROI -> Should detect
        assert detect_sentinel_value(img, roi_mask=roi_img) == -1024.0

        # CASE: Sentinel value IS the tissue value inside ROI?
        # If ROI is filled with -1024, and outside is 0.
        # This shouldn't happen for -1024 (air), but logic check:
        # Sentinel candidate must be primarily OUTSIDE ROI.

        arr2 = np.zeros(shape, dtype=np.float32)
        arr2[5:15, 5:15, 5:15] = -1024.0  # Sentinel inside
        img2 = Image(arr2, (1, 1, 1), (0, 0, 0))

        # Should NOT detect -1024 as sentinel because it's inside ROI
        # (Assuming 0 is not candidate or we only check -1024)
        val = detect_sentinel_value(img2, candidate_values=(-1024.0,), roi_mask=roi_img)
        assert val is None

    def test_create_source_mask_from_sentinel(self):
        arr = np.array([-2048.0, 100.0, -2048.0], dtype=np.float32)
        img = Image(arr, (1, 1, 1), (0, 0, 0))

        mask = create_source_mask_from_sentinel(img, -2048.0)
        assert mask.modality == "SOURCE_MASK"
        # 0 where sentinel, 1 where valid
        assert_array_equal(mask.array, [0, 1, 0])

        # With tolerance
        arr_tol = np.array([-2048.1, -2047.9, 100.0], dtype=np.float32)
        img_tol = Image(arr_tol, (1, 1, 1), (0, 0, 0))
        mask_tol = create_source_mask_from_sentinel(img_tol, -2048.0, tolerance=0.5)
        # Both close to -2048 should be 0
        assert_array_equal(mask_tol.array, [0, 0, 1])

    def test_resample_with_source_mask_coverage(self):
        # This tests the branch in resample_image that calls _resample_with_source_mask

        # 1D-like 3D image: [10, sentinel, 30]
        # shape = (3, 1, 1)
        arr = np.array([[[10.0]], [[-1000.0]], [[30.0]]], dtype=np.float32)
        img = Image(arr, (1.0, 1.0, 1.0), (0, 0, 0))

        source_mask = np.array([[[1]], [[0]], [[1]]], dtype=np.uint8)  # center invalid
        # Image constructor requires origin
        img_masked = img.with_source_mask(Image(source_mask, img.spacing, img.origin))

        # Resample to same grid (identity) but with interpolation potentially?
        # Or resample to finer grid to force interpolation

        resampled = resample_image(
            img_masked,
            new_spacing=(0.5, 1.0, 1.0),  # 2x upsampling in x
            interpolation="linear",
        )

        # The center value (originally -1000) should be interpolated from neighbors 10 and 30?
        # Or if we hit exactly the center voxel, does it stay 0/NaN?
        # Normalized convolution fills gaps!
        # So we expect values between 10 and 30, NOT close to -1000.

        data = resampled.array.flatten()
        assert np.all(data > 0)  # No negative sentinel
        assert np.all(data < 40)
