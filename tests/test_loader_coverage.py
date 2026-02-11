import numpy as np
import pytest

from pictologics.loader import Image


class TestLoaderCoverage:
    def test_with_source_mask_validation(self):
        shape = (10, 10, 10)
        arr = np.zeros(shape, dtype=np.float32)
        img = Image(arr, (1, 1, 1), (0, 0, 0))

        # logical_not mask of mismatched shape
        bad_mask = np.zeros((5, 5, 5), dtype=bool)

        with pytest.raises(ValueError, match="Source mask shape"):
            img.with_source_mask(bad_mask)

        # Valid mask
        good_mask = np.zeros(shape, dtype=bool)
        img_masked = img.with_source_mask(good_mask)
        assert img_masked.has_source_mask

        # Valid mask from image
        mask_img = Image(np.zeros(shape, dtype=np.uint8), (1, 1, 1), (0, 0, 0))
        img_masked_img = img.with_source_mask(mask_img)
        assert img_masked_img.has_source_mask

    def test_with_source_mask_int_array(self):
        # Allow int/float arrays as bool mask (non-zero -> True)
        shape = (5, 5, 5)
        arr = np.zeros(shape, dtype=np.float32)
        img = Image(arr, (1, 1, 1), (0, 0, 0))

        int_mask = np.zeros(shape, dtype=np.int32)
        int_mask[2, 2, 2] = 1

        img_masked = img.with_source_mask(int_mask)
        # Check that it converted correctly
        assert img_masked.source_mask[2, 2, 2]
        assert not img_masked.source_mask[0, 0, 0]
