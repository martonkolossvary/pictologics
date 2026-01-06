import unittest
import numpy as np
from pictologics.features._utils import (
    compute_nonzero_bbox,
    crop_arrays_to_bbox,
    BBoxInfo,
)


class TestInternalUtils(unittest.TestCase):
    def test_crop_arrays_to_bbox(self):
        # Create data 5x5x5
        shape = (5, 5, 5)
        mask = np.zeros(shape, dtype=int)
        # ROI from (1,1,1) to (3,3,3) inclusive
        mask[1:4, 1:4, 1:4] = 1

        data1 = np.ones(shape, dtype=int) * 10
        data2 = np.ones(shape, dtype=float) * 0.5

        cropped, info = crop_arrays_to_bbox(data1, data2, mask=mask)

        self.assertIsNotNone(info)
        self.assertIsInstance(info, BBoxInfo)

        c1, c2 = cropped
        # Expected shape (3,3,3)
        self.assertEqual(c1.shape, (3, 3, 3))
        self.assertEqual(c2.shape, (3, 3, 3))

        # Check values
        self.assertTrue(np.all(c1 == 10))
        self.assertTrue(np.all(c2 == 0.5))

        # Check info
        # Slices should be 1:4
        self.assertEqual(
            info.slices, (slice(1, 4, None), slice(1, 4, None), slice(1, 4, None))
        )
        self.assertEqual(info.origin_offset, (1, 1, 1))

    def test_crop_arrays_empty_mask(self):
        shape = (5, 5, 5)
        mask = np.zeros(shape, dtype=int)
        data = np.zeros(shape)

        cropped, info = crop_arrays_to_bbox(data, mask=mask)

        self.assertEqual(cropped, (data,))
        self.assertIsNone(info)
