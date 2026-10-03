# ruff: noqa: E402
import warnings

# Suppress "NumPy module was reloaded" warning which can happen in test setups
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import os

import numpy as np
import pytest

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

from pictologics.loader import (
    Image,
    _direction_matrix,
    _ensure_3d,
    _find_best_dicom_series_dir,
    _load_dicom_file,
    _load_dicom_series,
    _load_nifti,
    _row_order,
    create_full_mask,
    load_and_merge_images,
    load_image,
)


def _image_mock() -> MagicMock:
    """A mock DICOM image dataset: `in` finds its pixel data, as for a real image."""
    dataset = MagicMock()
    dataset.__contains__.return_value = True
    return dataset


class TestLoader(unittest.TestCase):
    # --- _ensure_3d Tests ---
    def test_ensure_3d_2d(self) -> None:
        arr = np.zeros((10, 10))
        res = _ensure_3d(arr)
        self.assertEqual(res.shape, (10, 10, 1))

    def test_ensure_3d_3d(self) -> None:
        arr = np.zeros((10, 10, 5))
        res = _ensure_3d(arr)
        self.assertEqual(res.shape, (10, 10, 5))

    def test_ensure_3d_4d_valid(self) -> None:
        arr = np.zeros((10, 10, 5, 3))
        res = _ensure_3d(arr, dataset_index=1)
        self.assertEqual(res.shape, (10, 10, 5))

    def test_ensure_3d_4d_invalid_index(self) -> None:
        arr = np.zeros((10, 10, 5, 3))
        with self.assertRaises(ValueError):
            _ensure_3d(arr, dataset_index=5)
        with self.assertRaises(ValueError):
            _ensure_3d(arr, dataset_index=-1)

    def test_ensure_3d_invalid_dims(self) -> None:
        arr = np.zeros((10,))
        with self.assertRaises(ValueError):
            _ensure_3d(arr)

        arr = np.zeros((10, 10, 5, 3, 2))
        with self.assertRaises(ValueError):
            _ensure_3d(arr)

    # --- create_full_mask Tests ---
    def test_create_full_mask(self) -> None:
        ref_img = Image(
            array=np.zeros((10, 10, 5)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        mask = create_full_mask(ref_img)
        self.assertEqual(mask.array.shape, (10, 10, 5))
        self.assertTrue(np.all(mask.array == 1))
        self.assertEqual(mask.modality, "mask")
        self.assertEqual(mask.spacing, ref_img.spacing)

    def test_create_full_mask_invalid_input(self) -> None:
        bad_img = Image(
            array=np.zeros((10, 10)),  # 2D, invalid
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "must be 3D"):
            create_full_mask(bad_img)

    def test_with_source_mask_rejects_geometry_mismatch(self) -> None:
        image = Image(
            array=np.zeros((4, 4, 4)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )
        shifted_mask = Image(
            array=np.ones((4, 4, 4), dtype=np.uint8),
            spacing=(1.0, 1.0, 1.0),
            origin=(10.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        with self.assertRaisesRegex(ValueError, "Origin mismatch"):
            image.with_source_mask(shifted_mask)

    def test_direction_matrix_rejects_invalid_shape(self) -> None:
        from pictologics.loader import _direction_matrix

        with self.assertRaisesRegex(ValueError, "Direction must be a 3x3 matrix"):
            _direction_matrix([1.0, 0.0, 0.0])

    # --- _find_best_dicom_series_dir Tests ---
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    def test_find_best_dicom_series_dir_success(self, mock_is_dicom: MagicMock) -> None:
        # Construct a mock directory tree
        # root/
        #   subdir1/ (0 dicoms)
        #   subdir2/ (5 dicoms)
        #   subdir3/ (2 dicoms)

        mock_root = MagicMock()
        mock_root.exists.return_value = True

        subdir1 = MagicMock()
        subdir1.is_dir.return_value = True
        subdir1.iterdir.return_value = []  # Empty

        subdir2 = MagicMock()
        subdir2.is_dir.return_value = True
        # 5 dicom files
        files2 = [MagicMock() for _ in range(5)]
        for f in files2:
            f.is_file.return_value = True
        subdir2.iterdir.return_value = files2

        subdir3 = MagicMock()
        subdir3.is_dir.return_value = True
        # 2 dicom files
        files3 = [MagicMock() for _ in range(2)]
        for f in files3:
            f.is_file.return_value = True
        subdir3.iterdir.return_value = files3

        # rglob returns all subdirs
        mock_root.rglob.return_value = [subdir1, subdir2, subdir3]
        mock_root.iterdir.return_value = []  # Root has no files

        mock_is_dicom.return_value = True

        best = _find_best_dicom_series_dir(mock_root)
        self.assertEqual(best, subdir2)

    @patch("pictologics.loader.pydicom.misc.is_dicom")
    def test_find_best_dicom_series_dir_with_oserror(self, mock_is_dicom: MagicMock) -> None:
        mock_root = MagicMock()
        mock_root.exists.return_value = True

        subdir_ok = MagicMock()
        subdir_ok.is_dir.return_value = True
        file_ok = MagicMock()
        file_ok.is_file.return_value = True
        subdir_ok.iterdir.return_value = [file_ok]

        subdir_err = MagicMock()
        subdir_err.is_dir.return_value = True
        subdir_err.iterdir.side_effect = OSError("Permission denied")

        mock_root.rglob.return_value = [subdir_ok, subdir_err]
        mock_root.iterdir.return_value = []

        mock_is_dicom.return_value = True

        best = _find_best_dicom_series_dir(mock_root)
        self.assertEqual(best, subdir_ok)

    @patch("pictologics.loader.pydicom.misc.is_dicom")
    def test_find_best_dicom_series_dir_none_found(self, mock_is_dicom: MagicMock) -> None:
        mock_root = MagicMock()
        mock_root.exists.return_value = True
        mock_root.rglob.return_value = []
        mock_root.iterdir.return_value = [MagicMock()]  # One file
        # Mock file check fails or is_dicom fails
        mock_root.iterdir.return_value[0].is_file.return_value = True
        mock_is_dicom.return_value = False

        with self.assertRaisesRegex(ValueError, "No DICOM files found"):
            _find_best_dicom_series_dir(mock_root)

    def test_find_best_dicom_series_dir_not_exist(self) -> None:
        mock_p = MagicMock()
        mock_p.exists.return_value = False
        with self.assertRaisesRegex(ValueError, "Path does not exist"):
            _find_best_dicom_series_dir(mock_p)

    # --- load_image Dispatch Tests ---
    @patch("pictologics.loader.Path")
    def test_load_image_not_exists(self, mock_Path_cls: MagicMock) -> None:
        mock_Path_cls.return_value.exists.return_value = False
        with self.assertRaises(ValueError):
            load_image("fake_path")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_dicom_series")
    def test_load_image_directory(
        self, mock_load_series: MagicMock, mock_Path_cls: MagicMock
    ) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = True

        load_image("some_dir")
        mock_load_series.assert_called_once_with(mock_path_obj, 0, True, None)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_nifti")
    def test_load_image_nifti(self, mock_load_nifti: MagicMock, mock_Path_cls: MagicMock) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False
        # Need to simulate string behavior or ensure logic uses original path string for endswith check?
        # load_image uses `path.lower().endswith` on the input string, which isn't mocked.

        load_image("image.nii")
        mock_load_nifti.assert_called_once_with("image.nii", 0)

        mock_load_nifti.reset_mock()
        load_image("image.nii.gz", dataset_index=2)
        mock_load_nifti.assert_called_once_with("image.nii.gz", 2)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_dicom_file")
    def test_load_image_dicom_file(
        self, mock_load_dcm: MagicMock, mock_Path_cls: MagicMock
    ) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False

        load_image("image.dcm")
        mock_load_dcm.assert_called_once_with("image.dcm", True, 0)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_dicom_file")
    def test_load_image_unknown_format_fallback(
        self, mock_load_dcm: MagicMock, mock_Path_cls: MagicMock
    ) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False

        # Should try dicom loader if extension doesn't match nifti
        load_image("image.unknown")
        mock_load_dcm.assert_called_once_with("image.unknown", True, 0)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_dicom_file")
    def test_load_image_unknown_format_failure(
        self, mock_load_dcm: MagicMock, mock_Path_cls: MagicMock
    ) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False
        mock_load_dcm.side_effect = Exception("Not a DICOM")

        with self.assertRaises(ValueError):
            load_image("image.unknown")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_dicom_series")
    def test_load_image_exception_wrapping(
        self, mock_load_series: MagicMock, mock_Path_cls: MagicMock
    ) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = True
        mock_load_series.side_effect = RuntimeError("Unexpected error")

        with self.assertRaisesRegex(
            ValueError, "Failed to load image from 'some_dir': Unexpected error"
        ):
            load_image("some_dir")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_dicom_series")
    def test_load_image_directory_recursive_return(
        self, mock_load_series: MagicMock, mock_Path_cls: MagicMock
    ) -> None:
        # Verify the return value of _load_dicom_series is propagated
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = True

        mock_img = MagicMock()
        mock_load_series.return_value = mock_img

        # We need to mock _find_best_dicom_series_dir too since recursive=True calls it
        # Note: _find_best_dicom_series_dir is imported in test global scope
        with patch("pictologics.loader._find_best_dicom_series_dir") as mock_find:
            mock_find.return_value = mock_path_obj
            res = load_image("some_dir", recursive=True)
            self.assertEqual(res, mock_img)

    # --- _load_nifti Tests ---
    @patch("pictologics.loader.nib.load")
    def test_load_nifti_success(self, mock_nib_load: MagicMock) -> None:
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.zeros((10, 10, 5))
        mock_img.header.get_zooms.return_value = (1.0, 1.0, 2.0)
        mock_img.affine = np.eye(4)
        mock_nib_load.return_value = mock_img

        img = _load_nifti("test.nii")
        self.assertEqual(img.array.shape, (10, 10, 5))
        self.assertEqual(img.spacing, (1.0, 1.0, 2.0))
        self.assertEqual(img.origin, (0.0, 0.0, 0.0))
        # The RAS+ axes of the affine in the LPS+ frame of the package
        np.testing.assert_array_equal(img.direction, np.diag([-1.0, -1.0, 1.0]))
        self.assertEqual(img.modality, "Nifti")

    @patch("pictologics.loader.nib.load")
    def test_load_nifti_normalizes_affine_direction(self, mock_nib_load: MagicMock) -> None:
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.zeros((10, 10, 5))
        mock_img.header.get_zooms.return_value = (2.0, 3.0, 4.0)
        direction = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag([2.0, 3.0, 4.0])
        affine[:3, 3] = (10.0, -20.0, 30.0)
        mock_img.affine = affine
        mock_nib_load.return_value = mock_img

        img = _load_nifti("test.nii")

        self.assertEqual(img.spacing, (2.0, 3.0, 4.0))
        # RAS+ to LPS+: the X and Y rows change sign
        np.testing.assert_allclose(img.direction, np.diag([-1.0, -1.0, 1.0]) @ direction)
        self.assertEqual(img.origin, (-10.0, 20.0, 30.0))

    @patch("pictologics.loader.nib.load")
    def test_load_nifti_2d_zooms(self, mock_nib_load: MagicMock) -> None:
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.zeros((10, 10))  # 2D data
        mock_img.header.get_zooms.return_value = (0.5, 0.5)  # 2D zooms
        mock_img.affine = np.eye(4)
        mock_nib_load.return_value = mock_img

        img = _load_nifti("test.nii")
        self.assertEqual(img.array.shape, (10, 10, 1))  # Promoted to 3D
        self.assertEqual(img.spacing, (0.5, 0.5, 1.0))  # Padded spacing

    @patch("pictologics.loader.nib.load")
    def test_load_nifti_returns_row_order_array(self, mock_nib_load: MagicMock) -> None:
        # nibabel gives column-order (Fortran) data. A large array must come back in row
        # order with the same values; row-order and small arrays must not be copied.
        data_f = np.asfortranarray(np.arange(60, dtype=np.float64).reshape(3, 4, 5))
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = data_f
        mock_img.header.get_zooms.return_value = (1.0, 1.0, 1.0)
        mock_img.affine = np.eye(4)
        mock_nib_load.return_value = mock_img

        self.assertIs(_load_nifti("test.nii").array, data_f)  # below the size gate

        with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", 8):
            img = _load_nifti("test.nii")
            self.assertTrue(img.array.flags.c_contiguous)
            np.testing.assert_array_equal(img.array, data_f)

            data_c = np.zeros((3, 4, 5))
            mock_img.get_fdata.return_value = data_c
            self.assertIs(_load_nifti("test.nii").array, data_c)

            # Any other layout (here a strided view) takes the plain numpy copy.
            strided = np.arange(120, dtype=np.float64).reshape(6, 4, 5)[::2]
            out = _row_order(strided)
            self.assertTrue(out.flags.c_contiguous)
            np.testing.assert_array_equal(out, strided)

    def test_row_order_copies_every_loaded_type(self) -> None:
        # The tiled copy takes the NIfTI, DICOM and SEG types; another type takes numpy's
        # copy. Both give row order with the same values and type.
        from pictologics import loader

        with (
            patch("pictologics.loader._ROW_ORDER_MIN_SIZE", 8),
            patch(
                "pictologics.loader._to_row_order_numba", wraps=loader._to_row_order_numba
            ) as tiled,
        ):
            for dtype in (np.float64, np.int16, np.uint16, np.uint8, np.float32):
                col = np.asfortranarray(np.arange(60).reshape(3, 4, 5).astype(dtype))
                out = _row_order(col)
                self.assertTrue(out.flags.c_contiguous)
                self.assertEqual(out.dtype, dtype)
                np.testing.assert_array_equal(out, col)
        self.assertEqual(tiled.call_count, 4)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_load_dicom_series_memory_order(
        self,
        mock_split_phases: MagicMock,
        mock_dcmread: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        # The slices stack on a new first axis: a small volume comes back in column order
        # and a large one in row order, with the same (X, Y, Z) values and type.
        files = [MagicMock() for _ in range(3)]
        for f in files:
            f.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = files
        mock_split_phases.return_value = [[{"file_path": f} for f in files]]
        slices = []
        for z in range(3):
            s = _image_mock()
            s.pixel_array = np.arange(6, dtype=np.int16).reshape(2, 3) + 10 * z  # (Y, X)
            s.ImagePositionPatient = [0.0, 0.0, float(z)]
            s.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            s.PixelSpacing = [1.0, 1.0]
            s.SliceThickness = 1.0
            s.RescaleSlope = 1.0
            s.RescaleIntercept = 0.0
            s.Modality = "MR"
            del s.SpacingBetweenSlices
            slices.append(s)
        expected = np.stack([s.pixel_array.T for s in slices], axis=-1)  # (X, Y, Z)
        for limit, order in ((1 << 20, "f_contiguous"), (8, "c_contiguous")):
            mock_dcmread.side_effect = slices + slices  # header reads, then full reads
            with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", limit):
                img = _load_dicom_series("dicom_dir")
            np.testing.assert_array_equal(img.array, expected)
            self.assertEqual(img.array.dtype, np.float64)  # apply_rescale gives float64
            self.assertTrue(getattr(img.array.flags, order))

    def test_slice_spacing_prefers_positions_over_a_wrong_tag(self) -> None:
        # The tag wins when it agrees with the slice positions within 1%; else the
        # positions win, with a warning. Without positions, the tag or 1.0.
        from pictologics.loader import _slice_spacing

        z = np.array([0.0, 0.0, 1.0])
        positions = [np.array([0.0, 0.0, 0.625 * k]) for k in (3, 0, 2, 1)]  # any order
        self.assertEqual(_slice_spacing(None, positions, z), 0.625)
        self.assertEqual(_slice_spacing(0.63, positions, z), 0.63)  # within 1%
        with self.assertWarns(UserWarning):
            self.assertEqual(_slice_spacing(1.25, positions, z), 0.625)
        same = [np.zeros(3), np.zeros(3)]  # one position twice: no step to measure
        self.assertEqual(_slice_spacing(2.0, same, z), 2.0)
        self.assertEqual(_slice_spacing(None, positions[:1], z), 1.0)
        self.assertEqual(_slice_spacing(3.0, None, z), 3.0)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_load_dicom_series_overlapping_slices(
        self,
        mock_split_phases: MagicMock,
        mock_dcmread: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        # 1.25 mm slices every 0.625 mm: the spacing is the distance between the slices.
        files = [MagicMock() for _ in range(4)]
        for f in files:
            f.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = files
        mock_split_phases.return_value = [[{"file_path": f} for f in files]]
        slices = []
        for z in range(4):
            s = _image_mock()
            s.pixel_array = np.zeros((2, 3), dtype=np.int16)
            s.ImagePositionPatient = [0.0, 0.0, 0.625 * z]
            s.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            s.PixelSpacing = [0.5, 0.5]
            s.SliceThickness = 1.25
            s.RescaleSlope = 1.0
            s.RescaleIntercept = 0.0
            del s.SpacingBetweenSlices
            slices.append(s)
        mock_dcmread.side_effect = slices + slices
        with self.assertWarns(UserWarning):
            img = _load_dicom_series("dicom_dir")
        self.assertEqual(img.spacing, (0.5, 0.5, 0.625))

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_large_volume_row_order(self, mock_dcmread: MagicMock) -> None:
        # A large multiframe (Z, Y, X) volume comes back as (X, Y, Z) in row order.
        frames = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
        mock_dcm = MagicMock()
        mock_dcm.pixel_array = frames
        mock_dcm.PixelSpacing = [0.5, 0.5]
        mock_dcm.SliceThickness = 1.0
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        del mock_dcm.SpacingBetweenSlices
        mock_dcmread.return_value = mock_dcm

        with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", 8):
            img = _load_dicom_file("test.dcm")
        self.assertTrue(img.array.flags.c_contiguous)
        np.testing.assert_array_equal(img.array, frames.transpose(2, 1, 0))

    @patch("pictologics.loader.nib.load")
    def test_load_nifti_failure(self, mock_nib_load: MagicMock) -> None:
        mock_nib_load.side_effect = Exception("Corrupt file")
        with self.assertRaises(ValueError):
            _load_nifti("bad.nii")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_load_dicom_series_success(
        self,
        mock_split_phases: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        # Mock file iteration
        file1 = MagicMock()
        file1.is_file.return_value = True
        file2 = MagicMock()
        file2.is_file.return_value = True

        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        # Mock split_dicom_phases to return single phase with all files
        mock_split_phases.return_value = [[{"file_path": file1}, {"file_path": file2}]]

        # Create mock slices
        slice1 = _image_mock()
        slice1.pixel_array = np.zeros((512, 512))  # Y, X
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # Identity
        slice1.PixelSpacing = [0.5, 0.5]  # Row (Y), Col (X)
        slice1.SliceThickness = 1.0
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = -1024.0
        slice1.Modality = "CT"
        del slice1.SpacingBetweenSlices  # Ensure falls back to SliceThickness

        slice2 = _image_mock()
        slice2.pixel_array = np.zeros((512, 512))
        slice2.ImagePositionPatient = [0.0, 0.0, 1.0]  # Z=1
        slice2.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice2.PixelSpacing = [0.5, 0.5]
        slice2.SliceThickness = 1.0
        slice2.RescaleSlope = 1.0
        slice2.RescaleIntercept = -1024.0
        slice2.Modality = "CT"

        # First two calls are header reads, next two are full reads
        mock_dcmread.side_effect = [slice1, slice2, slice2, slice1]

        img = _load_dicom_series("dicom_dir")

        # Check shape: (X, Y, Z) -> (512, 512, 2)
        self.assertEqual(img.array.shape, (512, 512, 2))
        self.assertEqual(img.spacing, (0.5, 0.5, 1.0))
        self.assertEqual(img.origin, (0.0, 0.0, 0.0))
        self.assertEqual(img.modality, "CT")
        self.assertEqual(img.array[0, 0, 0], -1024.0)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_load_dicom_series_rescales_each_slice(
        self,
        mock_split_phases: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        file2 = MagicMock()
        file2.is_file.return_value = True

        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.iterdir.return_value = [file1, file2]
        mock_is_dicom.return_value = True
        mock_split_phases.return_value = [[{"file_path": file1}, {"file_path": file2}]]

        slice1 = _image_mock()
        slice1.pixel_array = np.array([[10]], dtype=np.uint16)
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice1.PixelSpacing = [1.0, 1.0]
        slice1.SliceThickness = 1.0
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = 0.0
        slice1.Modality = "CT"
        del slice1.SpacingBetweenSlices

        slice2 = _image_mock()
        slice2.pixel_array = np.array([[10]], dtype=np.uint16)
        slice2.ImagePositionPatient = [0.0, 0.0, 1.0]
        slice2.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice2.PixelSpacing = [1.0, 1.0]
        slice2.SliceThickness = 1.0
        slice2.RescaleSlope = 1.0
        slice2.RescaleIntercept = 100.0
        slice2.Modality = "CT"
        del slice2.SpacingBetweenSlices

        mock_dcmread.side_effect = [slice1, slice2, slice1, slice2]

        img = _load_dicom_series("dicom_dir")

        self.assertEqual(img.array.shape, (1, 1, 2))
        self.assertEqual(img.array[0, 0, 0], 10.0)
        self.assertEqual(img.array[0, 0, 1], 110.0)

    @patch("pictologics.loader.Path")
    def test_load_dicom_series_no_files(self, mock_Path_cls: MagicMock) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.iterdir.return_value = []  # Empty dir or no dicoms
        with self.assertRaises(ValueError):
            _load_dicom_series("empty_dir")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_load_dicom_series_fallback_sorting(
        self,
        mock_split_phases: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        file2 = MagicMock()
        file2.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1, file2]
        mock_is_dicom.return_value = True

        # Mock split_dicom_phases to return single phase with all files
        mock_split_phases.return_value = [[{"file_path": file1}, {"file_path": file2}]]

        # Slices without ImagePositionPatient/Orientation
        slice1 = _image_mock()
        slice1.pixel_array = np.zeros((10, 10))
        del slice1.ImagePositionPatient
        del slice1.ImageOrientationPatient
        slice1.InstanceNumber = 1
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = 0.0

        slice2 = _image_mock()
        slice2.pixel_array = np.zeros((10, 10))
        del slice2.ImagePositionPatient
        del slice2.ImageOrientationPatient
        slice2.InstanceNumber = 2
        slice2.RescaleSlope = 1.0
        slice2.RescaleIntercept = 0.0

        # First two calls are header reads, next two are full reads
        mock_dcmread.side_effect = [slice1, slice2, slice2, slice1]

        img = _load_dicom_series("dicom_dir")
        self.assertEqual(img.array.shape, (10, 10, 2))
        self.assertEqual(img.spacing, (1.0, 1.0, 1.0))

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_load_dicom_series_spacing_calculation(
        self,
        mock_split_phases: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        file2 = MagicMock()
        file2.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1, file2]
        mock_is_dicom.return_value = True

        # Mock split_dicom_phases to return single phase with all files
        mock_split_phases.return_value = [[{"file_path": file1}, {"file_path": file2}]]

        slice1 = _image_mock()
        slice1.pixel_array = np.zeros((10, 10))
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.PixelSpacing = [0.5, 0.5]
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = 0.0
        del slice1.SliceThickness
        del slice1.SpacingBetweenSlices

        slice2 = _image_mock()
        slice2.pixel_array = np.zeros((10, 10))
        slice2.ImagePositionPatient = [0.0, 0.0, 2.0]  # 2mm diff
        slice2.PixelSpacing = [0.5, 0.5]
        slice2.RescaleSlope = 1.0
        slice2.RescaleIntercept = 0.0

        # First two calls are header reads, next two are full reads
        mock_dcmread.side_effect = [slice1, slice2, slice1, slice2]

        img = _load_dicom_series("dicom_dir")
        self.assertEqual(img.spacing, (0.5, 0.5, 2.0))

    # --- _load_dicom_file Tests ---
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_success(self, mock_dcmread: MagicMock) -> None:
        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.zeros((100, 100))
        mock_dcm.PixelSpacing = [0.5, 0.5]
        del mock_dcm.SpacingBetweenSlices  # Ensure fallback to SliceThickness
        mock_dcm.SliceThickness = 2.0
        mock_dcm.ImagePositionPatient = [10.0, 10.0, 10.0]
        mock_dcm.Modality = "MR"
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")
        self.assertEqual(img.array.shape, (100, 100, 1))
        self.assertEqual(img.spacing, (0.5, 0.5, 2.0))
        self.assertEqual(img.origin, (10.0, 10.0, 10.0))

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_applies_top_level_rescale(self, mock_dcmread: MagicMock) -> None:
        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.array([[10, 20]], dtype=np.int16)
        mock_dcm.PixelSpacing = [1.0, 1.0]
        del mock_dcm.SpacingBetweenSlices
        mock_dcm.SliceThickness = 1.0
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.RescaleSlope = 2.0
        mock_dcm.RescaleIntercept = -5.0
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")

        self.assertEqual(img.array[0, 0, 0], 15.0)
        self.assertEqual(img.array[1, 0, 0], 35.0)

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_missing_metadata(self, mock_dcmread: MagicMock) -> None:
        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.zeros((100, 100))
        del mock_dcm.PixelSpacing
        del mock_dcm.ImagePositionPatient
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")
        self.assertEqual(img.spacing, (1.0, 1.0, 1.0))
        self.assertEqual(img.origin, (0.0, 0.0, 0.0))

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_failure(self, mock_dcmread: MagicMock) -> None:
        mock_dcmread.side_effect = Exception("Corrupt")
        with self.assertRaises(ValueError):
            _load_dicom_file("bad.dcm")

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_enhanced_multiframe(self, mock_dcmread: MagicMock) -> None:
        """Enhanced multiframe: geometry lives in functional groups (not top-level
        tags) and frames are stored out of spatial order, so the loader reads the
        shared/per-frame groups, spatially reorders the frames, and estimates the
        z-spacing from the frame positions."""
        pixel = np.arange(3 * 2 * 2, dtype=np.int16).reshape(3, 2, 2)  # (Z, Y, X)

        measures = SimpleNamespace(PixelSpacing=[0.5, 0.6])
        orient = SimpleNamespace(ImageOrientationPatient=[1, 0, 0, 0, 1, 0])
        shared = SimpleNamespace(
            PixelMeasuresSequence=[measures],
            PlaneOrientationSequence=[orient],
        )

        def frame(z: float) -> SimpleNamespace:
            return SimpleNamespace(
                PlanePositionSequence=[SimpleNamespace(ImagePositionPatient=[0.0, 0.0, z])]
            )

        per_frame = [frame(2.0), frame(1.0), frame(0.0)]  # descending z -> reorder

        dcm = MagicMock()
        dcm.pixel_array = pixel
        del dcm.ImageOrientationPatient
        del dcm.PixelSpacing
        del dcm.SpacingBetweenSlices
        del dcm.SliceThickness
        del dcm.ImagePositionPatient
        dcm.SharedFunctionalGroupsSequence = [shared]
        dcm.PerFrameFunctionalGroupsSequence = per_frame
        dcm.RescaleSlope = 1.0
        dcm.RescaleIntercept = 0.0
        mock_dcmread.return_value = dcm

        img = _load_dicom_file("mf.dcm")
        # spacing: X=PixelSpacing[1], Y=PixelSpacing[0], Z estimated from positions (=1)
        self.assertAlmostEqual(img.spacing[0], 0.6)
        self.assertAlmostEqual(img.spacing[1], 0.5)
        self.assertAlmostEqual(img.spacing[2], 1.0)
        # origin from the first spatially-sorted frame (z=0)
        self.assertEqual(img.origin, (0.0, 0.0, 0.0))

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_shared_plane_position(self, mock_dcmread: MagicMock) -> None:
        """A 2D frame with no per-frame positions falls back to the shared
        PlanePositionSequence for its origin."""
        dcm = MagicMock()
        dcm.pixel_array = np.zeros((2, 2), dtype=np.int16)
        dcm.PixelSpacing = [1.0, 1.0]
        del dcm.SpacingBetweenSlices
        dcm.SliceThickness = 1.0
        del dcm.ImageOrientationPatient
        del dcm.ImagePositionPatient
        pos = SimpleNamespace(ImagePositionPatient=[3.0, 4.0, 5.0])
        dcm.SharedFunctionalGroupsSequence = [SimpleNamespace(PlanePositionSequence=[pos])]
        dcm.RescaleSlope = 1.0
        dcm.RescaleIntercept = 0.0
        mock_dcmread.return_value = dcm

        img = _load_dicom_file("seg2d.dcm")
        self.assertEqual(img.origin, (3.0, 4.0, 5.0))

    @patch("pictologics.loader._is_dicom_seg")
    @patch("pictologics.loader.Path")
    def test_load_image_seg_ignores_dataset_index(
        self, mock_Path_cls: MagicMock, mock_is_seg: MagicMock
    ) -> None:
        """load_image warns that dataset_index/fill_value are ignored for DICOM SEG."""
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False
        mock_is_seg.return_value = True

        with patch.object(
            __import__("pictologics.loaders.seg_loader", fromlist=["load_seg"]),
            "load_seg",
            return_value=MagicMock(),
        ):
            with self.assertWarns(UserWarning):
                load_image("seg.dcm", dataset_index=1)

    # --- with_source_mask validation (merged from test_loader_coverage.py) ---
    def test_with_source_mask_shape_mismatch(self) -> None:
        img = Image(np.zeros((10, 10, 10), np.float32), (1, 1, 1), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "Source mask shape"):
            img.with_source_mask(np.zeros((5, 5, 5), dtype=bool))
        # A matching bool array and an Image mask are both accepted.
        self.assertTrue(img.with_source_mask(np.zeros((10, 10, 10), bool)).has_source_mask)
        mask_img = Image(np.zeros((10, 10, 10), np.uint8), (1, 1, 1), (0, 0, 0))
        self.assertTrue(img.with_source_mask(mask_img).has_source_mask)

    def test_with_source_mask_int_array(self) -> None:
        # Non-bool arrays are accepted; nonzero -> valid.
        img = Image(np.zeros((5, 5, 5), np.float32), (1, 1, 1), (0, 0, 0))
        int_mask = np.zeros((5, 5, 5), np.int32)
        int_mask[2, 2, 2] = 1
        masked = img.with_source_mask(int_mask)
        self.assertTrue(masked.source_mask[2, 2, 2])
        self.assertFalse(masked.source_mask[0, 0, 0])

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_pixel_array_failure(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1]
        mock_is_dicom.return_value = True

        slice1 = _image_mock()
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

        # Make accessing pixel_array raise exception
        type(slice1).pixel_array = PropertyMock(side_effect=Exception("Pixel error"))
        mock_dcmread.return_value = slice1

        with self.assertRaisesRegex(ValueError, "Failed to extract pixel arrays"):
            _load_dicom_series("dicom_dir")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_spacing_between_slices(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1]
        mock_is_dicom.return_value = True

        slice1 = _image_mock()
        slice1.pixel_array = np.zeros((10, 10))
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice1.PixelSpacing = [0.5, 0.5]
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = 0.0
        slice1.SpacingBetweenSlices = 2.5  # Should be preferred
        slice1.SliceThickness = 1.0  # Ignored if SpacingBetweenSlices exists

        mock_dcmread.return_value = slice1

        img = _load_dicom_series("dicom_dir")
        self.assertEqual(img.spacing, (0.5, 0.5, 2.5))

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_single_slice_no_thickness(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1]
        mock_is_dicom.return_value = True

        slice1 = _image_mock()
        slice1.pixel_array = np.zeros((10, 10))
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice1.PixelSpacing = [0.5, 0.5]
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = 0.0
        del slice1.SpacingBetweenSlices
        del slice1.SliceThickness  # Missing both

        mock_dcmread.return_value = slice1

        img = _load_dicom_series("dicom_dir")
        self.assertEqual(img.spacing, (0.5, 0.5, 1.0))  # Defaults to 1.0

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_read_error(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1]
        mock_is_dicom.return_value = True
        mock_dcmread.side_effect = Exception("Corrupt file")

        with self.assertRaisesRegex(ValueError, "Could not read any DICOM files"):
            _load_dicom_series("dicom_dir")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_missing_tags(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        file1 = MagicMock()
        file1.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1]
        mock_is_dicom.return_value = True

        slice1 = _image_mock()
        # Simulate missing attributes
        del slice1.ImagePositionPatient
        del slice1.PixelSpacing

        mock_dcmread.return_value = slice1

        img = _load_dicom_series("dicom_dir")
        # Should fallback to defaults
        self.assertEqual(img.spacing, (1.0, 1.0, 1.0))
        self.assertEqual(img.origin, (0.0, 0.0, 0.0))

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_load_dicom_series_dataset_index_out_of_range(
        self,
        mock_split_phases: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path_cls: MagicMock,
    ) -> None:
        """Test dataset_index out of range error."""
        file1 = MagicMock()
        file1.is_file.return_value = True
        mock_Path_cls.return_value.iterdir.return_value = [file1]
        mock_is_dicom.return_value = True

        dcm = _image_mock()
        dcm.InstanceNumber = 1
        dcm.ImagePositionPatient = [0, 0, 0]
        mock_dcmread.return_value = dcm

        # Single phase
        mock_split_phases.return_value = [[{"file_path": file1}]]

        with self.assertRaisesRegex(ValueError, "dataset_index 5 is out of range"):
            _load_dicom_series("dicom_dir", dataset_index=5)

    @patch("pictologics.loader._is_dicom_seg")
    @patch("pictologics.loader.Path")
    def test_load_image_seg_dict_return(
        self,
        mock_Path_cls: MagicMock,
        mock_is_seg: MagicMock,
    ) -> None:
        """Test load_image handles dict return from load_seg."""
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False

        mock_is_seg.return_value = True

        # Create mock images for dict return
        mock_img1 = MagicMock()
        mock_img2 = MagicMock()

        # Mock at the location where it's used (when import happens inside function)
        with patch.object(
            __import__("pictologics.loaders.seg_loader", fromlist=["load_seg"]),
            "load_seg",
        ) as mock_load_seg:
            # Mock load_seg returning a dict (combine_segments=False behavior)
            mock_load_seg.return_value = {"segment1": mock_img1, "segment2": mock_img2}
            result = load_image("seg.dcm")
            # Should return first value from dict
            self.assertEqual(result, mock_img1)

    @patch("pictologics.loader.pydicom.dcmread")
    @patch("pictologics.loader.Path")
    def test_load_image_seg_non_dict_return(
        self,
        mock_Path_cls: MagicMock,
        mock_dcmread: MagicMock,
    ) -> None:
        """Test load_image handles Image return from load_seg (line 435)."""
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_dir.return_value = False

        # Create a mock DICOM object that looks like a SEG
        mock_dcm = MagicMock()
        mock_dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"  # SEG SOP Class UID
        mock_dcmread.return_value = mock_dcm

        mock_img = MagicMock()

        with patch.object(
            __import__("pictologics.loaders.seg_loader", fromlist=["load_seg"]),
            "load_seg",
            return_value=mock_img,
        ):
            result = load_image("seg.dcm")
            self.assertEqual(result, mock_img)

    # --- load_and_merge_images Tests ---
    @patch("pictologics.loader.load_image")
    def test_load_and_merge_success(self, mock_load_image: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.array([[[1, 0], [0, 0]]])
        mask1.spacing = (1.0, 1.0, 1.0)
        mask1.origin = (0.0, 0.0, 0.0)
        mask1.direction = np.eye(3)

        mask2 = MagicMock()
        mask2.array = np.array([[[0, 2], [0, 0]]])
        mask2.spacing = (1.0, 1.0, 1.0)
        mask2.origin = (0.0, 0.0, 0.0)
        mask2.direction = np.eye(3)

        mock_load_image.side_effect = [mask1, mask2]

        merged = load_and_merge_images(["path1", "path2"])
        expected = np.array([[[1, 2], [0, 0]]])
        np.testing.assert_array_equal(merged.array, expected)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_conflict_resolution(self, mock_load_image: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.array([[[10]]])
        mask1.spacing = (1.0, 1.0, 1.0)
        mask1.origin = (0.0, 0.0, 0.0)
        mask1.direction = np.eye(3)

        mask2 = MagicMock()
        mask2.array = np.array([[[20]]])
        mask2.spacing = (1.0, 1.0, 1.0)
        mask2.origin = (0.0, 0.0, 0.0)
        mask2.direction = np.eye(3)

        # 'max'
        mock_load_image.side_effect = [mask1, mask2]
        merged = load_and_merge_images(["p1", "p2"], conflict_resolution="max")
        self.assertEqual(merged.array[0, 0, 0], 20)

        # 'min'
        mock_load_image.side_effect = [mask1, mask2]
        merged = load_and_merge_images(["p1", "p2"], conflict_resolution="min")
        self.assertEqual(merged.array[0, 0, 0], 10)

        # 'first'
        mock_load_image.side_effect = [mask1, mask2]
        merged = load_and_merge_images(["p1", "p2"], conflict_resolution="first")
        self.assertEqual(merged.array[0, 0, 0], 10)

        # 'last'
        mock_load_image.side_effect = [mask1, mask2]
        merged = load_and_merge_images(["p1", "p2"], conflict_resolution="last")
        self.assertEqual(merged.array[0, 0, 0], 20)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_geometry_mismatch_spacing(self, mock_load: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.zeros((10,))
        mask1.spacing = (1, 1, 1)
        mask1.origin = (0, 0, 0)
        mask1.direction = None
        mask2 = MagicMock()
        mask2.array = np.zeros((10,))
        mask2.spacing = (2, 2, 2)
        mask2.origin = (0, 0, 0)
        mask2.direction = None
        mock_load.side_effect = [mask1, mask2]
        with self.assertRaisesRegex(ValueError, "Spacing mismatch"):
            load_and_merge_images(["p1", "p2"])

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_geometry_mismatch_origin(self, mock_load: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.zeros((10,))
        mask1.spacing = (1, 1, 1)
        mask1.origin = (0, 0, 0)
        mask1.direction = None
        mask2 = MagicMock()
        mask2.array = np.zeros((10,))
        mask2.spacing = (1, 1, 1)
        mask2.origin = (10, 0, 0)
        mask2.direction = None
        mock_load.side_effect = [mask1, mask2]
        with self.assertRaisesRegex(ValueError, "Origin mismatch"):
            load_and_merge_images(["p1", "p2"])

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_geometry_mismatch_direction(self, mock_load: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.zeros((10,))
        mask1.spacing = (1, 1, 1)
        mask1.origin = (0, 0, 0)
        mask1.direction = np.eye(3)
        # Use a rotated matrix for mismatch
        rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        mask2 = MagicMock()
        mask2.array = np.zeros((10,))
        mask2.spacing = (1, 1, 1)
        mask2.origin = (0, 0, 0)
        mask2.direction = rot
        mock_load.side_effect = [mask1, mask2]
        with self.assertRaisesRegex(ValueError, "Direction mismatch"):
            load_and_merge_images(["p1", "p2"])

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_validation_against_reference(self, mock_load: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.zeros((10, 10, 10))
        mask1.spacing = (1, 1, 1)
        mask1.origin = (0, 0, 0)
        mask1.direction = None
        mock_load.side_effect = [mask1]

        ref = MagicMock()
        ref.array = np.zeros((5, 5, 5))
        ref.spacing = (1, 1, 1)
        ref.origin = (0, 0, 0)
        ref.direction = None

        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            load_and_merge_images(["p1"], reference_image=ref)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_load_failure(self, mock_load: MagicMock) -> None:
        # First image fails
        mock_load.side_effect = Exception("Read Error")
        with self.assertRaisesRegex(ValueError, "Failed to load first image"):
            load_and_merge_images(["p1"])

        # Second image fails
        mask1 = MagicMock()
        mask1.array = np.zeros((10,))
        mask1.spacing = (1, 1, 1)
        mask1.origin = (0, 0, 0)
        mask1.direction = None
        mock_load.side_effect = [mask1, Exception("Read Error 2")]
        with self.assertRaisesRegex(ValueError, "Failed to load image 'p2'"):
            load_and_merge_images(["p1", "p2"])

    def test_load_and_merge_empty_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_paths cannot be empty"):
            load_and_merge_images([])

    def test_load_and_merge_invalid_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid conflict_resolution"):
            load_and_merge_images(["p1"], conflict_resolution="invalid")

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_binarization(self, mock_load_image: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.array([[[0, 1, 2], [5, 10, 0]]])
        mask1.spacing = (1.0, 1.0, 1.0)
        mask1.origin = (0, 0, 0)
        mask1.direction = None
        mock_load_image.return_value = mask1

        # 1. Binarize=True (x > 0)
        merged = load_and_merge_images(["p1"], binarize=True)
        np.testing.assert_array_equal(
            merged.array, np.array([[[0, 1, 1], [1, 1, 0]]], dtype=np.uint8)
        )

        # 2. Binarize=int (e.g., 2)
        merged = load_and_merge_images(["p1"], binarize=2)
        np.testing.assert_array_equal(
            merged.array, np.array([[[0, 0, 1], [0, 0, 0]]], dtype=np.uint8)
        )

        # 3. Binarize=List (e.g., [1, 5])
        merged = load_and_merge_images(["p1"], binarize=[1, 5])
        np.testing.assert_array_equal(
            merged.array, np.array([[[0, 1, 0], [1, 0, 0]]], dtype=np.uint8)
        )

        # 4. Binarize=Tuple (range, e.g., (2, 5))
        merged = load_and_merge_images(["p1"], binarize=(2, 5))
        np.testing.assert_array_equal(
            merged.array, np.array([[[0, 0, 1], [1, 0, 0]]], dtype=np.uint8)
        )

        # 5. Fallback/Safe-guard (False should leave as is)
        merged = load_and_merge_images(["p1"], binarize=False)
        np.testing.assert_array_equal(merged.array, mask1.array)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_unknown_binarize_type(self, mock_load: MagicMock) -> None:
        mask1 = MagicMock()
        mask1.array = np.zeros((2, 2))
        mask1.spacing = (1, 1, 1)
        mask1.origin = (0, 0, 0)
        mask1.direction = None
        mock_load.return_value = mask1
        with self.assertRaisesRegex(ValueError, "Unsupported binarize value"):
            load_and_merge_images(["p1"], binarize="unsupported string")  # type: ignore


class TestRepositioning(unittest.TestCase):
    """Tests for the image repositioning functionality."""

    def test_position_in_reference_basic(self) -> None:
        """Test basic repositioning of a cropped image."""
        from pictologics.loader import _position_in_reference

        # Create a reference image (10x10x10)
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        # Create a cropped image (3x3x3) at position (2, 3, 4)
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.0, 3.0, 4.0),
            direction=np.eye(3),
            modality="mask",
        )

        result = _position_in_reference(cropped, reference)

        # Check shape matches reference
        self.assertEqual(result.array.shape, (10, 10, 10))

        # Check data is in correct position
        self.assertEqual(result.array[2, 3, 4], 1.0)
        self.assertEqual(result.array[4, 5, 6], 1.0)
        self.assertEqual(result.array[0, 0, 0], 0.0)  # Outside cropped region

        # Check geometry matches reference
        self.assertEqual(result.spacing, reference.spacing)
        self.assertEqual(result.origin, reference.origin)

    def test_position_in_reference_uses_direction_for_offset(self) -> None:
        """World offsets should be converted through reference direction."""
        from pictologics.loader import _position_in_reference

        direction = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        reference = Image(
            array=np.zeros((3, 3, 1)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=direction,
            modality="CT",
        )
        cropped = Image(
            array=np.ones((1, 1, 1)),
            spacing=(1.0, 1.0, 1.0),
            origin=tuple(direction[:, 0]),
            direction=direction,
            modality="mask",
        )

        result = _position_in_reference(cropped, reference)

        self.assertEqual(result.array[1, 0, 0], 1.0)
        self.assertEqual(result.array[0, 1, 0], 0.0)

    def test_position_in_reference_rejects_fractional_voxel_offset(self) -> None:
        """sub-voxel offset exceeding subvoxel_tolerance raises ValueError."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        # origin=(2.5, ...) gives max_offset_diff=0.5 (banker's rounding: round(2.5)=2)
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.5, 3.0, 4.0),
            direction=np.eye(3),
            modality="mask",
        )

        # Default tolerance=0.5: 0.5 is NOT > 0.5, so it snaps with a warning
        with self.assertWarns(UserWarning):
            result = _position_in_reference(cropped, reference)
        self.assertEqual(result.array.shape, (10, 10, 10))

        # With a stricter tolerance=0.3, 0.5 > 0.3 raises ValueError
        with self.assertRaisesRegex(ValueError, "subvoxel_tolerance"):
            _position_in_reference(cropped, reference, subvoxel_tolerance=0.3)

    def test_position_in_reference_boundary_clipping(self) -> None:
        """Test that cropped images extending beyond reference are clipped."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        # Cropped image that extends beyond reference (starts at 8, goes to 11)
        cropped = Image(
            array=np.ones((5, 5, 5)),
            spacing=(1.0, 1.0, 1.0),
            origin=(8.0, 8.0, 8.0),
            direction=np.eye(3),
            modality="mask",
        )

        result = _position_in_reference(cropped, reference, min_overlap_fraction=0.0)

        # Should be clipped to fit within reference
        self.assertEqual(result.array.shape, (10, 10, 10))
        # Only the portion within reference (8-9 in each dim) should have data
        self.assertEqual(result.array[8, 8, 8], 1.0)
        self.assertEqual(result.array[9, 9, 9], 1.0)
        self.assertEqual(result.array[7, 7, 7], 0.0)

    def test_position_in_reference_no_overlap_warning(self) -> None:
        """With min_overlap_fraction=0.0, a fully disjoint mask warns and returns empty."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        # Cropped image completely outside reference
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(100.0, 100.0, 100.0),  # Far outside
            direction=np.eye(3),
            modality="mask",
        )

        # With min_overlap_fraction=0.0, preserves old warn+empty-array behaviour.
        with self.assertWarns(UserWarning):
            result = _position_in_reference(cropped, reference, min_overlap_fraction=0.0)

        self.assertEqual(result.array.shape, (10, 10, 10))
        self.assertEqual(result.array.sum(), 0.0)

    def test_position_in_reference_no_overlap_raises_by_default(self) -> None:
        """By default a fully disjoint mask raises ValueError."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(100.0, 100.0, 100.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "min_overlap_fraction"):
            _position_in_reference(cropped, reference)

    def test_position_in_reference_partial_overlap_below_fraction_raises(self) -> None:
        """A mask with partial but insufficient overlap raises ValueError."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        # Mask is 4x4x4=64 voxels; only 1x1x1=1 voxel overlaps (1/64 ≈ 1.6%)
        cropped = Image(
            array=np.ones((4, 4, 4)),
            spacing=(1.0, 1.0, 1.0),
            origin=(9.0, 9.0, 9.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "min_overlap_fraction"):
            _position_in_reference(cropped, reference, min_overlap_fraction=0.5)

    def test_position_in_reference_partial_overlap_above_fraction_succeeds(self) -> None:
        """A mask with sufficient overlap repositions without error."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        # Mask is 4x4x4=64 voxels; 2x2x2=8 voxels overlap (8/64=12.5% > 0.05)
        cropped = Image(
            array=np.ones((4, 4, 4)),
            spacing=(1.0, 1.0, 1.0),
            origin=(8.0, 8.0, 8.0),
            direction=np.eye(3),
            modality="mask",
        )
        result = _position_in_reference(cropped, reference, min_overlap_fraction=0.05)
        self.assertEqual(result.array.shape, (10, 10, 10))
        self.assertGreater(result.array.sum(), 0)

    def test_position_in_reference_subvoxel_warning_threshold(self) -> None:
        """Drift above warning threshold but below tolerance emits UserWarning."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        # offset 0.05 voxels — above default warning threshold 0.01, below tolerance 0.5
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.05, 3.0, 4.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertWarns(UserWarning):
            result = _position_in_reference(cropped, reference)
        self.assertEqual(result.array.shape, (10, 10, 10))

    def test_position_in_reference_tiny_drift_is_silent(self) -> None:
        """Drift below warning threshold produces no warning."""
        import warnings as _warnings

        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        # origin has floating-point noise below the 0.01 warning threshold
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.001, 3.0, 4.0),
            direction=np.eye(3),
            modality="mask",
        )

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            result = _position_in_reference(cropped, reference)
        self.assertEqual(result.array.shape, (10, 10, 10))

    def test_position_in_reference_rejects_invalid_min_overlap_fraction(self) -> None:
        """min_overlap_fraction outside [0, 1] raises ValueError immediately."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.0, 3.0, 4.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "min_overlap_fraction must be in"):
            _position_in_reference(cropped, reference, min_overlap_fraction=2.0)
        with self.assertRaisesRegex(ValueError, "min_overlap_fraction must be in"):
            _position_in_reference(cropped, reference, min_overlap_fraction=-0.1)

    def test_position_in_reference_rejects_negative_tolerance(self) -> None:
        """Negative subvoxel_tolerance or subvoxel_warning_threshold raises ValueError."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.0, 3.0, 4.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "subvoxel_tolerance must be"):
            _position_in_reference(cropped, reference, subvoxel_tolerance=-0.1)
        with self.assertRaisesRegex(ValueError, "subvoxel_warning_threshold must be"):
            _position_in_reference(cropped, reference, subvoxel_warning_threshold=-0.1)

    def test_position_in_reference_rejects_threshold_above_tolerance(self) -> None:
        """subvoxel_warning_threshold > subvoxel_tolerance raises ValueError."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.0, 3.0, 4.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "subvoxel_warning_threshold"):
            _position_in_reference(
                cropped, reference, subvoxel_tolerance=0.1, subvoxel_warning_threshold=0.4
            )

    def test_position_in_reference_transpose_axes(self) -> None:
        """Test axis transposition during repositioning."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 5)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        # Create a cropped image with different axis order
        cropped_data = np.zeros((2, 3, 4))
        cropped_data[0, 0, 0] = 1.0
        cropped = Image(
            array=cropped_data,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            modality="mask",
        )

        # Transpose Y and Z (axes 1 and 2)
        result = _position_in_reference(cropped, reference, transpose_axes=(0, 2, 1))

        # After transposition, shape should work with reference
        self.assertEqual(result.array.shape, (10, 10, 5))
        self.assertEqual(result.array[0, 0, 0], 1.0)

    def test_position_in_reference_transpose_axes_updates_geometry(self) -> None:
        """Axis transposition must permute spacing and direction metadata too."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((5, 5, 5)),
            spacing=(1.0, 2.0, 3.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        # The voxel data will be transposed from (X, Z, Y) to (X, Y, Z).
        # Its spacing and direction columns must follow the same axis permutation.
        cropped_data = np.zeros((2, 3, 4))
        cropped_data[1, 2, 3] = 7.0
        cropped = Image(
            array=cropped_data,
            spacing=(1.0, 3.0, 2.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            modality="mask",
        )

        result = _position_in_reference(cropped, reference, transpose_axes=(0, 2, 1))

        self.assertEqual(result.array.shape, reference.array.shape)
        self.assertEqual(result.array[1, 3, 2], 7.0)

    def test_position_in_reference_spacing_mismatch(self) -> None:
        """Test error when spacing is incompatible."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(2.0, 2.0, 2.0),  # Different spacing
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "Spacing mismatch"):
            _position_in_reference(cropped, reference)

    def test_position_in_reference_invalid_transpose_axes(self) -> None:
        """Invalid axis permutations should be rejected before repositioning."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "transpose_axes"):
            _position_in_reference(cropped, reference, transpose_axes=(0, 0, 1))

    def test_position_in_reference_orientation_mismatch_error(self) -> None:
        """Axes that are not the reference axes in another order need resampling."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        rotated = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=rotated,
            modality="mask",
        )

        with self.assertRaisesRegex(ValueError, "Orientation mismatch"):
            _position_in_reference(cropped, reference)

    def test_position_in_reference_fill_value(self) -> None:
        """Test custom fill value."""
        from pictologics.loader import _position_in_reference

        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        cropped = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(5.0, 5.0, 5.0),
            direction=np.eye(3),
            modality="mask",
        )

        result = _position_in_reference(cropped, reference, fill_value=-1.0)

        # Background should be -1
        self.assertEqual(result.array[0, 0, 0], -1.0)
        # Data region should have original values
        self.assertEqual(result.array[5, 5, 5], 1.0)

    @patch("pictologics.loader._load_dicom_file")
    @patch("pictologics.loader.Path")
    def test_load_image_with_reference_repositioning(
        self, mock_Path: MagicMock, mock_load_dcm: MagicMock
    ) -> None:
        """Test load_image with reference_image triggers repositioning."""
        mock_Path.return_value.exists.return_value = True
        mock_Path.return_value.is_dir.return_value = False

        # Create mock loaded image (cropped)
        cropped_img = Image(
            array=np.ones((3, 3, 3)),
            spacing=(1.0, 1.0, 1.0),
            origin=(2.0, 2.0, 2.0),
            direction=np.eye(3),
            modality="mask",
        )
        mock_load_dcm.return_value = cropped_img

        # Reference image
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        result = load_image("cropped.dcm", reference_image=reference)

        # Result should have reference shape
        self.assertEqual(result.array.shape, (10, 10, 10))
        # Data should be at correct position
        self.assertEqual(result.array[2, 2, 2], 1.0)

    @patch("pictologics.loader._load_dicom_file")
    @patch("pictologics.loader.Path")
    def test_load_image_with_reference_same_shape_validates_geometry(
        self, mock_Path: MagicMock, mock_load_dcm: MagicMock
    ) -> None:
        """Same-shaped images still need physical geometry validation."""
        mock_Path.return_value.exists.return_value = True
        mock_Path.return_value.is_dir.return_value = False

        shifted_img = Image(
            array=np.ones((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(5.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="mask",
        )
        mock_load_dcm.return_value = shifted_img
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        with self.assertRaisesRegex(ValueError, "Origin mismatch"):
            load_image("mask.dcm", reference_image=reference)

    @patch("pictologics.loaders.seg_loader.load_seg")
    @patch("pictologics.loader._is_dicom_seg")
    @patch("pictologics.loader.Path")
    def test_load_image_passes_alignment_controls_to_seg_loader(
        self,
        mock_Path: MagicMock,
        mock_is_seg: MagicMock,
        mock_load_seg: MagicMock,
    ) -> None:
        """DICOM SEG path loading must honour reference-alignment controls."""
        mock_Path.return_value.exists.return_value = True
        mock_Path.return_value.is_dir.return_value = False
        mock_is_seg.return_value = True
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )
        seg = Image(
            array=np.zeros((10, 10, 10), dtype=np.uint8),
            spacing=reference.spacing,
            origin=reference.origin,
            direction=reference.direction,
            modality="SEG",
        )
        mock_load_seg.return_value = seg

        result = load_image(
            "mask_seg.dcm",
            reference_image=reference,
            transpose_axes=(0, 2, 1),
            subvoxel_tolerance=0.05,
            subvoxel_warning_threshold=0.01,
            min_overlap_fraction=0.25,
        )

        self.assertIs(result, seg)
        mock_load_seg.assert_called_once_with(
            "mask_seg.dcm",
            reference_image=reference,
            transpose_axes=(0, 2, 1),
            subvoxel_tolerance=0.05,
            subvoxel_warning_threshold=0.01,
            min_overlap_fraction=0.25,
        )

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_reposition_to_reference(self, mock_load: MagicMock) -> None:
        """Test load_and_merge_images with reposition_to_reference=True."""

        # Reference image
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="CT",
        )

        # Two cropped masks at different positions
        mask1 = Image(
            array=np.ones((2, 2, 2)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            modality="mask",
        )
        mask2 = Image(
            array=np.ones((2, 2, 2)) * 2,
            spacing=(1.0, 1.0, 1.0),
            origin=(5.0, 5.0, 5.0),
            direction=np.eye(3),
            modality="mask",
        )

        mock_load.side_effect = [mask1, mask2]

        merged = load_and_merge_images(
            ["p1", "p2"],
            reference_image=reference,
            reposition_to_reference=True,
        )

        # Should have reference shape
        self.assertEqual(merged.array.shape, (10, 10, 10))
        # Both masks should be present at their positions
        self.assertEqual(merged.array[0, 0, 0], 1.0)
        self.assertEqual(merged.array[5, 5, 5], 2.0)

    def test_load_and_merge_reposition_no_reference_error(self) -> None:
        """Test error when reposition_to_reference=True but no reference provided."""
        with self.assertRaisesRegex(ValueError, "reference_image must be provided"):
            load_and_merge_images(["p1"], reposition_to_reference=True, reference_image=None)

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_spacing_between_slices_preferred(
        self, mock_dcmread: MagicMock
    ) -> None:
        """Test that SpacingBetweenSlices is preferred over SliceThickness."""
        from pictologics.loader import _load_dicom_file

        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.zeros((10, 10))
        mock_dcm.PixelSpacing = [0.5, 0.5]
        mock_dcm.SpacingBetweenSlices = 2.5  # Should be used
        mock_dcm.SliceThickness = 1.0  # Should be ignored
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")
        self.assertEqual(img.spacing[2], 2.5)

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_3d_data(self, mock_dcmread: MagicMock) -> None:
        """Test that 3D DICOM data is handled correctly with axis swap."""
        from pictologics.loader import _load_dicom_file

        # 3D data in (Z, Y, X) format
        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.zeros((5, 10, 20))  # Z=5, Y=10, X=20
        mock_dcm.PixelSpacing = [0.5, 0.5]
        mock_dcm.SliceThickness = 1.0
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        # Remove SpacingBetweenSlices to test SliceThickness fallback
        del mock_dcm.SpacingBetweenSlices
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")
        # Should be swapped to (X, Y, Z)
        self.assertEqual(img.array.shape, (20, 10, 5))

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_multiframe_rescales_each_frame(self, mock_dcmread: MagicMock) -> None:
        """Enhanced multiframe DICOM can store per-frame rescale parameters."""
        from pictologics.loader import _load_dicom_file

        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.array([[[1, 2]], [[1, 2]]], dtype=np.int16)
        mock_dcm.PixelSpacing = [1.0, 1.0]
        mock_dcm.SliceThickness = 1.0
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        del mock_dcm.SpacingBetweenSlices

        frame_0 = MagicMock()
        frame_0.PixelValueTransformationSequence = [
            MagicMock(RescaleSlope=1.0, RescaleIntercept=0.0)
        ]
        frame_1 = MagicMock()
        frame_1.PixelValueTransformationSequence = [
            MagicMock(RescaleSlope=1.0, RescaleIntercept=100.0)
        ]
        mock_dcm.PerFrameFunctionalGroupsSequence = [frame_0, frame_1]
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")

        self.assertEqual(img.array.shape, (2, 1, 2))
        self.assertEqual(img.array[0, 0, 0], 1.0)
        self.assertEqual(img.array[1, 0, 0], 2.0)
        self.assertEqual(img.array[0, 0, 1], 101.0)
        self.assertEqual(img.array[1, 0, 1], 102.0)

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_multiframe_uses_shared_rescale_sequence(
        self, mock_dcmread: MagicMock
    ) -> None:
        """Shared functional groups apply when frame-level transforms are absent."""
        from pictologics.loader import _load_dicom_file

        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.array([[[1]], [[2]]], dtype=np.int16)
        mock_dcm.PixelSpacing = [1.0, 1.0]
        mock_dcm.SliceThickness = 1.0
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        del mock_dcm.SpacingBetweenSlices

        frame_0 = MagicMock()
        frame_0.PixelValueTransformationSequence = []
        frame_1 = MagicMock()
        frame_1.PixelValueTransformationSequence = []
        shared = MagicMock()
        shared.PixelValueTransformationSequence = [
            MagicMock(RescaleSlope=2.0, RescaleIntercept=10.0)
        ]
        mock_dcm.PerFrameFunctionalGroupsSequence = [frame_0, frame_1]
        mock_dcm.SharedFunctionalGroupsSequence = [shared]
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")

        self.assertEqual(img.array[0, 0, 0], 12.0)
        self.assertEqual(img.array[0, 0, 1], 14.0)

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_no_spacing_fallback(self, mock_dcmread: MagicMock) -> None:
        """Test fallback to 1.0 when neither SpacingBetweenSlices nor SliceThickness."""
        from pictologics.loader import _load_dicom_file

        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.zeros((10, 10))
        mock_dcm.PixelSpacing = [0.5, 0.5]
        # No SpacingBetweenSlices or SliceThickness
        del mock_dcm.SpacingBetweenSlices
        del mock_dcm.SliceThickness
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")
        self.assertEqual(img.spacing[2], 1.0)

    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_direction_extraction(self, mock_dcmread: MagicMock) -> None:
        """Test direction matrix extraction from ImageOrientationPatient."""
        from pictologics.loader import _load_dicom_file

        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.zeros((10, 10))
        mock_dcm.PixelSpacing = [0.5, 0.5]
        mock_dcm.SliceThickness = 1.0
        mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_dcm.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dcm.RescaleSlope = 1.0
        mock_dcm.RescaleIntercept = 0.0
        del mock_dcm.SpacingBetweenSlices
        mock_dcmread.return_value = mock_dcm

        img = _load_dicom_file("test.dcm")
        # Direction should be extracted as 3x3 matrix
        self.assertEqual(img.direction.shape, (3, 3))
        np.testing.assert_array_almost_equal(img.direction[:, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(img.direction[:, 1], [0.0, 1.0, 0.0])

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_relabel_masks_reposition(self, mock_load_image: MagicMock) -> None:
        """Test relabel_masks with reposition_to_reference."""
        # Reference image
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        # Two cropped masks with binary values
        mask1 = Image(
            array=np.ones((3, 3, 3)),  # All 1s
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )
        mask2 = Image(
            array=np.ones((3, 3, 3)),  # All 1s
            spacing=(1.0, 1.0, 1.0),
            origin=(3.0, 3.0, 3.0),  # Different origin
            direction=np.eye(3),
        )

        mock_load_image.side_effect = [mask1, mask2]

        merged = load_and_merge_images(
            ["p1", "p2"],
            reference_image=reference,
            reposition_to_reference=True,
            relabel_masks=True,  # Should assign 1 to first, 2 to second
        )

        # Check labels are different
        self.assertIn(1.0, merged.array)
        self.assertIn(2.0, merged.array)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_conflict_resolution_min_reposition(
        self, mock_load_image: MagicMock
    ) -> None:
        """Test conflict_resolution='min' with reposition."""
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        # Two overlapping masks with different values
        mask1 = Image(
            array=np.full((5, 5, 5), 5.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )
        mask2 = Image(
            array=np.full((5, 5, 5), 3.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),  # Same origin - overlap
            direction=np.eye(3),
        )

        mock_load_image.side_effect = [mask1, mask2]

        merged = load_and_merge_images(
            ["p1", "p2"],
            reference_image=reference,
            reposition_to_reference=True,
            conflict_resolution="min",
        )

        # Min of 5 and 3 is 3
        self.assertEqual(merged.array[0, 0, 0], 3.0)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_conflict_resolution_last_reposition(
        self, mock_load_image: MagicMock
    ) -> None:
        """Test conflict_resolution='last' with reposition."""
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        mask1 = Image(
            array=np.full((5, 5, 5), 5.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )
        mask2 = Image(
            array=np.full((5, 5, 5), 9.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        mock_load_image.side_effect = [mask1, mask2]

        merged = load_and_merge_images(
            ["p1", "p2"],
            reference_image=reference,
            reposition_to_reference=True,
            conflict_resolution="last",
        )

        # Last wins: should be 9
        self.assertEqual(merged.array[0, 0, 0], 9.0)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_conflict_resolution_first_reposition(
        self, mock_load_image: MagicMock
    ) -> None:
        """Test conflict_resolution='first' with reposition."""
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        mask1 = Image(
            array=np.full((5, 5, 5), 5.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )
        mask2 = Image(
            array=np.full((5, 5, 5), 9.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        mock_load_image.side_effect = [mask1, mask2]

        merged = load_and_merge_images(
            ["p1", "p2"],
            reference_image=reference,
            reposition_to_reference=True,
            conflict_resolution="first",
        )

        # First wins: should be 5
        self.assertEqual(merged.array[0, 0, 0], 5.0)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_conflict_resolution_max_reposition(
        self, mock_load_image: MagicMock
    ) -> None:
        """Test conflict_resolution='max' with reposition (default behavior)."""
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        # Overlapping masks
        mask1 = Image(
            array=np.full((5, 5, 5), 5.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )
        mask2 = Image(
            array=np.full((5, 5, 5), 9.0),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        mock_load_image.side_effect = [mask1, mask2]

        merged = load_and_merge_images(
            ["p1", "p2"],
            reference_image=reference,
            reposition_to_reference=True,
            conflict_resolution="max",
        )

        # Max wins: should be 9
        self.assertEqual(merged.array[0, 0, 0], 9.0)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_relabel_masks_standard_mode(self, mock_load_image: MagicMock) -> None:
        """Test relabel_masks in standard mode (no repositioning)."""
        # Two masks with identical geometry
        mask1 = Image(
            array=np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]]),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )
        mask2 = Image(
            array=np.array([[[0, 0], [0, 1]], [[0, 0], [0, 0]]]),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        mock_load_image.side_effect = [mask1, mask2]

        merged = load_and_merge_images(
            ["p1", "p2"],
            relabel_masks=True,
        )

        # First mask's 1 should stay as 1, second mask's 1 should become 2
        self.assertEqual(merged.array[0, 0, 0], 1)
        self.assertEqual(merged.array[0, 1, 1], 2)

    @patch("pictologics.loader.load_image")
    def test_load_and_merge_load_error_reposition(self, mock_load_image: MagicMock) -> None:
        """Test error handling when load_image fails in reposition mode."""
        reference = Image(
            array=np.zeros((10, 10, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        mock_load_image.side_effect = Exception("Failed to read file")

        with self.assertRaisesRegex(ValueError, "Failed to load image"):
            load_and_merge_images(
                ["bad_path"],
                reference_image=reference,
                reposition_to_reference=True,
            )


def _write_series(folder: "os.PathLike[str]", rescale: list[tuple[float, float]]) -> np.ndarray:
    """A small CT series, one slice per (slope, intercept), written in shuffled order.
    Returns the expected (X, Y, Z) volume: each slice as float64 * slope + intercept when
    it has a rescale, else its stored values."""
    from pathlib import Path

    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    rng = np.random.default_rng(12)
    series = generate_uid()
    planes = []
    order = rng.permutation(len(rescale))
    for k, (slope, intercept) in enumerate(rescale):
        pixels = rng.integers(-2000, 2000, (6, 5), dtype=np.int16)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = pydicom.Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID, ds.Modality, ds.InstanceNumber = series, "CT", k + 1
        ds.Rows, ds.Columns = pixels.shape
        ds.PixelSpacing, ds.SliceThickness = [0.5, 0.75], 2.0
        ds.ImagePositionPatient = [0.0, 0.0, 2.0 * k]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.PixelData = pixels.tobytes()
        ds.save_as(Path(folder) / f"{order[k]:02d}.dcm", enforce_file_format=True)
        rescaled = slope != 1.0 or intercept != 0.0
        planes.append(pixels.astype(np.float64) * slope + intercept if rescaled else pixels)
    return np.stack([p.T for p in planes], axis=-1)


def test_dicom_series_reads_each_file_once_and_rescales_in_one_pass(
    tmp_path: "os.PathLike[str]",
) -> None:
    # Each file is read once. A large rescaled series goes through one kernel into the
    # row-order float64 output; it equals the per-slice float64 rescale, bit for bit, also
    # with slices that keep their stored values. A small one keeps the stacked path.
    import pydicom

    expected = _write_series(tmp_path, [(1.0, 0.0), (2.5, -1024.0), (1.0, -1024.0), (0.5, 0.0)])
    with patch("pictologics.loader.pydicom.dcmread", side_effect=pydicom.dcmread) as reads:
        small = _load_dicom_series(tmp_path)
    assert reads.call_count == 4
    with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", 8):
        large = _load_dicom_series(tmp_path)
    for image in (small, large):
        assert image.array.dtype == np.float64
        np.testing.assert_array_equal(image.array.view(np.uint64), expected.view(np.uint64))
    assert large.array.flags.c_contiguous and small.array.flags.f_contiguous
    assert large.spacing == small.spacing == (0.75, 0.5, 2.0)

    # No rescale tags: float64 values (the fused kernel for a large series), as from
    # NIfTI; with apply_rescale=False the stored values in their stored type
    plain = tmp_path / "plain"
    plain.mkdir()
    expected = _write_series(plain, [(1.0, 0.0)] * 3)
    for limit in (1 << 20, 8):
        with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", limit):
            image = _load_dicom_series(plain)
        assert image.array.dtype == np.float64
        np.testing.assert_array_equal(image.array, expected)
    raw = _load_dicom_series(plain, apply_rescale=False)
    assert raw.array.dtype == np.int16
    np.testing.assert_array_equal(raw.array, expected)


def test_nifti_loads_the_values_of_get_fdata_in_one_pass(tmp_path: "os.PathLike[str]") -> None:
    # A large 3D NIfTI image goes from its stored values to the row-order float64 output in
    # one pass, scaled as nibabel scales for get_fdata; the values are the same, bit for bit.
    from pathlib import Path

    import nibabel as nib

    rng = np.random.default_rng(13)
    affine = np.diag([0.5, 0.75, 2.0, 1.0])
    cases = {
        "int16.nii": rng.integers(-1000, 3000, (7, 6, 5)).astype(np.int16),
        "uint8.nii.gz": rng.integers(0, 2, (7, 6, 5)).astype(np.uint8),
        "float32.nii": rng.normal(size=(7, 6, 5)).astype(np.float32),
    }
    scaled = nib.Nifti1Image(rng.normal(0.0, 500.0, (7, 6, 5)), affine)
    scaled.set_data_dtype(np.int16)  # nibabel picks a slope and an intercept on save
    for name, data in cases.items():
        nib.save(nib.Nifti1Image(data, affine), Path(tmp_path) / name)
    nib.save(scaled, Path(tmp_path) / "scaled.nii")
    # scl_slope 1 and scl_inter 5 (header bytes 112-119): only the intercept applies
    nib.save(nib.Nifti1Image(cases["int16.nii"], affine), Path(tmp_path) / "offset.nii")
    with open(Path(tmp_path) / "offset.nii", "r+b") as fh:
        fh.seek(112)
        fh.write(np.array([1.0, 5.0], dtype="<f4").tobytes())
    for name in (*cases, "scaled.nii", "offset.nii"):
        path = str(Path(tmp_path) / name)
        expected = nib.load(path).get_fdata()
        with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", 8):
            array = _load_nifti(path).array
        assert array.flags.c_contiguous and array.dtype == np.float64
        np.testing.assert_array_equal(array.view(np.uint64), expected.view(np.uint64))
    assert nib.load(str(Path(tmp_path) / "scaled.nii")).dataobj.slope != 1.0
    assert nib.load(str(Path(tmp_path) / "offset.nii")).dataobj.inter == 5.0


def test_load_image_accepts_path_objects(tmp_path: "os.PathLike[str]") -> None:
    # A pathlib.Path loads the same image as its string, also in load_and_merge_images.
    from pathlib import Path

    import nibabel as nib

    data = np.arange(60, dtype=np.int16).reshape(5, 4, 3)
    image = Path(tmp_path) / "image.nii.gz"
    mask = Path(tmp_path) / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), image)
    nib.save(nib.Nifti1Image((data > 30).astype(np.uint8), np.eye(4)), mask)
    np.testing.assert_array_equal(load_image(image).array, load_image(str(image)).array)
    np.testing.assert_array_equal(
        load_and_merge_images([mask, mask]).array,
        load_and_merge_images([str(mask), str(mask)]).array,
    )


if __name__ == "__main__":
    unittest.main()


# --- Series choice, phases and geometry checks (synthetic files, no patient data) ---


def _ras(lps: np.ndarray) -> np.ndarray:
    """The NIfTI (RAS+) affine of an LPS+ voxel-to-world matrix."""
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps


def _write_slice(
    path: "os.PathLike[str]",
    *,
    series: str = "1.2.826.0.1.3680043.2.1125.10",
    number: int = 1,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pixels: "np.ndarray | None" = None,
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    sop_class: str = "1.2.840.10008.5.1.4.1.1.2",
    **tags: object,
) -> None:
    """One synthetic CT slice; `pixels` is (Rows, Columns), or (Rows, Columns, 3) for RGB."""
    from pathlib import Path

    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    pixels = np.zeros((4, 3), np.int16) if pixels is None else pixels
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID, ds.Modality, ds.InstanceNumber = series, "CT", number
    ds.Rows, ds.Columns = pixels.shape[:2]
    ds.PixelSpacing, ds.SliceThickness = [0.5, 0.75], 1.5
    ds.ImagePositionPatient = list(position)
    ds.ImageOrientationPatient = list(orientation)
    if pixels.ndim == 3:
        ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    else:
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.PixelData = np.ascontiguousarray(pixels).tobytes()
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.save_as(Path(path), enforce_file_format=True)


def test_reoriented_turns_the_axes_to_the_reference() -> None:
    # Axes that are the reference axes in another order and with other signs are turned:
    # the array, the spacing, the origin (the new first voxel) and the source mask. The
    # same axes, or axes at 30 degrees, keep the image as it is.
    from numpy.testing import assert_array_equal

    from pictologics.loader import _reoriented

    reference = Image(np.zeros((4, 3, 2)), (1.0, 2.0, 3.0), (5.0, 6.0, 7.0))
    values = np.arange(24.0).reshape(4, 3, 2)
    stored = np.transpose(values, (2, 0, 1))[:, ::-1, :]  # (z, x flipped, y)
    direction = np.eye(3)[:, [2, 0, 1]] * [1.0, -1.0, 1.0]
    image = Image(stored, (3.0, 1.0, 2.0), (8.0, 6.0, 7.0), direction, source_mask=stored > 5)
    turned = _reoriented(image, reference)
    assert_array_equal(turned.array, values)
    assert turned.array.flags.c_contiguous
    assert turned.spacing == (1.0, 2.0, 3.0) and turned.origin == (5.0, 6.0, 7.0)
    assert_array_equal(turned.direction, np.eye(3))
    assert_array_equal(turned.source_mask, values > 5)
    assert _reoriented(reference, reference) is reference
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    oblique = Image(
        values, (1.0, 2.0, 3.0), (5.0, 6.0, 7.0), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    )
    assert _reoriented(oblique, reference) is oblique


def test_dicom_image_with_nifti_masks_in_other_voxel_orders(tmp_path: "os.PathLike[str]") -> None:
    # A NIfTI mask on the grid of a DICOM series loads onto it without a warning: in the
    # same voxel order (as ITK and 3D Slicer write), with flipped rows (as dcm2niix
    # writes), with swapped and flipped axes, and cropped. The pipeline then gives the
    # features of the same mask in memory.
    import warnings
    from pathlib import Path

    import nibabel as nib

    from pictologics import RadiomicsPipeline

    folder = Path(tmp_path) / "ct"
    folder.mkdir()
    rng = np.random.default_rng(6)
    for k in range(6):
        _write_slice(
            folder / f"{k}.dcm", number=k + 1, position=(10.0 + 1.5 * k, -20.0, 30.0),
            pixels=rng.integers(0, 100, (8, 7)).astype(np.int16),
            orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        )  # fmt: skip
    image = load_image(folder)
    pattern = np.zeros(image.array.shape)
    pattern[1:5, 2:6, 1:4] = 1.0
    pattern[2, 3, 2] = 2.0
    d, s, o = (
        _direction_matrix(image.direction),
        np.asarray(image.spacing),
        np.asarray(image.origin),
    )

    def affine(direction: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> np.ndarray:
        lps = np.eye(4)
        lps[:3, :3], lps[:3, 3] = direction * spacing, origin
        return _ras(lps)

    flipped = d * [1.0, -1.0, 1.0]
    swapped = d[:, [2, 0, 1]] * [-1.0, 1.0, 1.0]
    last_row = d[:, 1] * s[1] * (pattern.shape[1] - 1)
    cases = {
        "same": (pattern, affine(d, s, o)),
        "rows": (pattern[:, ::-1], affine(flipped, s, o + last_row)),
        "axes": (
            np.transpose(pattern, (2, 0, 1))[::-1],
            affine(swapped, s[[2, 0, 1]], o + d[:, 2] * s[2] * (pattern.shape[2] - 1)),
        ),
        "crop": (
            pattern[1:5, 2:6, 1:4][:, ::-1],
            affine(flipped, s, o + d @ (np.array([1, 2, 1]) * s) + d[:, 1] * s[1] * 3),
        ),
    }
    for name, (array, matrix) in cases.items():
        path = Path(tmp_path) / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array.astype(np.uint8), matrix), path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mask = load_image(path, reference_image=image)
        np.testing.assert_array_equal(mask.array, pattern)
        np.testing.assert_allclose(mask.origin, image.origin, atol=1e-9)
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "first_order", [{"step": "extract_features", "params": {"families": ["intensity"]}}]
    )
    in_memory = Image(pattern.astype(np.uint8), image.spacing, image.origin, image.direction)
    from_file = pipeline.run(folder, Path(tmp_path) / "rows.nii.gz", config_names=["first_order"])
    expected = pipeline.run(image, in_memory, config_names=["first_order"])
    assert from_file["first_order"].equals(expected["first_order"])


def test_dicom_folder_with_two_series_needs_a_series_uid(tmp_path: "os.PathLike[str]") -> None:
    # Two reconstructions at the same positions never mix: the error names both series,
    # and series_uid loads one of them.
    from pathlib import Path

    from pictologics.utilities import get_dicom_phases

    folder = Path(tmp_path)
    for k in range(3):
        for uid, value, number, text in (("1.2.3.1", 100, 2, "soft"), ("1.2.3.2", 900, 3, "lung")):
            _write_slice(
                folder / f"{text}{k}.dcm", series=uid, number=k + 1, position=(0.0, 0.0, 1.5 * k),
                pixels=np.full((4, 3), value, np.int16), SeriesNumber=number, SeriesDescription=text,
            )  # fmt: skip
    with pytest.raises(ValueError, match="holds 2 image series") as error:
        load_image(folder)
    assert "1.2.3.1: series 2, CT, 'soft', 3 files" in str(error.value)
    image = load_image(folder, series_uid="1.2.3.2")
    assert image.array.shape == (3, 4, 3) and np.all(image.array == 900.0)
    with pytest.raises(ValueError, match="Series 9.9 is not in"):
        load_image(folder, series_uid="9.9")
    with pytest.raises(ValueError, match="holds 2 image series"):
        get_dicom_phases(folder)
    assert [p.num_slices for p in get_dicom_phases(folder, series_uid="1.2.3.1")] == [3]


def test_dicom_folder_skips_objects_that_are_not_slices(tmp_path: "os.PathLike[str]") -> None:
    # An RTSTRUCT (no pixel data), an RT dose grid and a SEG in the folder of a scan are not
    # its slices; a scout of the same series with another orientation is left out.
    from pathlib import Path

    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder = Path(tmp_path)
    for k in range(3):
        _write_slice(folder / f"{k}.dcm", number=k + 1, position=(0.0, 0.0, 1.5 * k))
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    rt = pydicom.Dataset()
    rt.file_meta = meta
    rt.SOPClassUID, rt.SOPInstanceUID = (
        meta.MediaStorageSOPClassUID,
        meta.MediaStorageSOPInstanceUID,
    )
    rt.SeriesInstanceUID, rt.Modality = "1.2.3.7", "RTSTRUCT"
    rt.save_as(folder / "rtstruct.dcm", enforce_file_format=True)
    _write_slice(folder / "dose.dcm", series="1.2.3.8", sop_class="1.2.840.10008.5.1.4.1.1.481.2")
    _write_slice(folder / "seg.dcm", series="1.2.3.9", sop_class="1.2.840.10008.5.1.4.1.1.66.4")
    assert load_image(folder).array.shape == (3, 4, 3)
    _write_slice(folder / "scout.dcm", number=9, orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
    with pytest.warns(UserWarning, match="Left out 1 of the 4 images"):
        assert load_image(folder).array.shape == (3, 4, 3)
    only_rt = Path(tmp_path) / "rt_only"
    only_rt.mkdir()
    rt.save_as(only_rt / "rtstruct.dcm", enforce_file_format=True)
    with pytest.raises(ValueError, match="Could not read any DICOM files with image data"):
        load_image(only_rt)


def test_dicom_series_warns_on_missing_slices_and_tilt(tmp_path: "os.PathLike[str]") -> None:
    # A missing slice or positions that move sideways (a gantry tilt) give a warning; an
    # even, straight series does not.
    import warnings
    from pathlib import Path

    for name, positions in (
        ("even", [(0.0, 0.0, 1.5 * k) for k in range(4)]),
        ("gap", [(0.0, 0.0, 1.5 * k) for k in (0, 1, 2, 4, 5)]),
        ("tilt", [(0.0, 0.5 * k, 1.5 * k) for k in range(4)]),
    ):
        folder = Path(tmp_path) / name
        folder.mkdir()
        for k, position in enumerate(positions):
            _write_slice(folder / f"{k}.dcm", number=k + 1, position=position)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_image(Path(tmp_path) / "even")
    with pytest.warns(
        UserWarning, match="not evenly spaced: the steps between them go from 1.5 to 3 mm"
    ):
        load_image(Path(tmp_path) / "gap")
    with pytest.warns(UserWarning, match="move sideways by 18.4 degrees"):
        load_image(Path(tmp_path) / "tilt")


def test_colour_dicom_is_an_error(tmp_path: "os.PathLike[str]") -> None:
    # RGB pixel data has no single value per voxel: a clear error for a file and a series.
    from pathlib import Path

    rgb = np.zeros((4, 3, 3), np.uint8)
    folder = Path(tmp_path)
    for k in range(2):
        _write_slice(folder / f"{k}.dcm", number=k + 1, position=(0.0, 0.0, 1.5 * k), pixels=rgb)
    with pytest.raises(ValueError, match=r"colour pixel data \(3 samples per pixel\)"):
        load_image(folder)
    with pytest.raises(ValueError, match=r"colour pixel data \(3 samples per pixel\)"):
        load_image(folder / "0.dcm")


def test_decode_errors_name_the_file_and_the_cause() -> None:
    from pictologics.loader import _decoded_pixels

    dataset = MagicMock()
    type(dataset).pixel_array = PropertyMock(side_effect=RuntimeError("no plugin"))
    dataset.filename = "slice.dcm"
    with pytest.raises(ValueError, match=r"cannot decode slice.dcm \(no plugin\)\.$"):
        _decoded_pixels(dataset)


@pytest.mark.parametrize(
    ("compression", "max_error"),
    [
        ("rle_lossless", 0),
        ("jpeg_lossless_sv1", 0),
        ("jpeg_lossless_p14", 0),
        ("jpegls_lossless", 0),
        ("jpegls_near_lossless", 3),  # the near-lossless bound of the file
        ("jpeg2000_lossless", 0),
        ("jpeg2000", 64),
        ("jpeg_baseline_8bit", 8),
    ],
)
def test_compressed_dicom_series_load(
    tmp_path: "os.PathLike[str]", compression: str, max_error: int
) -> None:
    # Synthetic two-slice CT series, stored once in each compression (JPEG-LS made with
    # pyjpegls, JPEG Lossless with GDCM, baseline JPEG and JPEG 2000 with Pillow, RLE with
    # pydicom): load_image decodes them and applies the rescale (intercept -1024 for the
    # 12-bit series; the 8-bit baseline series has none). Lossless data comes back exactly.
    from pathlib import Path

    folder = Path(tmp_path)
    with np.load(Path(__file__).parent / "data" / "dicom_codecs.npz") as stored:
        for k in range(2):
            (folder / f"{k}.dcm").write_bytes(stored[f"file_{compression}_{k}"].tobytes())
        pixels = np.stack([stored[f"pixels_{compression}_{k}"] for k in range(2)])  # (Z, Y, X)
    expected = np.transpose(pixels.astype(np.float64), (2, 1, 0))
    if compression != "jpeg_baseline_8bit":
        expected -= 1024.0
    image = load_image(folder)
    assert image.array.shape == expected.shape == (20, 16, 2)
    assert np.max(np.abs(image.array - expected)) <= max_error


def test_single_dicom_files_load_as_float64(tmp_path: "os.PathLike[str]") -> None:
    # Without rescale tags (or with slope 1 and intercept 0) a file is float64 too, like
    # the same NIfTI image; apply_rescale=False keeps the stored type. A single volume has
    # no dataset_index 1, and a series no negative one.
    from pathlib import Path

    from pictologics.loader import _load_dicom_series

    pixels = np.arange(12, dtype=np.int16).reshape(4, 3)
    path = Path(tmp_path) / "one.dcm"
    _write_slice(path, pixels=pixels)
    image = load_image(path)
    assert image.array.dtype == np.float64
    np.testing.assert_array_equal(image.array[..., 0], pixels.T)
    assert load_image(path, apply_rescale=False).array.dtype == np.int16
    with pytest.raises(ValueError, match="dataset_index 1 is out of range: .* holds one volume"):
        load_image(path, dataset_index=1)
    with pytest.raises(ValueError, match="dataset_index must be 0 or more"):
        _load_dicom_series(Path(tmp_path), dataset_index=-1)


def _write_multiframe(
    path: "os.PathLike[str]",
    frames: list[tuple[float, "int | None", "float | None", float, float]],
) -> np.ndarray:
    """An enhanced MR file; each frame is (z, temporal index, cardiac phase %, slope,
    intercept), its pixels are 10 * frame + (row, column) pattern. Returns the stored frames."""
    from pathlib import Path

    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    stored = np.stack(
        [np.arange(6, dtype=np.int16).reshape(3, 2) + 10 * k for k in range(len(frames))]
    )
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4.1"  # Enhanced MR Image
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = (
        meta.MediaStorageSOPClassUID,
        meta.MediaStorageSOPInstanceUID,
    )
    ds.Modality, ds.NumberOfFrames = "MR", len(frames)
    ds.Rows, ds.Columns = 3, 2
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    shared = pydicom.Dataset()
    shared.PixelMeasuresSequence = Sequence([pydicom.Dataset()])
    shared.PixelMeasuresSequence[0].PixelSpacing = [0.5, 0.5]
    shared.PixelMeasuresSequence[0].SliceThickness = 1.0
    shared.PlaneOrientationSequence = Sequence([pydicom.Dataset()])
    shared.PlaneOrientationSequence[0].ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    groups = []
    for z, temporal, cardiac, slope, intercept in frames:
        group = pydicom.Dataset()
        group.PlanePositionSequence = Sequence([pydicom.Dataset()])
        group.PlanePositionSequence[0].ImagePositionPatient = [0.0, 0.0, z]
        group.FrameContentSequence = Sequence([pydicom.Dataset()])
        if temporal is not None:
            group.FrameContentSequence[0].TemporalPositionIndex = temporal
        if cardiac is not None:
            group.CardiacSynchronizationSequence = Sequence([pydicom.Dataset()])
            group.CardiacSynchronizationSequence[0].NominalPercentageOfCardiacPhase = cardiac
        group.PixelValueTransformationSequence = Sequence([pydicom.Dataset()])
        group.PixelValueTransformationSequence[0].RescaleSlope = slope
        group.PixelValueTransformationSequence[0].RescaleIntercept = intercept
        groups.append(group)
    ds.PerFrameFunctionalGroupsSequence = Sequence(groups)
    ds.PixelData = stored.tobytes()
    ds.save_as(Path(path), enforce_file_format=True)
    return stored


def test_multiframe_volumes_split_like_series(tmp_path: "os.PathLike[str]") -> None:
    # Frames at repeated positions hold several volumes: dataset_index picks one, by the
    # temporal position index, the cardiac phase, or else the frame order. The frames are
    # sorted and rescaled one by one (float64), in the fused path for large volumes too.
    from pathlib import Path

    folder = Path(tmp_path)
    # stored in the order (t1 z2), (t2 z2), (t1 z0), (t2 z0), (t1 z1), (t2 z1)
    layout = [(2.0, 1), (2.0, 2), (0.0, 1), (0.0, 2), (1.0, 1), (1.0, 2)]
    rescale = [(2.0, -5.0), (1.0, 0.0), (1.0, 3.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.0)]
    cases = {
        "temporal": [(z, t, None, *rescale[k]) for k, (z, t) in enumerate(layout)],
        "cardiac": [(z, None, 30.0 * t, *rescale[k]) for k, (z, t) in enumerate(layout)],
        "order": [(z, None, None, *rescale[k]) for k, (z, _) in enumerate(layout)],
    }
    for name, frames in cases.items():
        path = folder / f"{name}.dcm"
        stored = _write_multiframe(path, frames)
        for index, wanted in ((0, (2, 4, 0)), (1, (3, 5, 1))):
            expected = np.stack(
                [stored[f].astype(np.float64) * rescale[f][0] + rescale[f][1] for f in wanted],
                axis=0,
            ).transpose(2, 1, 0)
            for limit in (1 << 20, 8):
                with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", limit):
                    image = load_image(path, dataset_index=index)
                np.testing.assert_array_equal(image.array.view(np.uint64), expected.view(np.uint64))
                assert image.spacing == (0.5, 0.5, 1.0) and image.origin == (0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="dataset_index 2 is out of range: .* holds 2 volume"):
            load_image(path, dataset_index=2)


def _merge_full(arrays: list[np.ndarray], fill: float, rule: str) -> np.ndarray:
    """The merge of full-size arrays, written out voxel set by voxel set."""
    merged = np.full(arrays[0].shape, fill)
    for full in arrays:
        present, taken = full != fill, merged != fill
        merged[present & ~taken] = full[present & ~taken]
        both = present & taken
        if rule == "max":
            merged[both] = np.maximum(merged[both], full[both])
        elif rule == "min":
            merged[both] = np.minimum(merged[both], full[both])
        elif rule == "last":
            merged[both] = full[both]
    return merged


def test_merging_masks_in_their_boxes_equals_the_full_size_merge(
    tmp_path: "os.PathLike[str]",
) -> None:
    # Box-only merging gives the merge of the repositioned full-size masks, for every
    # conflict rule, with and without relabelling and with fill values 0 and 7. A mask
    # outside the reference is skipped (min_overlap_fraction=0, after a warning). The
    # standard mode (masks on the reference grid) merges the same way.
    from pathlib import Path

    import nibabel as nib

    from pictologics.loader import _position_in_reference

    rng = np.random.default_rng(5)
    reference = Image(np.zeros((20, 18, 12)), (1.0, 1.0, 2.0), (0.0, 0.0, 0.0))
    boxes = [((2, 3, 1), (8, 7, 5)), ((5, 6, 3), (9, 8, 6)), ((15, 12, 8), (10, 10, 6))]
    paths, full_paths = [], []
    for k, (offset, shape) in enumerate(boxes):
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        affine[:3, 3] = np.array(offset) * (1.0, 1.0, 2.0)
        data = rng.integers(0, 4, shape).astype(np.float64) * (k + 1)
        paths.append(Path(tmp_path) / f"box{k}.nii.gz")
        nib.save(nib.Nifti1Image(data, _ras(affine)), paths[-1])
        full = np.zeros(reference.array.shape)
        full[2 + k : 10 + k, 3:9, 1:6] = data[:8, :6, :5] if k < 2 else 0.0
        full_paths.append(Path(tmp_path) / f"full{k}.nii.gz")
        nib.save(nib.Nifti1Image(full, _ras(np.diag([1.0, 1.0, 2.0, 1.0]))), full_paths[-1])
    outside = Path(tmp_path) / "outside.nii.gz"
    away = np.diag([1.0, 1.0, 2.0, 1.0])
    away[:3, 3] = (40.0, 40.0, 80.0)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), _ras(away)), outside)
    for fill in (0.0, 7.0):
        for rule in ("max", "min", "first", "last"):
            for relabel in (False, True):
                with pytest.warns(UserWarning, match="does not overlap"):
                    merged = load_and_merge_images(
                        [*paths, outside], reference_image=reference, reposition_to_reference=True,
                        conflict_resolution=rule, relabel_masks=relabel, fill_value=fill,
                        min_overlap_fraction=0.0,
                    )  # fmt: skip
                arrays = []
                for i, path in enumerate(paths):
                    full = _position_in_reference(
                        load_image(path), reference, fill, min_overlap_fraction=0.0
                    ).array
                    arrays.append(np.where(full != fill, i + 1, fill) if relabel else full)
                np.testing.assert_array_equal(merged.array, _merge_full(arrays, fill, rule))
    for rule in ("max", "min", "first", "last"):
        merged = load_and_merge_images(full_paths, conflict_resolution=rule)
        expected = _merge_full([load_image(p).array for p in full_paths], 0.0, rule)
        np.testing.assert_array_equal(merged.array, expected)


def test_4d_nifti_reads_only_the_requested_volume(tmp_path: "os.PathLike[str]") -> None:
    # One volume of a 4D file is read and scaled as get_fdata scales, bit for bit; the
    # image does not keep the other volumes (its array owns its own memory).
    from pathlib import Path

    import nibabel as nib

    rng = np.random.default_rng(4)
    data = (rng.random((7, 6, 5, 4)) * 300).astype(np.int16)
    nii = nib.Nifti1Image(data, np.diag([0.5, 0.5, 2.0, 1.0]))
    nii.header.set_slope_inter(0.1, -7.3)
    path = Path(tmp_path) / "four.nii"
    nib.save(nii, path)
    full = nib.load(path).get_fdata()
    for k in range(4):
        for limit in (1 << 20, 8):
            with patch("pictologics.loader._ROW_ORDER_MIN_SIZE", limit):
                image = load_image(path, dataset_index=k)
            np.testing.assert_array_equal(image.array.view(np.uint64), full[..., k].view(np.uint64))
            assert image.array.base is None or image.array.base.size == image.array.size
    with pytest.raises(ValueError, match="Dataset index 4 is out of bounds for 4D image"):
        load_image(path, dataset_index=4)


def test_load_image_takes_the_files_of_one_phase(tmp_path: "os.PathLike[str]") -> None:
    # A DicomPhaseInfo from get_dicom_phases, or a list of files, loads that phase and
    # reads only its files; the image equals the one from dataset_index.
    from pathlib import Path

    import pydicom

    from pictologics.utilities import get_dicom_phases

    folder = Path(tmp_path)
    for phase in range(3):
        for k in range(4):
            _write_slice(
                folder / f"p{phase}_{k}.dcm", number=phase * 4 + k + 1, position=(0.0, 0.0, 1.5 * k),
                pixels=np.full((4, 3), 100 * phase + k, np.int16),
                NominalPercentageOfCardiacPhase=30 * phase,
            )  # fmt: skip
    phases = get_dicom_phases(folder)
    assert [p.num_slices for p in phases] == [4, 4, 4]
    with patch("pictologics.loader.pydicom.dcmread", side_effect=pydicom.dcmread) as reads:
        image = load_image(phases[1])
    assert reads.call_count == 4
    np.testing.assert_array_equal(image.array, load_image(folder, dataset_index=1).array)
    files = [str(f) for f in phases[2].file_paths]
    np.testing.assert_array_equal(
        load_image(files).array, load_image(folder, dataset_index=2).array
    )
    reference = load_image(folder)
    assert load_image(phases[0], reference_image=reference).array.shape == reference.array.shape
    with pytest.raises(ValueError, match="No DICOM files found"):
        load_image([])
