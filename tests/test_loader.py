import warnings

# Suppress "NumPy module was reloaded" warning which can happen in test setups
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import os

import numpy as np

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from pictologics.loader import (
    Image,
    _ensure_3d,
    _find_best_dicom_series_dir,
    _load_dicom_file,
    _load_dicom_series,
    _load_nifti,
    create_full_mask,
    load_and_merge_images,
    load_image,
)


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
    def test_find_best_dicom_series_dir_with_oserror(
        self, mock_is_dicom: MagicMock
    ) -> None:
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
    def test_find_best_dicom_series_dir_none_found(
        self, mock_is_dicom: MagicMock
    ) -> None:
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
        mock_load_series.assert_called_once_with(mock_path_obj)

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader._load_nifti")
    def test_load_image_nifti(
        self, mock_load_nifti: MagicMock, mock_Path_cls: MagicMock
    ) -> None:
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
        mock_load_dcm.assert_called_once_with("image.dcm")

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
        mock_load_dcm.assert_called_once_with("image.unknown")

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
        np.testing.assert_array_equal(img.direction, np.eye(3))
        self.assertEqual(img.modality, "Nifti")

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
    def test_load_nifti_failure(self, mock_nib_load: MagicMock) -> None:
        mock_nib_load.side_effect = Exception("Corrupt file")
        with self.assertRaises(ValueError):
            _load_nifti("bad.nii")

    # --- _load_dicom_series Tests ---
    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_success(
        self,
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

        # Create mock slices
        slice1 = MagicMock()
        slice1.pixel_array = np.zeros((512, 512))  # Y, X
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # Identity
        slice1.PixelSpacing = [0.5, 0.5]  # Row (Y), Col (X)
        slice1.SliceThickness = 1.0
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = -1024.0
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = -1024.0
        slice1.Modality = "CT"
        del slice1.SpacingBetweenSlices  # Ensure falls back to SliceThickness

        slice2 = MagicMock()
        slice2.pixel_array = np.zeros((512, 512))
        slice2.ImagePositionPatient = [0.0, 0.0, 1.0]  # Z=1
        slice2.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        slice2.PixelSpacing = [0.5, 0.5]
        slice2.SliceThickness = 1.0
        slice2.RescaleSlope = 1.0
        slice2.RescaleIntercept = -1024.0
        slice2.Modality = "CT"

        # Return in reverse order to test sorting
        mock_dcmread.side_effect = [slice2, slice1]

        img = _load_dicom_series("dicom_dir")

        # Check shape: (X, Y, Z) -> (512, 512, 2)
        self.assertEqual(img.array.shape, (512, 512, 2))
        self.assertEqual(img.spacing, (0.5, 0.5, 1.0))
        self.assertEqual(img.origin, (0.0, 0.0, 0.0))
        self.assertEqual(img.modality, "CT")
        self.assertEqual(img.array[0, 0, 0], -1024.0)

    @patch("pictologics.loader.Path")
    def test_load_dicom_series_no_files(self, mock_Path_cls: MagicMock) -> None:
        mock_path_obj = mock_Path_cls.return_value
        mock_path_obj.iterdir.return_value = []  # Empty dir or no dicoms
        with self.assertRaises(ValueError):
            _load_dicom_series("empty_dir")

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_fallback_sorting(
        self,
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

        # Slices without ImagePositionPatient/Orientation
        slice1 = MagicMock()
        slice1.pixel_array = np.zeros((10, 10))
        del slice1.ImagePositionPatient
        del slice1.ImageOrientationPatient
        slice1.InstanceNumber = 1
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = 0.0

        slice2 = MagicMock()
        slice2.pixel_array = np.zeros((10, 10))
        del slice2.ImagePositionPatient
        del slice2.ImageOrientationPatient
        slice2.InstanceNumber = 2
        slice2.RescaleSlope = 1.0
        slice2.RescaleIntercept = 0.0

        mock_dcmread.side_effect = [slice2, slice1]

        img = _load_dicom_series("dicom_dir")
        self.assertEqual(img.array.shape, (10, 10, 2))
        self.assertEqual(img.spacing, (1.0, 1.0, 1.0))

    @patch("pictologics.loader.Path")
    @patch("pictologics.loader.pydicom.misc.is_dicom")
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_series_spacing_calculation(
        self,
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

        slice1 = MagicMock()
        slice1.pixel_array = np.zeros((10, 10))
        slice1.ImagePositionPatient = [0.0, 0.0, 0.0]
        slice1.PixelSpacing = [0.5, 0.5]
        slice1.RescaleSlope = 1.0
        slice1.RescaleIntercept = 0.0
        del slice1.SliceThickness
        del slice1.SpacingBetweenSlices

        slice2 = MagicMock()
        slice2.pixel_array = np.zeros((10, 10))
        slice2.ImagePositionPatient = [0.0, 0.0, 2.0]  # 2mm diff
        slice2.PixelSpacing = [0.5, 0.5]
        slice2.RescaleSlope = 1.0
        slice2.RescaleIntercept = 0.0

        mock_dcmread.side_effect = [slice1, slice2]

        img = _load_dicom_series("dicom_dir")
        self.assertEqual(img.spacing, (0.5, 0.5, 2.0))

    # --- _load_dicom_file Tests ---
    @patch("pictologics.loader.pydicom.dcmread")
    def test_load_dicom_file_success(self, mock_dcmread: MagicMock) -> None:
        mock_dcm = MagicMock()
        mock_dcm.pixel_array = np.zeros((100, 100))
        mock_dcm.PixelSpacing = [0.5, 0.5]
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

        slice1 = MagicMock()
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

        slice1 = MagicMock()
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

        slice1 = MagicMock()
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

        with self.assertRaisesRegex(ValueError, "Error reading DICOM files"):
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

        slice1 = MagicMock()
        # Simulate missing attributes
        del slice1.ImagePositionPatient
        del slice1.PixelSpacing

        mock_dcmread.return_value = slice1

        img = _load_dicom_series("dicom_dir")
        # Should fallback to defaults
        self.assertEqual(img.spacing, (1.0, 1.0, 1.0))
        self.assertEqual(img.origin, (0.0, 0.0, 0.0))

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
    def test_load_and_merge_conflict_resolution(
        self, mock_load_image: MagicMock
    ) -> None:
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
    def test_load_and_merge_geometry_mismatch_spacing(
        self, mock_load: MagicMock
    ) -> None:
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
    def test_load_and_merge_geometry_mismatch_origin(
        self, mock_load: MagicMock
    ) -> None:
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
    def test_load_and_merge_geometry_mismatch_direction(
        self, mock_load: MagicMock
    ) -> None:
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
    def test_load_and_merge_validation_against_reference(
        self, mock_load: MagicMock
    ) -> None:
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


if __name__ == "__main__":
    unittest.main()
