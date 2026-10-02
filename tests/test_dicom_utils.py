# ruff: noqa: E402
"""Tests for pictologics.utilities.dicom_utils module."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

from pictologics.utilities.dicom_utils import (
    MULTI_PHASE_TAGS,
    DicomPhaseInfo,
    get_dicom_phases,
    split_dicom_phases,
)


def _image_mock() -> MagicMock:
    """A mock DICOM image dataset: `in` finds its pixel data, as for a real image."""
    dataset = MagicMock()
    dataset.__contains__.return_value = True
    return dataset


def _write_image(path: Path, **tags: object) -> None:
    """A minimal 2 x 2 MR image file with the given tags (synthetic, no patient data)."""
    import numpy as np
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.2.1125.7"
    ds.Rows = ds.Columns = 2
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelData = np.zeros((2, 2), np.uint16).tobytes()
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


class TestSplitDicomPhases(unittest.TestCase):
    """Tests for split_dicom_phases function."""

    def test_empty_metadata(self) -> None:
        """Empty list returns list with empty list."""
        result = split_dicom_phases([])
        self.assertEqual(result, [[]])

    def test_single_file(self) -> None:
        """Single file returns unchanged."""
        meta = [{"file_path": Path("test.dcm")}]
        result = split_dicom_phases(meta)
        self.assertEqual(result, [meta])

    def test_split_by_cardiac_phase(self) -> None:
        """Split by NominalPercentageOfCardiacPhase tag."""
        meta = [
            {"file_path": Path("1.dcm"), "NominalPercentageOfCardiacPhase": 0},
            {"file_path": Path("2.dcm"), "NominalPercentageOfCardiacPhase": 0},
            {"file_path": Path("3.dcm"), "NominalPercentageOfCardiacPhase": 50},
            {"file_path": Path("4.dcm"), "NominalPercentageOfCardiacPhase": 50},
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)  # Phase 0%
        self.assertEqual(len(result[1]), 2)  # Phase 50%

    def test_split_by_temporal_position(self) -> None:
        """Split by TemporalPositionIdentifier tag."""
        meta = [
            {"file_path": Path("1.dcm"), "TemporalPositionIdentifier": 1},
            {"file_path": Path("2.dcm"), "TemporalPositionIdentifier": 2},
            {"file_path": Path("3.dcm"), "TemporalPositionIdentifier": 1},
            {"file_path": Path("4.dcm"), "TemporalPositionIdentifier": 2},
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 2)

    def test_split_by_echo_number(self) -> None:
        """Split by the EchoNumbers tag, also a multi-valued one (VM 1-n)."""
        meta = [
            {"file_path": Path("1.dcm"), "EchoNumbers": 1},
            {"file_path": Path("2.dcm"), "EchoNumbers": 2},
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 2)
        from pydicom.multival import MultiValue

        meta = [
            {"file_path": Path("1.dcm"), "EchoNumbers": MultiValue(int, [1, 2])},
            {"file_path": Path("2.dcm"), "EchoNumbers": [3]},
        ]
        self.assertEqual([len(p) for p in split_dicom_phases(meta)], [1, 1])

    def test_split_by_acquisition_number(self) -> None:
        """Split by AcquisitionNumber tag."""
        meta = [
            {"file_path": Path("1.dcm"), "AcquisitionNumber": 1},
            {"file_path": Path("2.dcm"), "AcquisitionNumber": 2},
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 2)

    def test_split_by_trigger_time(self) -> None:
        """Split by TriggerTime tag."""
        meta = [
            {"file_path": Path("1.dcm"), "TriggerTime": 0.0},
            {"file_path": Path("2.dcm"), "TriggerTime": 100.0},
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 2)

    def test_partial_tag_coverage_no_split(self) -> None:
        """Don't split if tag doesn't cover all files."""
        meta = [
            {"file_path": Path("1.dcm"), "AcquisitionNumber": 1},
            {"file_path": Path("2.dcm"), "AcquisitionNumber": 2},
            {"file_path": Path("3.dcm")},  # Missing tag
        ]
        result = split_dicom_phases(meta)
        # Should not split because tag coverage is incomplete
        self.assertEqual(len(result), 1)

    def test_split_by_spatial_duplication(self) -> None:
        """Split by duplicate ImagePositionPatient (fallback)."""
        meta = [
            {
                "file_path": Path("1.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 1,
            },
            {
                "file_path": Path("2.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 2,
            },
            {
                "file_path": Path("3.dcm"),
                "ImagePositionPatient": (0, 0, 1),
                "InstanceNumber": 3,
            },
            {
                "file_path": Path("4.dcm"),
                "ImagePositionPatient": (0, 0, 1),
                "InstanceNumber": 4,
            },
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 2)

    def test_spatial_with_missing_position(self) -> None:
        """Handle files with no ImagePositionPatient in spatial split."""
        meta = [
            {
                "file_path": Path("1.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 1,
            },
            {
                "file_path": Path("2.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 2,
            },
            {"file_path": Path("3.dcm"), "InstanceNumber": 3},  # Missing position
        ]
        result = split_dicom_phases(meta)
        # Files without position go to phase 0
        self.assertEqual(len(result), 2)

    def test_spatial_with_excess_duplicates(self) -> None:
        """Handle more duplicates than detected phases."""
        meta = [
            {
                "file_path": Path("1.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 1,
            },
            {
                "file_path": Path("2.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 2,
            },
            {
                "file_path": Path("3.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 3,
            },
            {
                "file_path": Path("4.dcm"),
                "ImagePositionPatient": (0, 0, 1),
                "InstanceNumber": 4,
            },
            {
                "file_path": Path("5.dcm"),
                "ImagePositionPatient": (0, 0, 1),
                "InstanceNumber": 5,
            },
        ]
        result = split_dicom_phases(meta)
        # Max duplicates = 3, so 3 phases
        self.assertEqual(len(result), 3)

    def test_spatial_excess_at_single_position(self) -> None:
        """Test edge case where one position has many more duplicates (triggers line 133)."""
        # Create scenario where we abuse the algorithm:
        # Position A has 2 instances, Position B has 2 instances
        # But we add an extra instance at Position A
        meta = [
            {
                "file_path": Path("1.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 1,
            },
            {
                "file_path": Path("2.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 2,
            },
            {
                "file_path": Path("3.dcm"),
                "ImagePositionPatient": (0, 0, 0),
                "InstanceNumber": 3,
            },
            {
                "file_path": Path("4.dcm"),
                "ImagePositionPatient": (0, 0, 1),
                "InstanceNumber": 4,
            },
        ]
        result = split_dicom_phases(meta)
        # 3 phases (max duplicates at position (0,0,0))
        self.assertEqual(len(result), 3)
        # All files should be distributed
        total_files = sum(len(phase) for phase in result)
        self.assertEqual(total_files, 4)

    def test_no_split_single_values(self) -> None:
        """Don't split if all files have same tag value."""
        meta = [
            {"file_path": Path("1.dcm"), "AcquisitionNumber": 1},
            {"file_path": Path("2.dcm"), "AcquisitionNumber": 1},
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 1)

    def test_distinct_positions_are_one_volume(self) -> None:
        """A step-and-shoot CT writes a new AcquisitionNumber per table step; with no
        repeated position the files stay one volume."""
        meta = [
            {
                "file_path": Path(f"{k}.dcm"),
                "ImagePositionPatient": (0.0, 0.0, 1.5 * k),
                "AcquisitionNumber": k // 4 + 1,
                "InstanceNumber": k + 1,
            }
            for k in range(40)
        ]
        self.assertEqual(split_dicom_phases(meta), [meta])

    def test_drifting_trigger_times_split_by_position(self) -> None:
        """Cine MR: 5 phases at 10 positions, the trigger time drifts 0.1 ms per slice.
        50 trigger-time groups are not phases, so the positions split the files."""
        from pictologics.utilities.dicom_utils import _split_tag

        meta = [
            {
                "file_path": k,
                "ImagePositionPatient": (0.0, 0.0, 10.0 * (k % 10)),
                "TriggerTime": 35.0 * (k // 10) + 0.1 * (k % 10),
                "InstanceNumber": k + 1,
            }
            for k in range(50)
        ]
        phases = split_dicom_phases(meta)
        self.assertEqual([len(p) for p in phases], [10] * 5)
        for i, phase in enumerate(phases):
            self.assertEqual({m["file_path"] // 10 for m in phase}, {i})
        self.assertEqual(_split_tag(phases), "spatial")

    def test_tag_groups_need_unique_positions(self) -> None:
        """An AcquisitionNumber that pairs the files at one position does not split
        phases; the duplicate positions do."""
        meta = [
            {
                "file_path": k,
                "ImagePositionPatient": (0.0, 0.0, float(k // 2)),
                "AcquisitionNumber": 1 if k < 2 else 2,
                "InstanceNumber": k + 1,
            }
            for k in range(4)
        ]
        phases = split_dicom_phases(meta)
        self.assertEqual([[m["file_path"] for m in p] for p in phases], [[0, 2], [1, 3]])

    def test_tag_groups_split_repeated_positions(self) -> None:
        """Two cardiac phases at the same positions split by their tag; the tag names
        the phases."""
        from pictologics.utilities.dicom_utils import _split_tag

        meta = [
            {
                "file_path": k,
                "ImagePositionPatient": (0.0, 0.0, float(k % 3)),
                "NominalPercentageOfCardiacPhase": 40 * (k // 3),
                "InstanceNumber": 6 - k,
            }
            for k in range(6)
        ]
        phases = split_dicom_phases(meta)
        self.assertEqual([[m["file_path"] for m in p] for p in phases], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(_split_tag(phases), "NominalPercentageOfCardiacPhase")
        self.assertIsNone(_split_tag(phases[:1]))

    def test_same_layout(self) -> None:
        """Images of one series layout share their size and (within 1e-3) orientation."""
        from pictologics.utilities.dicom_utils import _same_layout

        axial = ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 512, 512)
        self.assertTrue(_same_layout(axial, ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0005), 512, 512)))
        self.assertFalse(_same_layout(axial, ((1.0, 0.0, 0.0, 0.0, 0.0, -1.0), 512, 512)))
        self.assertFalse(_same_layout(axial, (axial[0], 256, 256)))
        self.assertFalse(_same_layout(axial, (None, 512, 512)))
        self.assertTrue(_same_layout((None, 512, 512), (None, 512, 512)))

    def test_series_list_without_a_readable_file(self) -> None:
        """The error listing still names a series whose first file cannot be read."""
        from pictologics.utilities.dicom_utils import _series_list

        listing = _series_list({"1.2.3": [{"file_path": Path("/no/such/file.dcm")}]})
        self.assertEqual(listing, "  1.2.3: series None, None, '', 1 files")

    def test_position_as_list(self) -> None:
        """Handle ImagePositionPatient as list (not tuple)."""
        meta = [
            {
                "file_path": Path("1.dcm"),
                "ImagePositionPatient": [0, 0, 0],
                "InstanceNumber": 1,
            },
            {
                "file_path": Path("2.dcm"),
                "ImagePositionPatient": [0, 0, 0],
                "InstanceNumber": 2,
            },
        ]
        result = split_dicom_phases(meta)
        self.assertEqual(len(result), 2)


class TestGetDicomPhases(unittest.TestCase):
    """Tests for get_dicom_phases function."""

    def test_path_not_exists(self) -> None:
        """Raise FileNotFoundError for non-existent path."""
        with self.assertRaises(FileNotFoundError):
            get_dicom_phases("/non/existent/path")

    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    def test_single_file_not_dicom(self, mock_is_dicom: MagicMock) -> None:
        """Raise ValueError for non-DICOM file."""
        mock_is_dicom.return_value = False
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "not a DICOM file"):
                get_dicom_phases(temp_path)
        finally:
            os.unlink(temp_path)

    @patch("pictologics.utilities.dicom_utils.Path")
    def test_directory_no_dicom_files(self, mock_Path: MagicMock) -> None:
        """Raise ValueError when directory has no DICOM files."""
        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = []

        with self.assertRaisesRegex(ValueError, "No DICOM files found"):
            get_dicom_phases("empty_dir")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    def test_single_phase_series(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Single phase series returns one DicomPhaseInfo."""
        file1 = MagicMock()
        file1.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1]

        mock_is_dicom.return_value = True

        dcm = _image_mock()
        dcm.InstanceNumber = 1
        dcm.ImagePositionPatient = [0, 0, 0]
        mock_dcmread.return_value = dcm

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], DicomPhaseInfo)
        self.assertEqual(result[0].index, 0)
        self.assertEqual(result[0].num_slices, 1)
        self.assertEqual(result[0].label, "Dataset 0")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_cardiac_phase_labels(
        self,
        mock_split: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Cardiac phases get correct labels."""
        file1, file2 = MagicMock(), MagicMock()
        file1.is_file.return_value = True
        file2.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        dcm1 = _image_mock()
        dcm1.InstanceNumber = 1
        dcm1.NominalPercentageOfCardiacPhase = 0
        del dcm1.ImagePositionPatient

        dcm2 = _image_mock()
        dcm2.InstanceNumber = 2
        dcm2.NominalPercentageOfCardiacPhase = 50
        del dcm2.ImagePositionPatient

        mock_dcmread.side_effect = [dcm1, dcm2]

        # Mock split_dicom_phases to return pre-split phases
        mock_split.return_value = [
            [{"file_path": file1, "NominalPercentageOfCardiacPhase": 0}],
            [{"file_path": file2, "NominalPercentageOfCardiacPhase": 50}],
        ]

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "Phase 0%")
        self.assertEqual(result[1].label, "Phase 50%")
        self.assertEqual(result[0].split_tag, "NominalPercentageOfCardiacPhase")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_temporal_position_labels(
        self,
        mock_split: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Temporal positions get correct labels."""
        file1, file2 = MagicMock(), MagicMock()
        file1.is_file.return_value = True
        file2.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        dcm1 = _image_mock()
        dcm1.InstanceNumber = 1
        dcm1.TemporalPositionIdentifier = 1
        del dcm1.ImagePositionPatient

        dcm2 = _image_mock()
        dcm2.InstanceNumber = 2
        dcm2.TemporalPositionIdentifier = 2
        del dcm2.ImagePositionPatient

        mock_dcmread.side_effect = [dcm1, dcm2]

        mock_split.return_value = [
            [{"file_path": file1, "TemporalPositionIdentifier": 1}],
            [{"file_path": file2, "TemporalPositionIdentifier": 2}],
        ]

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "Temporal 1")
        self.assertEqual(result[1].label, "Temporal 2")
        self.assertEqual(result[0].split_tag, "TemporalPositionIdentifier")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_echo_number_labels(
        self,
        mock_split: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Echo numbers get correct labels."""
        file1, file2 = MagicMock(), MagicMock()
        file1.is_file.return_value = True
        file2.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        dcm1 = _image_mock()
        dcm1.InstanceNumber = 1
        dcm1.EchoNumbers = 1
        del dcm1.ImagePositionPatient

        dcm2 = _image_mock()
        dcm2.InstanceNumber = 2
        dcm2.EchoNumbers = 2
        del dcm2.ImagePositionPatient

        mock_dcmread.side_effect = [dcm1, dcm2]

        mock_split.return_value = [
            [{"file_path": file1, "EchoNumbers": 1}],
            [{"file_path": file2, "EchoNumbers": 2}],
        ]

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "Echo 1")
        self.assertEqual(result[1].label, "Echo 2")
        self.assertEqual(result[0].split_tag, "EchoNumbers")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_acquisition_number_labels(
        self,
        mock_split: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Acquisition numbers get correct labels."""
        file1, file2 = MagicMock(), MagicMock()
        file1.is_file.return_value = True
        file2.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        dcm1 = _image_mock()
        dcm1.InstanceNumber = 1
        dcm1.AcquisitionNumber = 1
        del dcm1.ImagePositionPatient

        dcm2 = _image_mock()
        dcm2.InstanceNumber = 2
        dcm2.AcquisitionNumber = 2
        del dcm2.ImagePositionPatient

        mock_dcmread.side_effect = [dcm1, dcm2]

        mock_split.return_value = [
            [{"file_path": file1, "AcquisitionNumber": 1}],
            [{"file_path": file2, "AcquisitionNumber": 2}],
        ]

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "Acquisition 1")
        self.assertEqual(result[1].label, "Acquisition 2")
        self.assertEqual(result[0].split_tag, "AcquisitionNumber")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_trigger_time_labels(
        self,
        mock_split: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Trigger times get correct labels."""
        file1, file2 = MagicMock(), MagicMock()
        file1.is_file.return_value = True
        file2.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        dcm1 = _image_mock()
        dcm1.InstanceNumber = 1
        dcm1.TriggerTime = 0.0
        del dcm1.ImagePositionPatient

        dcm2 = _image_mock()
        dcm2.InstanceNumber = 2
        dcm2.TriggerTime = 100.0
        del dcm2.ImagePositionPatient

        mock_dcmread.side_effect = [dcm1, dcm2]

        mock_split.return_value = [
            [{"file_path": file1, "TriggerTime": 0.0}],
            [{"file_path": file2, "TriggerTime": 100.0}],
        ]

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "Trigger 0.0ms")
        self.assertEqual(result[1].label, "Trigger 100.0ms")
        self.assertEqual(result[0].split_tag, "TriggerTime")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    @patch("pictologics.utilities.dicom_utils.split_dicom_phases")
    def test_spatial_split_labels(
        self,
        mock_split: MagicMock,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Spatial duplicates get 'Volume N' labels."""
        file1, file2 = MagicMock(), MagicMock()
        file1.is_file.return_value = True
        file2.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        dcm1 = _image_mock()
        dcm1.InstanceNumber = 1
        dcm1.ImagePositionPatient = [0, 0, 0]

        dcm2 = _image_mock()
        dcm2.InstanceNumber = 2
        dcm2.ImagePositionPatient = [0, 0, 0]  # Same position = duplicate

        mock_dcmread.side_effect = [dcm1, dcm2]

        # Spatial split - no multi-phase tags in metadata
        mock_split.return_value = [
            [{"file_path": file1, "ImagePositionPatient": (0, 0, 0)}],
            [{"file_path": file2, "ImagePositionPatient": (0, 0, 0)}],
        ]

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, "Volume 1")
        self.assertEqual(result[1].label, "Volume 2")
        self.assertEqual(result[0].split_tag, "spatial")

    def test_recursive_search(self) -> None:
        """Recursive search uses the folder with the most DICOM files, as load_image does."""
        from pictologics import load_image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, count in (("a", 2), ("b", 3)):
                (root / name).mkdir()
                for k in range(count):
                    _write_image(
                        root / name / f"{k}.dcm",
                        InstanceNumber=k + 1,
                        ImagePositionPatient=[0.0, 0.0, float(k)],
                    )
            result = get_dicom_phases(root, recursive=True)
            self.assertEqual([p.num_slices for p in result], [3])
            self.assertEqual({f.parent.name for f in result[0].file_paths}, {"b"})
            self.assertEqual(load_image(root, recursive=True).array.shape, (2, 2, 3))

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    def test_single_dicom_file(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Handle single DICOM file path (not directory)."""
        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = True

        mock_is_dicom.return_value = True

        dcm = _image_mock()
        dcm.InstanceNumber = 1
        dcm.ImagePositionPatient = [0, 0, 0]
        mock_dcmread.return_value = dcm

        result = get_dicom_phases("test.dcm")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].num_slices, 1)

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    def test_dcmread_exception_skipped(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Files that fail to read are skipped."""
        file1, file2 = MagicMock(), MagicMock()
        file1.is_file.return_value = True
        file2.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1, file2]

        mock_is_dicom.return_value = True

        dcm = _image_mock()
        dcm.InstanceNumber = 1
        dcm.ImagePositionPatient = [0, 0, 0]

        # First file fails, second succeeds
        mock_dcmread.side_effect = [Exception("Read error"), dcm]

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].num_slices, 1)

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    def test_all_files_fail_to_read(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Raise ValueError if all files fail to read."""
        file1 = MagicMock()
        file1.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1]

        mock_is_dicom.return_value = True
        mock_dcmread.side_effect = Exception("Read error")

        with self.assertRaisesRegex(ValueError, "Could not read any DICOM files"):
            get_dicom_phases("test_dir")

    @patch("pictologics.utilities.dicom_utils.Path")
    @patch("pictologics.utilities.dicom_utils.pydicom.misc.is_dicom")
    @patch("pictologics.utilities.dicom_utils.pydicom.dcmread")
    def test_missing_position_in_metadata(
        self,
        mock_dcmread: MagicMock,
        mock_is_dicom: MagicMock,
        mock_Path: MagicMock,
    ) -> None:
        """Handle missing ImagePositionPatient gracefully."""
        file1 = MagicMock()
        file1.is_file.return_value = True

        mock_path_obj = mock_Path.return_value
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path_obj.iterdir.return_value = [file1]

        mock_is_dicom.return_value = True

        dcm = _image_mock()
        dcm.InstanceNumber = 1
        del dcm.ImagePositionPatient  # Missing

        mock_dcmread.return_value = dcm

        result = get_dicom_phases("test_dir")

        self.assertEqual(len(result), 1)


class TestDicomPhaseInfo(unittest.TestCase):
    """Tests for DicomPhaseInfo dataclass."""

    def test_dataclass_creation(self) -> None:
        """DicomPhaseInfo can be created with all fields."""
        info = DicomPhaseInfo(
            index=0,
            num_slices=64,
            file_paths=[Path("1.dcm"), Path("2.dcm")],
            label="Phase 0%",
            split_tag="NominalPercentageOfCardiacPhase",
            split_value=0,
        )
        self.assertEqual(info.index, 0)
        self.assertEqual(info.num_slices, 64)
        self.assertEqual(len(info.file_paths), 2)
        self.assertEqual(info.label, "Phase 0%")

    def test_dataclass_defaults(self) -> None:
        """DicomPhaseInfo has correct defaults."""
        info = DicomPhaseInfo(
            index=0,
            num_slices=10,
            file_paths=[],
        )
        self.assertIsNone(info.label)
        self.assertIsNone(info.split_tag)
        self.assertIsNone(info.split_value)


class TestMultiPhaseTags(unittest.TestCase):
    """Tests for MULTI_PHASE_TAGS constant."""

    def test_tags_defined(self) -> None:
        """MULTI_PHASE_TAGS contains expected tags."""
        self.assertIn("NominalPercentageOfCardiacPhase", MULTI_PHASE_TAGS)
        self.assertIn("TemporalPositionIdentifier", MULTI_PHASE_TAGS)
        self.assertIn("TriggerTime", MULTI_PHASE_TAGS)
        self.assertIn("AcquisitionNumber", MULTI_PHASE_TAGS)
        self.assertIn("EchoNumbers", MULTI_PHASE_TAGS)
        self.assertEqual(len(MULTI_PHASE_TAGS), 5)


class TestHeaderWorkerPool(unittest.TestCase):
    def test_workers_skip_the_warmup(self) -> None:
        # The workers see PICTOLOGICS_DISABLE_WARMUP=1; the previous value comes back.
        from pictologics.utilities.dicom_utils import worker_pool

        key = "PICTOLOGICS_DISABLE_WARMUP"
        for previous in (None, "0"):
            with patch.dict(os.environ, {} if previous is None else {key: previous}):
                if previous is None:
                    os.environ.pop(key, None)
                with worker_pool(1) as pool:
                    self.assertEqual(pool.submit(os.getenv, key).result(), "1")
                self.assertEqual(os.environ.get(key), previous)

    def test_a_forkserver_default_gives_a_spawn_pool(self) -> None:
        # A forkserver would keep the environment of its first start; the pool spawns
        # instead. Other start methods stay as they are.
        import multiprocessing

        from pictologics.utilities import dicom_utils

        for method, spawn, expected in (
            ("forkserver", False, "spawn"),
            ("spawn", False, "spawn"),
            ("fork", False, "fork"),
            ("fork", True, "spawn"),
        ):
            with (
                patch.object(multiprocessing, "get_start_method", return_value=method),
                patch.object(dicom_utils, "ProcessPoolExecutor") as pool,
            ):
                with dicom_utils.worker_pool(2, spawn=spawn):
                    pass
            self.assertEqual(pool.call_args.kwargs["mp_context"].get_start_method(), expected)


def test_echo_numbers_split_real_files(tmp_path: Path) -> None:
    """Two echoes of real files split by their EchoNumbers tag. The instance numbers
    alternate, so the duplicate-position fallback would mix the echoes."""
    for k, (z, echo) in enumerate(((0.0, 1), (0.0, 2), (1.0, 2), (1.0, 1))):
        _write_image(
            tmp_path / f"{k}.dcm", InstanceNumber=k + 1, EchoNumbers=echo,
            ImagePositionPatient=[0.0, 0.0, z],
        )  # fmt: skip
    phases = get_dicom_phases(str(tmp_path))
    assert [p.label for p in phases] == ["Echo 1", "Echo 2"]
    assert [sorted(f.name for f in p.file_paths) for p in phases] == [
        ["0.dcm", "3.dcm"],
        ["1.dcm", "2.dcm"],
    ]


if __name__ == "__main__":
    unittest.main()
