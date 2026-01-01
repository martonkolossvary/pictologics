"""
Test suite for DICOM Database Utilities.

Tests cover all functionality including:
- DICOM file discovery and scanning
- Hierarchy construction (Patient/Study/Series/Instance)
- Completeness validation
- Multi-level DataFrame exports
- CSV and JSON export
- Common metadata extraction
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

# Disable JIT warmup for tests to prevent NumPy reload warning and speed up collection
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"


import pandas as pd
import pytest

from pictologics.utilities import (
    DicomDatabase,
    DicomInstance,
    DicomPatient,
    DicomSeries,
    DicomStudy,
)
from pictologics.utilities.dicom_database import (
    _build_hierarchy,
    _combine_datetime,
    _extract_all_metadata,
    _extract_common_metadata,
    _extract_single_file_metadata,
    _get_num_workers,
    _get_tag_value,
    _is_dicom_file,
    _scan_dicom_files,
    _sort_hierarchy,
    _values_equal,
)

# ============================================================================
# Synthetic DICOM File Creation (used for all tests)
# ============================================================================


# ============================================================================
# Synthetic DICOM File Creation (for CI testing without real data)
# ============================================================================


def create_synthetic_dicom(
    file_path: Path,
    patient_id: str = "SYNTH_PATIENT_001",
    study_uid: str = "1.2.3.4.5.6.7.8.9.1",
    series_uid: str = "1.2.3.4.5.6.7.8.9.2",
    sop_uid: str = "1.2.3.4.5.6.7.8.9.3",
    instance_number: int = 1,
    image_position: Optional[tuple] = (0.0, 0.0, 0.0),
    slice_location: float = 0.0,
    **kwargs: Any,
) -> Path:
    """Create a minimal synthetic DICOM file for testing.

    No real patient information is included - all identifiers are synthetic.
    """
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian

    # Create minimal file meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create dataset
    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient level (synthetic - no real patient data)
    ds.PatientID = patient_id
    ds.PatientName = "SYNTHETIC^TEST^PATIENT"
    ds.PatientBirthDate = "19000101"
    ds.PatientSex = "O"  # Other

    # Study level
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000"
    ds.StudyDescription = "Synthetic Test Study"

    # Series level
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = "Synthetic Test Series"
    ds.Modality = "CT"
    ds.FrameOfReferenceUID = "1.2.3.4.5.6.7.8.9.4"

    # Instance level
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = instance_number
    ds.SliceLocation = slice_location
    ds.SliceLocation = slice_location
    if image_position is not None:
        ds.ImagePositionPatient = list(image_position)
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SliceThickness = 2.5
    ds.PixelSpacing = [0.5, 0.5]

    # Minimal image data
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = b"\x00" * 32  # 4x4 pixels * 2 bytes

    # Set any additional extra tags
    for key, value in kwargs.items():
        setattr(ds, key, value)

    # Save
    ds.save_as(str(file_path))
    return file_path


@pytest.fixture
def synthetic_dicom_dir(tmp_path: Path) -> Path:
    """Create a directory with synthetic DICOM files for testing.

    Creates a structure with:
    - 1 patient
    - 1 study
    - 1 series with 5 slices
    """
    dicom_dir = tmp_path / "synthetic_dicom"
    dicom_dir.mkdir()

    for i in range(5):
        create_synthetic_dicom(
            dicom_dir / f"slice_{i:03d}.dcm",
            sop_uid=f"1.2.3.4.5.6.7.8.9.{100+i}",
            instance_number=i + 1,
            image_position=(0.0, 0.0, float(i * 2.5)),
            slice_location=float(i * 2.5),
        )

    return dicom_dir


@pytest.fixture
def synthetic_multi_patient_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple patients for testing sorting and hierarchy."""
    dicom_dir = tmp_path / "multi_patient"
    dicom_dir.mkdir()

    # Patient 1, Study 1, Series 1 (3 slices)
    for i in range(3):
        create_synthetic_dicom(
            dicom_dir / f"p1_s1_slice_{i:03d}.dcm",
            patient_id="SYNTH_PATIENT_001",
            study_uid="1.2.3.4.5.1.1",
            series_uid="1.2.3.4.5.1.1.1",
            sop_uid=f"1.2.3.4.5.1.1.1.{i+1}",
            instance_number=i + 1,
            image_position=(0.0, 0.0, float(i * 2.5)),
            slice_location=float(i * 2.5),
        )

    # Patient 2, Study 1, Series 1 (2 slices)
    for i in range(2):
        create_synthetic_dicom(
            dicom_dir / f"p2_s1_slice_{i:03d}.dcm",
            patient_id="SYNTH_PATIENT_002",
            study_uid="1.2.3.4.5.2.1",
            series_uid="1.2.3.4.5.2.1.1",
            sop_uid=f"1.2.3.4.5.2.1.1.{i+1}",
            instance_number=i + 1,
            image_position=(0.0, 0.0, float(i * 3.0)),
            slice_location=float(i * 3.0),
        )

    return dicom_dir


# ============================================================================
# Test DicomInstance
# ============================================================================


class TestDicomInstance:
    """Tests for DicomInstance dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating instance with minimal attributes."""
        instance = DicomInstance(
            sop_instance_uid="1.2.3.4.5",
            file_path=Path("/test/file.dcm"),
        )
        assert instance.sop_instance_uid == "1.2.3.4.5"
        assert instance.file_path == Path("/test/file.dcm")
        assert instance.instance_number is None
        assert instance.image_position_patient is None
        assert instance.metadata == {}

    def test_creation_full(self) -> None:
        """Test creating instance with all attributes."""
        instance = DicomInstance(
            sop_instance_uid="1.2.3.4.5",
            file_path=Path("/test/file.dcm"),
            instance_number=42,
            image_position_patient=(1.0, 2.0, 3.0),
            image_orientation_patient=(1, 0, 0, 0, 1, 0),
            slice_location=150.5,
            acquisition_datetime="20231231T120000",
            projection_score=100.5,
            metadata={"SliceThickness": 2.5},
        )
        assert instance.instance_number == 42
        assert instance.image_position_patient == (1.0, 2.0, 3.0)
        assert instance.projection_score == 100.5


# ============================================================================
# Test DicomSeries
# ============================================================================


class TestDicomSeries:
    """Tests for DicomSeries dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating series with minimal attributes."""
        series = DicomSeries(series_instance_uid="1.2.3.4.5.6")
        assert series.series_instance_uid == "1.2.3.4.5.6"
        assert series.instances == []
        assert series.common_metadata == {}

    def test_get_instance_uids(self) -> None:
        """Test getting list of instance UIDs."""
        series = DicomSeries(
            series_instance_uid="1.2.3",
            instances=[
                DicomInstance("uid1", Path("/a.dcm")),
                DicomInstance("uid2", Path("/b.dcm")),
            ],
        )
        uids = series.get_instance_uids()
        assert uids == ["uid1", "uid2"]

    def test_get_file_paths(self) -> None:
        """Test getting list of file paths."""
        series = DicomSeries(
            series_instance_uid="1.2.3",
            instances=[
                DicomInstance("uid1", Path("/a.dcm")),
                DicomInstance("uid2", Path("/b.dcm")),
            ],
        )
        paths = series.get_file_paths()
        assert paths == ["/a.dcm", "/b.dcm"]

    def test_get_sorted_instances_by_projection(self) -> None:
        """Test sorting instances by projection score."""
        series = DicomSeries(
            series_instance_uid="1.2.3",
            instances=[
                DicomInstance("uid1", Path("/a.dcm"), projection_score=30.0),
                DicomInstance("uid2", Path("/b.dcm"), projection_score=10.0),
                DicomInstance("uid3", Path("/c.dcm"), projection_score=20.0),
            ],
        )
        sorted_instances = series.get_sorted_instances()
        assert [i.sop_instance_uid for i in sorted_instances] == [
            "uid2",
            "uid3",
            "uid1",
        ]

    def test_get_sorted_instances_by_instance_number(self) -> None:
        """Test fallback sorting by instance number."""
        series = DicomSeries(
            series_instance_uid="1.2.3",
            instances=[
                DicomInstance("uid1", Path("/a.dcm"), instance_number=3),
                DicomInstance("uid2", Path("/b.dcm"), instance_number=1),
                DicomInstance("uid3", Path("/c.dcm"), instance_number=2),
            ],
        )
        sorted_instances = series.get_sorted_instances()
        assert [i.sop_instance_uid for i in sorted_instances] == [
            "uid2",
            "uid3",
            "uid1",
        ]

    def test_check_completeness_single_instance(self) -> None:
        """Test completeness check with single instance."""
        series = DicomSeries(
            series_instance_uid="1.2.3",
            instances=[DicomInstance("uid1", Path("/a.dcm"))],
        )
        result = series.check_completeness()
        assert result["total_instances"] == 1
        assert "fewer than 2 instances" in result["warnings"][0]

    def test_check_completeness_complete_series(self) -> None:
        """Test completeness check with complete series."""
        instances = [
            DicomInstance("uid1", Path("/a.dcm"), projection_score=float(i * 2.5))
            for i in range(10)
        ]
        series = DicomSeries(series_instance_uid="1.2.3", instances=instances)
        result = series.check_completeness()
        assert result["is_complete"] is True
        assert result["has_gaps"] is False
        assert result["total_instances"] == 10
        assert abs(result["spacing_mm"] - 2.5) < 0.01

    def test_check_completeness_with_gap(self) -> None:
        """Test completeness check detects gaps."""
        # Create series with a gap (missing instance at position 5)
        positions = [0, 2.5, 5.0, 7.5, 15.0, 17.5, 20.0]  # Gap between 7.5 and 15.0
        instances = [
            DicomInstance(f"uid{i}", Path(f"/{i}.dcm"), projection_score=p)
            for i, p in enumerate(positions)
        ]
        series = DicomSeries(series_instance_uid="1.2.3", instances=instances)
        result = series.check_completeness()
        assert result["has_gaps"] is True
        assert len(result["gap_indices"]) > 0

    def test_check_completeness_fallback_to_instance_number(self) -> None:
        """Test completeness falls back to instance number when no spatial data."""
        instances = [
            DicomInstance(f"uid{i}", Path(f"/{i}.dcm"), instance_number=i)
            for i in [1, 2, 4, 5]  # Missing instance 3
        ]
        series = DicomSeries(series_instance_uid="1.2.3", instances=instances)
        result = series.check_completeness()
        assert result["is_complete"] is False
        assert result["has_gaps"] is True
        assert 3 in result["gap_indices"]


# ============================================================================
# Test DicomStudy
# ============================================================================


class TestDicomStudy:
    """Tests for DicomStudy dataclass."""

    def test_get_instance_uids(self) -> None:
        """Test collecting UIDs across all series."""
        study = DicomStudy(
            study_instance_uid="1.2.3",
            series=[
                DicomSeries(
                    "series1",
                    instances=[DicomInstance("uid1", Path("/a.dcm"))],
                ),
                DicomSeries(
                    "series2",
                    instances=[
                        DicomInstance("uid2", Path("/b.dcm")),
                        DicomInstance("uid3", Path("/c.dcm")),
                    ],
                ),
            ],
        )
        uids = study.get_instance_uids()
        assert uids == ["uid1", "uid2", "uid3"]

    def test_get_file_paths(self) -> None:
        """Test collecting file paths across all series."""
        study = DicomStudy(
            study_instance_uid="1.2.3",
            series=[
                DicomSeries(
                    "series1",
                    instances=[DicomInstance("uid1", Path("/a.dcm"))],
                ),
                DicomSeries(
                    "series2",
                    instances=[DicomInstance("uid2", Path("/b.dcm"))],
                ),
            ],
        )
        paths = study.get_file_paths()
        assert len(paths) == 2


# ============================================================================
# Test DicomPatient
# ============================================================================


class TestDicomPatient:
    """Tests for DicomPatient dataclass."""

    def test_get_instance_uids(self) -> None:
        """Test collecting UIDs across all studies."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    "study1",
                    series=[
                        DicomSeries(
                            "series1",
                            instances=[DicomInstance("uid1", Path("/a.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        uids = patient.get_instance_uids()
        assert uids == ["uid1"]

    def test_get_file_paths(self) -> None:
        """Test collecting file paths across all studies."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    "study1",
                    series=[
                        DicomSeries(
                            "series1",
                            instances=[
                                DicomInstance("uid1", Path("/a.dcm")),
                                DicomInstance("uid2", Path("/b.dcm")),
                            ],
                        ),
                    ],
                ),
                DicomStudy(
                    "study2",
                    series=[
                        DicomSeries(
                            "series2",
                            instances=[DicomInstance("uid3", Path("/c.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        paths = patient.get_file_paths()
        assert len(paths) == 3
        assert "/a.dcm" in paths
        assert "/c.dcm" in paths


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Tests for private helper functions."""

    def test_values_equal_basic(self) -> None:
        """Test basic value comparison."""
        assert _values_equal(1, 1) is True
        assert _values_equal("a", "a") is True
        assert _values_equal(1, 2) is False

    def test_values_equal_none(self) -> None:
        """Test None handling."""
        assert _values_equal(None, None) is True
        assert _values_equal(None, 1) is False
        assert _values_equal(1, None) is False

    def test_values_equal_lists(self) -> None:
        """Test list/tuple comparison."""
        assert _values_equal([1, 2], [1, 2]) is True
        assert _values_equal((1, 2), [1, 2]) is True
        assert _values_equal([1, 2], [1, 3]) is False

    def test_combine_datetime(self) -> None:
        """Test datetime combination."""
        assert _combine_datetime("20231231", "120000") == "20231231T120000"
        assert _combine_datetime("20231231", None) == "20231231"
        assert _combine_datetime(None, "120000") is None
        assert _combine_datetime(None, None) is None

    def test_extract_common_metadata_empty(self) -> None:
        """Test with empty list."""
        assert _extract_common_metadata([]) == {}

    def test_extract_common_metadata(self) -> None:
        """Test extracting common metadata."""
        metadata_list = [
            {"PatientID": "P001", "Modality": "CT", "SliceThickness": 2.5},
            {"PatientID": "P001", "Modality": "CT", "SliceThickness": 2.5},
        ]
        common = _extract_common_metadata(metadata_list)
        assert common.get("SliceThickness") == 2.5

    def test_extract_common_metadata_different_values(self) -> None:
        """Test that different values are not included."""
        metadata_list = [
            {"PatientID": "P001", "InstanceNumber": 1},
            {"PatientID": "P001", "InstanceNumber": 2},
        ]
        common = _extract_common_metadata(metadata_list)
        # InstanceNumber should not be in common (it's in skip_tags)
        assert "InstanceNumber" not in common

    def test_extract_common_metadata_with_none_values(self) -> None:
        """Test that None values are excluded from common metadata."""
        metadata_list = [
            {"PatientID": "P001", "CustomTag": None, "ValidTag": "value"},
            {"PatientID": "P001", "CustomTag": None, "ValidTag": "value"},
        ]
        common = _extract_common_metadata(metadata_list)
        # None values should be excluded (line 1104-1105)
        assert "CustomTag" not in common
        assert common.get("ValidTag") == "value"

    def test_values_equal_with_exception(self) -> None:
        """Test _values_equal handles comparison exceptions."""

        # Create objects that will raise exception when compared
        class BadCompare:
            def __eq__(self, other: Any) -> bool:
                raise ValueError("Comparison failed")

        # Should return False when exception is raised (lines 1125-1126)
        assert _values_equal(BadCompare(), "test") is False

    def test_get_tag_value(self) -> None:
        """Test safe tag extraction."""
        mock_dcm = MagicMock()
        mock_dcm.PatientID = "P001"
        # For missing attributes, configure mock to return None via getattr
        mock_dcm.configure_mock(**{"NonExistent": None})

        assert _get_tag_value(mock_dcm, "PatientID") == "P001"
        # When value is None, should return default
        assert _get_tag_value(mock_dcm, "NonExistent", "default") == "default"

    def test_get_tag_value_person_name(self) -> None:
        """Test _get_tag_value converts PersonName to string."""
        mock_dcm = MagicMock()
        # Simulate PersonName with family_name attribute
        mock_name = MagicMock()
        mock_name.family_name = "Smith"
        mock_dcm.PatientName = mock_name

        result = _get_tag_value(mock_dcm, "PatientName")
        assert isinstance(result, str)

    def test_get_tag_value_multivalue(self) -> None:
        """Test _get_tag_value converts MultiValue to list."""
        mock_dcm = MagicMock()
        # Simulate MultiValue (iterable non-string)
        mock_dcm.PixelSpacing = [0.5, 0.5]

        result = _get_tag_value(mock_dcm, "PixelSpacing")
        assert result == [0.5, 0.5]

    def test_get_tag_value_exception(self) -> None:
        """Test _get_tag_value handles exceptions."""
        mock_dcm = MagicMock()
        # Configure to raise exception when accessing attribute
        mock_dcm.configure_mock(spec=[])  # Empty spec causes AttributeError
        # Use regular getattr behavior which will fail
        type(mock_dcm).BadAttr = property(
            lambda self: (_ for _ in ()).throw(Exception("Error"))
        )

        result = _get_tag_value(mock_dcm, "BadAttr", "default")
        # Should return default on exception (lines 898-899)
        assert result == "default"


# ============================================================================
# Test DicomDatabase
# ============================================================================


class TestDicomDatabase:
    """Tests for DicomDatabase class."""

    def test_creation_empty(self) -> None:
        """Test creating empty database."""
        db = DicomDatabase()
        assert db.patients == []
        assert db.spacing_tolerance == 0.1

    def test_from_folders_empty_path(self) -> None:
        """Test with non-existent path."""
        db = DicomDatabase.from_folders(["/nonexistent/path"], show_progress=False)
        assert len(db.patients) == 0

    def test_dataframe_filtering(self) -> None:
        """Test DataFrame filtering by patient/study/series."""
        # Create a simple database manually
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[
                                DicomInstance("inst1", Path("/a.dcm")),
                            ],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        # Test filtering
        studies = db.get_studies_df(patient_id="P001")
        assert len(studies) == 1

        studies_empty = db.get_studies_df(patient_id="NONEXISTENT")
        assert len(studies_empty) == 0

        series = db.get_series_df(study_uid="study1")
        assert len(series) == 1

        instances = db.get_instances_df(series_uid="series1")
        assert len(instances) == 1

    def test_get_patients_df_full(self) -> None:
        """Test patient DataFrame export with common metadata."""
        patient = DicomPatient(
            patient_id="P001",
            patients_name="Test Patient",
            patients_birth_date="19800101",
            patients_sex="M",
            common_metadata={"Manufacturer": "TestMfr"},
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    study_date="20231231",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            modality="CT",
                            instances=[
                                DicomInstance("inst1", Path("/a.dcm")),
                                DicomInstance("inst2", Path("/b.dcm")),
                            ],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        # Test default: no instance lists
        df = db.get_patients_df()
        assert len(df) == 1
        assert df["PatientID"].iloc[0] == "P001"
        assert df["NumStudies"].iloc[0] == 1
        assert df["NumSeries"].iloc[0] == 1
        assert df["NumInstances"].iloc[0] == 2
        assert "Manufacturer" in df.columns
        assert df["Manufacturer"].iloc[0] == "TestMfr"
        assert df["EarliestStudyDate"].iloc[0] == "20231231"
        assert "InstanceSOPUIDs" not in df.columns
        assert "InstanceFilePaths" not in df.columns

        # Test with instance lists
        df_with = db.get_patients_df(include_instance_lists=True)
        assert len(df_with["InstanceSOPUIDs"].iloc[0]) == 2
        assert len(df_with["InstanceFilePaths"].iloc[0]) == 2

    def test_get_patients_df_no_study_dates(self) -> None:
        """Test patient DataFrame when studies have no dates."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    study_date=None,  # No date
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])
        df = db.get_patients_df()

        assert df["EarliestStudyDate"].iloc[0] is None
        assert df["LatestStudyDate"].iloc[0] is None

    def test_get_studies_df_with_metadata(self) -> None:
        """Test study DataFrame with common metadata from parent and study levels."""
        patient = DicomPatient(
            patient_id="P001",
            patients_name="Test",
            patients_birth_date="19800101",
            patients_sex="M",
            common_metadata={"InstitutionName": "TestHospital"},
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    study_date="20231231",
                    study_time="120000",
                    study_description="Test Study",
                    common_metadata={"ProtocolName": "Protocol1"},
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            modality="CT",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                        DicomSeries(
                            series_instance_uid="series2",
                            modality="MR",
                            instances=[DicomInstance("inst2", Path("/b.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])
        df = db.get_studies_df()

        assert len(df) == 1
        assert df["StudyInstanceUID"].iloc[0] == "study1"
        assert df["NumSeries"].iloc[0] == 2
        assert df["NumInstances"].iloc[0] == 2
        assert set(df["ModalitiesPresent"].iloc[0]) == {"CT", "MR"}
        assert "InstitutionName" in df.columns
        assert "ProtocolName" in df.columns
        # Default: no instance lists
        assert "InstanceSOPUIDs" not in df.columns
        assert "InstanceFilePaths" not in df.columns

        # Test with instance lists
        df_with = db.get_studies_df(include_instance_lists=True)
        assert len(df_with["InstanceSOPUIDs"].iloc[0]) == 2
        assert len(df_with["InstanceFilePaths"].iloc[0]) == 2

    def test_get_series_df_with_metadata(self) -> None:
        """Test series DataFrame with all level metadata."""
        patient = DicomPatient(
            patient_id="P001",
            patients_name="Test",
            common_metadata={"Manufacturer": "TestMfr"},
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    study_date="20231231",
                    study_description="Test Study",
                    common_metadata={"ProtocolName": "Protocol1"},
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            series_number=1,
                            series_description="Test Series",
                            modality="CT",
                            frame_of_reference_uid="frame1",
                            common_metadata={"SliceThickness": 2.5},
                            instances=[
                                DicomInstance(
                                    "inst1",
                                    Path("/a.dcm"),
                                    projection_score=0.0,
                                ),
                                DicomInstance(
                                    "inst2",
                                    Path("/b.dcm"),
                                    projection_score=2.5,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])
        df = db.get_series_df()

        assert len(df) == 1
        assert df["SeriesInstanceUID"].iloc[0] == "series1"
        assert df["NumInstances"].iloc[0] == 2
        assert df["IsComplete"].iloc[0] == True  # noqa: E712
        assert "Manufacturer" in df.columns
        assert "ProtocolName" in df.columns
        assert "SliceThickness" in df.columns
        # Default: no instance lists
        assert "InstanceSOPUIDs" not in df.columns
        assert "InstanceFilePaths" not in df.columns

        # Test with instance lists
        df_with = db.get_series_df(include_instance_lists=True)
        assert len(df_with["InstanceSOPUIDs"].iloc[0]) == 2
        assert len(df_with["InstanceFilePaths"].iloc[0]) == 2

    def test_get_series_df_filtering(self) -> None:
        """Test series DataFrame with patient and study filtering."""
        patient1 = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        patient2 = DicomPatient(
            patient_id="P002",
            studies=[
                DicomStudy(
                    study_instance_uid="study2",
                    series=[
                        DicomSeries(
                            series_instance_uid="series2",
                            instances=[DicomInstance("inst2", Path("/b.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient1, patient2])

        # Test patient filter
        series_p1 = db.get_series_df(patient_id="P001")
        assert len(series_p1) == 1
        assert series_p1["SeriesInstanceUID"].iloc[0] == "series1"

        # Test study filter
        series_s2 = db.get_series_df(study_uid="study2")
        assert len(series_s2) == 1
        assert series_s2["SeriesInstanceUID"].iloc[0] == "series2"

        # Test no match
        series_none = db.get_series_df(patient_id="NONEXISTENT")
        assert len(series_none) == 0

    def test_get_instances_df_with_image_position(self) -> None:
        """Test instance DataFrame with image position patient data."""
        patient = DicomPatient(
            patient_id="P001",
            patients_name="Test",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    study_date="20231231",
                    study_description="Test",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            series_number=1,
                            series_description="Axial",
                            modality="CT",
                            instances=[
                                DicomInstance(
                                    "inst1",
                                    Path("/a.dcm"),
                                    instance_number=1,
                                    image_position_patient=(1.0, 2.0, 3.0),
                                    image_orientation_patient=(1, 0, 0, 0, 1, 0),
                                    slice_location=100.0,
                                    projection_score=50.0,
                                    acquisition_datetime="20231231T120000",
                                    metadata={"KVP": 120},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])
        df = db.get_instances_df()

        assert len(df) == 1
        assert df["SOPInstanceUID"].iloc[0] == "inst1"
        assert df["ImagePositionPatient_X"].iloc[0] == 1.0
        assert df["ImagePositionPatient_Y"].iloc[0] == 2.0
        assert df["ImagePositionPatient_Z"].iloc[0] == 3.0
        assert df["SliceLocation"].iloc[0] == 100.0
        assert df["ProjectionScore"].iloc[0] == 50.0
        assert df["KVP"].iloc[0] == 120

    def test_get_instances_df_without_image_position(self) -> None:
        """Test instance DataFrame when image position is None."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[
                                DicomInstance(
                                    "inst1",
                                    Path("/a.dcm"),
                                    image_position_patient=None,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])
        df = db.get_instances_df()

        assert df["ImagePositionPatient_X"].iloc[0] is None
        assert df["ImagePositionPatient_Y"].iloc[0] is None
        assert df["ImagePositionPatient_Z"].iloc[0] is None

    def test_get_instances_df_filtering(self) -> None:
        """Test instance DataFrame filtering by patient/study/series."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                    ],
                ),
                DicomStudy(
                    study_instance_uid="study2",
                    series=[
                        DicomSeries(
                            series_instance_uid="series2",
                            instances=[DicomInstance("inst2", Path("/b.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        # Filter by patient
        insts_p1 = db.get_instances_df(patient_id="P001")
        assert len(insts_p1) == 2

        # Filter by study
        insts_s1 = db.get_instances_df(study_uid="study1")
        assert len(insts_s1) == 1

        # Filter by non-matching patient
        insts_empty = db.get_instances_df(patient_id="NONEXISTENT")
        assert len(insts_empty) == 0

        # Filter by non-matching study
        insts_empty2 = db.get_instances_df(study_uid="NONEXISTENT")
        assert len(insts_empty2) == 0

    def test_export_csv_default_levels(self) -> None:
        """Test CSV export with default (all) levels."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[
                                DicomInstance("inst1", Path("/a.dcm")),
                            ],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "export"
            created = db.export_csv(str(base_path))  # Default levels=None

            assert "patients" in created
            assert "studies" in created
            assert "series" in created
            assert "instances" in created

            for _level, path in created.items():
                assert Path(path).exists()

    def test_export_csv_invalid_level(self) -> None:
        """Test CSV export ignores invalid level names."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "export"
            # Include invalid level name
            created = db.export_csv(
                str(base_path), levels=["patients", "invalid_level"]
            )

            # Only valid levels are exported
            assert "patients" in created
            assert "invalid_level" not in created
            assert len(created) == 1

    def test_export_csv_include_instance_lists(self) -> None:
        """Test CSV export with include_instance_lists parameter."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "export"
            # Export with instance lists
            created = db.export_csv(
                str(base_path), levels=["patients"], include_instance_lists=True
            )
            assert "patients" in created

            # Read CSV and verify columns exist
            df = pd.read_csv(created["patients"])
            assert "InstanceSOPUIDs" in df.columns
            assert "InstanceFilePaths" in df.columns

    def test_export_json_full(self) -> None:
        """Test JSON export with complete hierarchy."""
        patient = DicomPatient(
            patient_id="P001",
            patients_name="Test",
            patients_birth_date="19800101",
            patients_sex="M",
            common_metadata={"Manufacturer": "TestMfr"},
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    study_date="20231231",
                    study_time="120000",
                    study_description="Test Study",
                    common_metadata={"Protocol": "P1"},
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            series_number=1,
                            series_description="Axial",
                            modality="CT",
                            frame_of_reference_uid="frame1",
                            common_metadata={"SliceThickness": 2.5},
                            instances=[
                                DicomInstance(
                                    "inst1",
                                    Path("/a.dcm"),
                                    instance_number=1,
                                    image_position_patient=(1.0, 2.0, 3.0),
                                    slice_location=100.0,
                                    projection_score=50.0,
                                    metadata={"Extra": "value"},
                                ),
                                DicomInstance(
                                    "inst2",
                                    Path("/b.dcm"),
                                    instance_number=2,
                                    projection_score=52.5,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "export.json"
            result = db.export_json(str(json_path))

            assert Path(result).exists()

            with open(result) as f:
                data = json.load(f)

            assert len(data["patients"]) == 1
            patient_data = data["patients"][0]
            assert patient_data["patient_id"] == "P001"
            assert patient_data["patients_name"] == "Test"
            assert len(patient_data["studies"]) == 1

            study_data = patient_data["studies"][0]
            assert study_data["study_instance_uid"] == "study1"
            assert len(study_data["series"]) == 1

            series_data = study_data["series"][0]
            assert series_data["series_instance_uid"] == "series1"
            assert "completeness" in series_data
            assert len(series_data["instances"]) == 2

            # Default includes file paths
            assert "file_path" in series_data["instances"][0]
            assert "instance_uids" in patient_data
            assert "file_paths" in patient_data

    def test_export_json_without_instance_lists(self) -> None:
        """Test JSON export without instance lists."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "export.json"
            result = db.export_json(str(json_path), include_instance_lists=False)

            with open(result) as f:
                data = json.load(f)

            patient_data = data["patients"][0]
            # No instance_uids or file_paths at patient level
            assert "instance_uids" not in patient_data
            assert "file_paths" not in patient_data

            # No file_path at instance level
            instance = patient_data["studies"][0]["series"][0]["instances"][0]
            assert "file_path" not in instance


# ============================================================================
# Test File Scanning Functions
# ============================================================================


class TestFileScanning:
    """Tests for file scanning functions."""

    def test_scan_empty_directory(self) -> None:
        """Test scanning empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = _scan_dicom_files(
                [Path(tmpdir)], recursive=True, show_progress=False
            )
            assert files == []

    def test_scan_nonexistent_path(self) -> None:
        """Test scanning non-existent path."""
        files = _scan_dicom_files(
            [Path("/nonexistent/path")],
            recursive=True,
            show_progress=False,
        )
        assert files == []

    def test_scan_with_non_dicom_files(self) -> None:
        """Test scanning handles non-DICOM files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-DICOM file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Not a DICOM file")

            files = _scan_dicom_files(
                [Path(tmpdir)],
                recursive=True,
                show_progress=False,
            )
            # Should not include non-DICOM file
            assert files == []


# ============================================================================
# Test Build Hierarchy
# ============================================================================


class TestBuildHierarchy:
    """Tests for hierarchy building."""

    def test_build_hierarchy_basic(self) -> None:
        """Test building hierarchy from flat metadata."""
        metadata = [
            {
                "file_path": Path("/test/a.dcm"),
                "PatientID": "P001",
                "StudyInstanceUID": "study1",
                "SeriesInstanceUID": "series1",
                "SOPInstanceUID": "inst1",
            },
            {
                "file_path": Path("/test/b.dcm"),
                "PatientID": "P001",
                "StudyInstanceUID": "study1",
                "SeriesInstanceUID": "series1",
                "SOPInstanceUID": "inst2",
            },
        ]

        patients = _build_hierarchy(metadata, spacing_tolerance=0.1)
        assert len(patients) == 1
        assert patients[0].patient_id == "P001"
        assert len(patients[0].studies) == 1
        assert len(patients[0].studies[0].series) == 1
        assert len(patients[0].studies[0].series[0].instances) == 2

    def test_build_hierarchy_multiple_patients(self) -> None:
        """Test with multiple patients."""
        metadata = [
            {
                "file_path": Path("/test/a.dcm"),
                "PatientID": "P001",
                "StudyInstanceUID": "study1",
                "SeriesInstanceUID": "series1",
                "SOPInstanceUID": "inst1",
            },
            {
                "file_path": Path("/test/b.dcm"),
                "PatientID": "P002",
                "StudyInstanceUID": "study2",
                "SeriesInstanceUID": "series2",
                "SOPInstanceUID": "inst2",
            },
        ]

        patients = _build_hierarchy(metadata, spacing_tolerance=0.1)
        assert len(patients) == 2


# ============================================================================
# Synthetic DICOM Tests (for CI - no external data required)
# ============================================================================


class TestSyntheticDicom:
    """Tests using synthetic DICOM files - runs on CI without external data."""

    def test_from_folders_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test from_folders with synthetic DICOM files."""
        db = DicomDatabase.from_folders(
            [str(synthetic_dicom_dir)],
            recursive=True,
            show_progress=False,
        )

        assert len(db.patients) == 1
        assert db.patients[0].patient_id == "SYNTH_PATIENT_001"
        assert len(db.patients[0].studies) == 1
        assert len(db.patients[0].studies[0].series) == 1
        assert len(db.patients[0].studies[0].series[0].instances) == 5

    def test_multi_patient_hierarchy(self, synthetic_multi_patient_dir: Path) -> None:
        """Test hierarchy building with multiple synthetic patients."""
        db = DicomDatabase.from_folders(
            [str(synthetic_multi_patient_dir)],
            recursive=True,
            show_progress=False,
        )

        assert len(db.patients) == 2
        # Verify sorting (patients sorted by ID)
        assert db.patients[0].patient_id == "SYNTH_PATIENT_001"
        assert db.patients[1].patient_id == "SYNTH_PATIENT_002"

    def test_completeness_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test completeness detection with synthetic files."""
        db = DicomDatabase.from_folders(
            [str(synthetic_dicom_dir)],
            show_progress=False,
        )

        series = db.patients[0].studies[0].series[0]
        completeness = series.check_completeness()
        assert completeness["is_complete"] is True
        assert completeness["total_instances"] == 5

    def test_dataframes_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test DataFrame exports with synthetic files."""
        db = DicomDatabase.from_folders(
            [str(synthetic_dicom_dir)],
            show_progress=False,
        )

        patients_df = db.get_patients_df()
        studies_df = db.get_studies_df()
        series_df = db.get_series_df()
        instances_df = db.get_instances_df()

        assert len(patients_df) == 1
        assert len(studies_df) == 1
        assert len(series_df) == 1
        assert len(instances_df) == 5

        # Default: no instance lists
        assert "InstanceSOPUIDs" not in patients_df.columns
        assert "InstanceFilePaths" not in series_df.columns

        # Test with instance lists
        patients_df_with = db.get_patients_df(include_instance_lists=True)
        assert len(patients_df_with["InstanceSOPUIDs"].iloc[0]) == 5

    def test_export_csv_synthetic(
        self, synthetic_dicom_dir: Path, tmp_path: Path
    ) -> None:
        """Test CSV export with synthetic files."""
        db = DicomDatabase.from_folders(
            [str(synthetic_dicom_dir)],
            show_progress=False,
        )

        export_dir = tmp_path / "csv_export"
        result = db.export_csv(str(export_dir))

        assert "patients" in result
        assert "studies" in result
        assert "series" in result
        assert "instances" in result

        for _level, path in result.items():
            assert Path(path).exists()

    def test_export_json_synthetic(
        self, synthetic_dicom_dir: Path, tmp_path: Path
    ) -> None:
        """Test JSON export with synthetic files."""
        db = DicomDatabase.from_folders(
            [str(synthetic_dicom_dir)],
            show_progress=False,
        )

        json_path = tmp_path / "export.json"
        result = db.export_json(str(json_path))

        assert Path(result).exists()

        with open(result) as f:
            data = json.load(f)

        assert len(data["patients"]) == 1
        assert data["patients"][0]["patient_id"] == "SYNTH_PATIENT_001"

    def test_scan_files_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test file scanning with synthetic files."""
        files = _scan_dicom_files(
            [synthetic_dicom_dir],
            recursive=True,
            show_progress=False,
            num_workers=1,
        )
        assert len(files) == 5

    def test_parallel_scan_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test parallel file scanning with synthetic files."""
        files = _scan_dicom_files(
            [synthetic_dicom_dir],
            recursive=True,
            show_progress=False,
            num_workers=2,
        )
        assert len(files) == 5

    def test_extract_metadata_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test metadata extraction with synthetic files."""
        files = _scan_dicom_files(
            [synthetic_dicom_dir],
            recursive=True,
            show_progress=False,
            num_workers=1,
        )
        metadata = _extract_all_metadata(
            files,
            show_progress=False,
            extract_private_tags=False,
            num_workers=1,
        )
        assert len(metadata) == 5
        assert all(m["PatientID"] == "SYNTH_PATIENT_001" for m in metadata)

    def test_parallel_metadata_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test parallel metadata extraction with synthetic files."""
        files = _scan_dicom_files(
            [synthetic_dicom_dir],
            recursive=True,
            show_progress=False,
            num_workers=1,
        )
        metadata = _extract_all_metadata(
            files,
            show_progress=False,
            extract_private_tags=False,
            num_workers=2,
        )
        assert len(metadata) == 5

    def test_is_dicom_file_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test _is_dicom_file with synthetic file."""
        files = list(synthetic_dicom_dir.glob("*.dcm"))
        assert len(files) > 0
        result = _is_dicom_file(files[0])
        assert result == files[0]

    def test_single_file_metadata_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test single file metadata extraction with synthetic file."""
        files = list(synthetic_dicom_dir.glob("*.dcm"))
        result = _extract_single_file_metadata(files[0], extract_private_tags=False)

        assert result is not None
        assert result["PatientID"] == "SYNTH_PATIENT_001"
        assert result["Modality"] == "CT"
        assert result["ImagePositionPatient"] is not None

    def test_parallel_from_folders_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test from_folders with explicit worker count."""
        db1 = DicomDatabase.from_folders(
            [str(synthetic_dicom_dir)],
            show_progress=False,
            num_workers=1,
        )
        db2 = DicomDatabase.from_folders(
            [str(synthetic_dicom_dir)],
            show_progress=False,
            num_workers=2,
        )

        # Results should be identical
        assert len(db1.patients) == len(db2.patients)
        assert len(db1.patients[0].studies[0].series[0].instances) == len(
            db2.patients[0].studies[0].series[0].instances
        )

    def test_scan_single_file_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test scanning a single DICOM file path directly."""
        files = list(synthetic_dicom_dir.glob("*.dcm"))
        assert len(files) > 0
        single_file = files[0]

        # Pass list containing a single file path
        # This hits path.is_file() branch (line 763)
        result = _scan_dicom_files([single_file], recursive=False, show_progress=False)
        assert len(result) == 1
        assert result[0] == single_file

    def test_scan_non_recursive_synthetic(self, synthetic_dicom_dir: Path) -> None:
        """Test non-recursive directory scan."""
        # synthetic_dicom_dir has files at root
        # Create a subdirectory with a file
        subdir = synthetic_dicom_dir / "subdir"
        subdir.mkdir()

        # Create a new synthetic file in the subdir
        subfile = subdir / "sub.dcm"
        create_synthetic_dicom(
            subfile,
            patient_id="SUB_PATIENT",
            sop_uid="1.2.3.999",
        )

        # Scan non-recursively - should only find root files (original 5)
        # This hits recursive=False branch (line 768)
        result = _scan_dicom_files(
            [synthetic_dicom_dir], recursive=False, show_progress=False
        )
        assert len(result) == 5
        assert subfile not in result

        # Verify recursive scan finds it (total 6)
        result_rec = _scan_dicom_files(
            [synthetic_dicom_dir], recursive=True, show_progress=False
        )
        assert len(result_rec) == 6


# ============================================================================
# Test Metadata Extraction Functions
# ============================================================================


class TestMetadataExtraction:
    """Tests for metadata extraction functions."""

    def test_extract_single_file_metadata_invalid_file(self) -> None:
        """Test extracting metadata from invalid file returns None (lines 789-790)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid file
            invalid_file = Path(tmpdir) / "invalid.dcm"
            invalid_file.write_text("Not a valid DICOM file")

            result = _extract_single_file_metadata(
                invalid_file, extract_private_tags=True
            )
            assert result is None


# ============================================================================
# Test Additional Edge Cases for 100% Coverage
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases to achieve 100% coverage."""

    def test_instances_df_filter_series_uid_no_match(self) -> None:
        """Test instances DataFrame with non-matching series_uid (line 525)."""
        patient = DicomPatient(
            patient_id="P001",
            studies=[
                DicomStudy(
                    study_instance_uid="study1",
                    series=[
                        DicomSeries(
                            series_instance_uid="series1",
                            instances=[DicomInstance("inst1", Path("/a.dcm"))],
                        ),
                        DicomSeries(
                            series_instance_uid="series2",
                            instances=[DicomInstance("inst2", Path("/b.dcm"))],
                        ),
                    ],
                ),
            ],
        )
        db = DicomDatabase(patients=[patient])

        # Filter by series_uid that doesn't match first series but matches second
        df = db.get_instances_df(series_uid="series2")
        assert len(df) == 1
        assert df["SOPInstanceUID"].iloc[0] == "inst2"

        # Filter by non-existent series
        df_empty = db.get_instances_df(series_uid="nonexistent")
        assert len(df_empty) == 0

    def test_build_hierarchy_with_spatial_data(self) -> None:
        """Test hierarchy building with full spatial metadata."""
        metadata = [
            {
                "file_path": Path("/test/a.dcm"),
                "PatientID": "P001",
                "PatientsName": "Test Patient",
                "PatientsBirthDate": "19800101",
                "PatientsSex": "M",
                "StudyInstanceUID": "study1",
                "StudyDate": "20231231",
                "StudyTime": "120000",
                "StudyDescription": "Test Study",
                "SeriesInstanceUID": "series1",
                "SeriesNumber": 1,
                "SeriesDescription": "Axial",
                "Modality": "CT",
                "FrameOfReferenceUID": "frame1",
                "SOPInstanceUID": "inst1",
                "InstanceNumber": 1,
                "SliceLocation": 0.0,
                "ImagePositionPatient": (0.0, 0.0, 0.0),
                "ImageOrientationPatient": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                "ProjectionScore": 0.0,
                "AcquisitionDate": "20231231",
                "AcquisitionTime": "120000",
                "SliceThickness": 2.5,
                "Manufacturer": "TestMfr",
            },
            {
                "file_path": Path("/test/b.dcm"),
                "PatientID": "P001",
                "PatientsName": "Test Patient",
                "PatientsBirthDate": "19800101",
                "PatientsSex": "M",
                "StudyInstanceUID": "study1",
                "StudyDate": "20231231",
                "StudyTime": "120000",
                "StudyDescription": "Test Study",
                "SeriesInstanceUID": "series1",
                "SeriesNumber": 1,
                "SeriesDescription": "Axial",
                "Modality": "CT",
                "FrameOfReferenceUID": "frame1",
                "SOPInstanceUID": "inst2",
                "InstanceNumber": 2,
                "SliceLocation": 2.5,
                "ImagePositionPatient": (0.0, 0.0, 2.5),
                "ImageOrientationPatient": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                "ProjectionScore": 2.5,
                "AcquisitionDate": "20231231",
                "AcquisitionTime": "120001",
                "SliceThickness": 2.5,
                "Manufacturer": "TestMfr",
            },
        ]

        patients = _build_hierarchy(metadata, spacing_tolerance=0.1)
        assert len(patients) == 1
        patient = patients[0]

        # Check patient attributes
        assert patient.patient_id == "P001"
        assert patient.patients_name == "Test Patient"

        # Check series
        series = patient.studies[0].series[0]
        assert len(series.instances) == 2

        # Check instances have spatial data
        inst = series.instances[0]
        assert inst.image_position_patient is not None
        assert inst.projection_score is not None

        # Check common metadata was extracted
        assert (
            "SliceThickness" in series.common_metadata
            or "Manufacturer" in series.common_metadata
        )

    def test_check_completeness_no_instance_numbers(self) -> None:
        """Test completeness check when instances have no instance numbers."""
        instances = [
            DicomInstance(
                f"uid{i}",
                Path(f"/{i}.dcm"),
                instance_number=None,  # No instance number
                projection_score=None,  # No projection score
            )
            for i in range(3)
        ]
        series = DicomSeries(series_instance_uid="1.2.3", instances=instances)
        result = series.check_completeness()
        # Should have warning about insufficient spatial data
        assert len(result["warnings"]) > 0


# ============================================================================
# Mock-Based Tests for Exception Handling Coverage
# ============================================================================


class TestExceptionHandling:
    """Tests using mocks to cover exception handling branches."""

    def test_scan_dicom_files_is_dicom_exception(self) -> None:
        """Test that exceptions in is_dicom check are handled (lines 738-740)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file that will cause an exception when checked
            test_file = Path(tmpdir) / "problem.dcm"
            test_file.write_bytes(b"\x00" * 100)  # Invalid binary data

            with patch(
                "pictologics.utilities.dicom_database.pydicom.misc.is_dicom"
            ) as mock_is_dicom:
                # Make is_dicom raise an exception
                mock_is_dicom.side_effect = Exception("Mock exception")

                files = _scan_dicom_files(
                    [Path(tmpdir)], recursive=True, show_progress=False
                )
                # Should handle exception gracefully and return empty list
                assert files == []

    def test_extract_all_metadata_exception(self) -> None:
        """Test that exceptions in metadata extraction are handled (lines 768-770)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy file
            test_file = Path(tmpdir) / "test.dcm"
            test_file.write_text("dummy")

            with patch(
                "pictologics.utilities.dicom_database._extract_single_file_metadata"
            ) as mock_extract:
                # Make extraction raise an exception
                mock_extract.side_effect = Exception("Mock exception")

                result = _extract_all_metadata(
                    [test_file], show_progress=False, extract_private_tags=True
                )
                # Should handle exception gracefully and return empty list
                assert result == []

    def test_extract_metadata_ipp_exception(self) -> None:
        """Test ImagePositionPatient extraction exception handling (lines 822-823)."""
        with patch(
            "pictologics.utilities.dicom_database.pydicom.dcmread"
        ) as mock_dcmread:
            mock_dcm = MagicMock()
            mock_dcm.PatientID = "P001"
            mock_dcm.StudyInstanceUID = "study1"
            mock_dcm.SeriesInstanceUID = "series1"
            mock_dcm.SOPInstanceUID = "inst1"

            # Set ImagePositionPatient and ImageOrientationPatient to None
            # to test the else branch (line 842-843)
            mock_dcm.ImagePositionPatient = None
            mock_dcm.ImageOrientationPatient = None

            # Mock __iter__ to return an empty iterator for private tags
            mock_dcm.__iter__ = MagicMock(return_value=iter([]))

            mock_dcmread.return_value = mock_dcm

            result = _extract_single_file_metadata(
                Path("/fake.dcm"), extract_private_tags=False
            )
            assert result is not None
            # ProjectionScore should be None when IPP/IOP are None (line 842-843)
            assert result["ProjectionScore"] is None

    def test_extract_metadata_private_tag_exception(self) -> None:
        """Test private tag extraction exception handling (lines 875-876)."""
        with patch(
            "pictologics.utilities.dicom_database.pydicom.dcmread"
        ) as mock_dcmread:
            mock_dcm = MagicMock()
            mock_dcm.PatientID = "P001"
            mock_dcm.StudyInstanceUID = "study1"
            mock_dcm.SeriesInstanceUID = "series1"
            mock_dcm.SOPInstanceUID = "inst1"

            # Configure normal attributes
            mock_dcm.ImagePositionPatient = None
            mock_dcm.ImageOrientationPatient = None

            # Create a private tag that raises exception when accessed
            mock_private_elem = MagicMock()
            mock_private_elem.tag = MagicMock()
            mock_private_elem.tag.is_private = True
            mock_private_elem.tag.group = 0x0019
            mock_private_elem.tag.element = 0x1001
            # Make value access raise exception
            type(mock_private_elem).value = property(
                lambda self: (_ for _ in ()).throw(Exception("Cannot read value"))
            )

            mock_dcm.__iter__ = MagicMock(return_value=iter([mock_private_elem]))

            mock_dcmread.return_value = mock_dcm

            result = _extract_single_file_metadata(
                Path("/fake.dcm"), extract_private_tags=True
            )
            # Should handle exception gracefully
            assert result is not None
            # Private tag should not be in result (exception was caught)
            assert "Private_0019_1001" not in result

    def test_extract_metadata_projection_score_exception(self) -> None:
        """Test projection score calculation exception handling (lines 840-841)."""
        with patch(
            "pictologics.utilities.dicom_database.pydicom.dcmread"
        ) as mock_dcmread:
            mock_dcm = MagicMock()
            mock_dcm.PatientID = "P001"
            mock_dcm.StudyInstanceUID = "study1"
            mock_dcm.SeriesInstanceUID = "series1"
            mock_dcm.SOPInstanceUID = "inst1"

            # Set valid IPP and IOP that would normally work
            mock_dcm.ImagePositionPatient = [0.0, 0.0, 0.0]
            mock_dcm.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

            mock_dcm.__iter__ = MagicMock(return_value=iter([]))

            mock_dcmread.return_value = mock_dcm

            # Mock numpy to raise exception during projection calculation
            with patch("pictologics.utilities.dicom_database.np.cross") as mock_cross:
                mock_cross.side_effect = Exception("Math error")

                result = _extract_single_file_metadata(
                    Path("/fake.dcm"), extract_private_tags=False
                )
                assert result is not None
                # Projection score should be None due to exception
                assert result["ProjectionScore"] is None


# ============================================================================
# Test Parallel Processing Functions
# ============================================================================


class TestParallelProcessing:
    """Tests for parallel processing functionality."""

    def test_get_num_workers_auto(self) -> None:
        """Test automatic worker count detection."""
        workers = _get_num_workers(None)
        cpu_count = os.cpu_count() or 1
        expected = max(1, cpu_count - 1)
        assert workers == expected

    def test_get_num_workers_explicit(self) -> None:
        """Test explicit worker count."""
        assert _get_num_workers(4) == 4
        assert _get_num_workers(1) == 1
        assert _get_num_workers(0) == 1  # Minimum 1
        assert _get_num_workers(-1) == 1  # Minimum 1

    def test_is_dicom_file_invalid(self) -> None:
        """Test _is_dicom_file with invalid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "invalid.txt"
            invalid_file.write_text("Not a DICOM file")
            result = _is_dicom_file(invalid_file)
            assert result is None

    def test_sort_hierarchy(self) -> None:
        """Test _sort_hierarchy sorts all levels correctly."""
        # Create unsorted data
        patients = [
            DicomPatient(
                patient_id="P002",
                studies=[
                    DicomStudy(
                        study_instance_uid="study2",
                        study_date="20230102",
                        series=[
                            DicomSeries(
                                series_instance_uid="series2",
                                series_number=2,
                                instances=[
                                    DicomInstance(
                                        "inst2", Path("/2.dcm"), projection_score=10.0
                                    ),
                                    DicomInstance(
                                        "inst1", Path("/1.dcm"), projection_score=5.0
                                    ),
                                ],
                            ),
                            DicomSeries(
                                series_instance_uid="series1",
                                series_number=1,
                                instances=[DicomInstance("inst3", Path("/3.dcm"))],
                            ),
                        ],
                    ),
                    DicomStudy(
                        study_instance_uid="study1",
                        study_date="20230101",
                        series=[],
                    ),
                ],
            ),
            DicomPatient(
                patient_id="P001",
                studies=[],
            ),
        ]

        sorted_patients = _sort_hierarchy(patients)

        # Check patients sorted by ID
        assert sorted_patients[0].patient_id == "P001"
        assert sorted_patients[1].patient_id == "P002"

        # Check studies sorted by date
        p2_studies = sorted_patients[1].studies
        assert p2_studies[0].study_date == "20230101"
        assert p2_studies[1].study_date == "20230102"

        # Check series sorted by number
        s2_series = p2_studies[1].series
        assert s2_series[0].series_number == 1
        assert s2_series[1].series_number == 2

        # Check instances sorted by projection score
        s2_instances = s2_series[1].instances
        assert s2_instances[0].projection_score == 5.0
        assert s2_instances[1].projection_score == 10.0

    def test_extract_all_metadata_empty(self) -> None:
        """Test _extract_all_metadata with empty list."""
        result = _extract_all_metadata(
            [], show_progress=False, extract_private_tags=False, num_workers=2
        )
        assert result == []

    def test_is_dicom_file_exception(self) -> None:
        """Test _is_dicom_file handles exceptions (lines 734-735)."""
        with patch(
            "pictologics.utilities.dicom_database.pydicom.misc.is_dicom"
        ) as mock:
            mock.side_effect = Exception("Read error")
            result = _is_dicom_file(Path("/fake/file.dcm"))
            assert result is None

    def test_extract_metadata_wrapper(self) -> None:
        """Test _extract_metadata_wrapper directly (lines 815-816)."""
        from pictologics.utilities.dicom_database import _extract_metadata_wrapper

        # Test with a non-existent file (should return None)
        result = _extract_metadata_wrapper((Path("/nonexistent/file.dcm"), False))
        assert result is None


# ============================================================================
# Test Multi-Series Splitting
# ============================================================================


class TestMultiSeriesSplitting:
    """Tests for multi-phase series splitting functionality."""

    def test_split_by_tag(self, synthetic_dicom_dir: Path) -> None:
        """Test splitting by specific tag (AcquisitionNumber)."""
        series_uid = "1.2.826.0.1.3680043.2.1125.1.1"  # Valid-looking OID

        # Create 2 files with AcquisitionNumber=1 and 2 files with AcquisitionNumber=2
        for i in range(1, 3):
            create_synthetic_dicom(
                synthetic_dicom_dir / f"p1_{i}.dcm",
                series_uid=series_uid,
                sop_uid=f"{series_uid}.1.{i}",
                instance_number=i,
                AcquisitionNumber=1,
            )
        for i in range(1, 3):
            create_synthetic_dicom(
                synthetic_dicom_dir / f"p2_{i}.dcm",
                series_uid=series_uid,
                sop_uid=f"{series_uid}.2.{i}",
                instance_number=i,
                AcquisitionNumber=2,
            )

        # 1. Test splitting enabled (default)
        db = DicomDatabase.from_folders(
            [synthetic_dicom_dir], split_multiseries=True, show_progress=False
        )
        series_df = db.get_series_df()

        # Manual filtering to avoid numpy/pandas TypeError with coverage
        series_uids = series_df["SeriesInstanceUID"].astype(str).tolist()
        series_uids = series_df["SeriesInstanceUID"].astype(str).tolist()
        # num_instances = series_df["NumInstances"].tolist()

        target_indices = [
            i for i, uid in enumerate(series_uids) if uid.startswith(series_uid)
        ]
        assert len(target_indices) == 2

        # Verify suffixes
        matched_uids = [series_uids[i] for i in target_indices]
        assert any(u.endswith(".1") for u in matched_uids)
        assert any(u.endswith(".2") for u in matched_uids)

        # 2. Test splitting disabled
        db_no_split = DicomDatabase.from_folders(
            [synthetic_dicom_dir], split_multiseries=False, show_progress=False
        )
        series_df_ns = db_no_split.get_series_df()

        # Manual filtering again
        ns_uids = series_df_ns["SeriesInstanceUID"].astype(str).tolist()
        ns_counts = series_df_ns["NumInstances"].tolist()

        matched_ns_indices = [i for i, uid in enumerate(ns_uids) if uid == series_uid]
        assert len(matched_ns_indices) == 1
        assert ns_counts[matched_ns_indices[0]] == 4

    def test_split_by_spatial_fallback(self, synthetic_dicom_dir: Path) -> None:
        """Test splitting by spatial duplicates when tags are missing."""
        series_uid = "1.2.826.0.1.3680043.2.1125.2.1"

        # Create 2 sets of images at same positions (duplicates)
        positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.5)]

        # Phase 1
        for i, pos in enumerate(positions):
            create_synthetic_dicom(
                synthetic_dicom_dir / f"dup1_{i}.dcm",
                series_uid=series_uid,
                sop_uid=f"{series_uid}.1.{i}",
                image_position=pos,
                instance_number=i + 1,
            )

        # Phase 2 (duplicates)
        for i, pos in enumerate(positions):
            create_synthetic_dicom(
                synthetic_dicom_dir / f"dup2_{i}.dcm",
                series_uid=series_uid,
                sop_uid=f"{series_uid}.2.{i}",
                image_position=pos,
                instance_number=i + 1,
            )

        db = DicomDatabase.from_folders(
            [synthetic_dicom_dir], split_multiseries=True, show_progress=False
        )
        series_df = db.get_series_df()

        # Manual filtering
        s_uids = series_df["SeriesInstanceUID"].astype(str).tolist()
        s_counts = series_df["NumInstances"].tolist()

        target_indices = [
            i for i, uid in enumerate(s_uids) if uid.startswith(series_uid)
        ]
        assert len(target_indices) == 2

        counts = [s_counts[i] for i in target_indices]
        assert all(n == 2 for n in counts)

    def test_no_split_needed(self, synthetic_dicom_dir: Path) -> None:
        """Test that single-phase data is not split."""
        series_uid = "1.2.826.0.1.3680043.2.1125.3.1"

        # Create normal contiguous series
        for i in range(3):
            create_synthetic_dicom(
                synthetic_dicom_dir / f"norm_{i}.dcm",
                series_uid=series_uid,
                sop_uid=f"{series_uid}.{i}",
                image_position=(0.0, 0.0, i * 2.5),
            )

        db = DicomDatabase.from_folders(
            [synthetic_dicom_dir], split_multiseries=True, show_progress=False
        )
        series_df = db.get_series_df()

        # Manual filtering
        s_uids = series_df["SeriesInstanceUID"].astype(str).tolist()
        s_counts = series_df["NumInstances"].tolist()

        matches = [i for i, uid in enumerate(s_uids) if uid == series_uid]
        assert len(matches) == 1
        assert s_counts[matches[0]] == 3

        assert s_uids[matches[0]] == series_uid

    def test_split_instance_no_position(self, synthetic_dicom_dir: Path) -> None:
        """Test splitting with instance missing ImagePositionPatient (lines 787-789)."""
        series_uid = "1.2.826.0.1.3680043.2.1125.4.1"

        # Create file with normal position
        create_synthetic_dicom(
            synthetic_dicom_dir / "pos.dcm",
            series_uid=series_uid,
            sop_uid=f"{series_uid}.1",
            image_position=(0.0, 0.0, 0.0),
        )

        # Create file without position (will trigger line 789)
        create_synthetic_dicom(
            synthetic_dicom_dir / "nopos.dcm",
            series_uid=series_uid,
            sop_uid=f"{series_uid}.2",
            image_position=None,
        )

        # Now run DB just for this folder/series which effectively tests _split_series_instances via fallback
        # Because we didn't add splitting tags, it falls back to spatial logic.
        # One file has position, one doesn't.

        db = DicomDatabase.from_folders(
            [synthetic_dicom_dir], split_multiseries=True, show_progress=False
        )
        series_df = db.get_series_df()

        # Should be handled gracefully (likely grouped into phase 0 or separate)
        # Current logic puts no-position instances into group 0.
        # If there are duplicates, group 0 might be the first phase.

        # Just verifying it doesn't crash and returns correct count
        s_uids = series_df["SeriesInstanceUID"].astype(str).tolist()
        matches = [i for i, uid in enumerate(s_uids) if uid == series_uid]

        # Might be split or not depending on duplicates.
        # (0,0,0) exists. None exists. They are not duplicates of each other.
        # So likely 1 series.

        assert len(matches) == 1
        # Should have found both files
        # Note: fixture creates 5 files initially again.
        # Filtering for our series UID.

        # Manual verify
        # series_df.loc[matches[0]] should have NumInstances
        # But for list approach:
        # We need to find the count.
        # Assuming only our series matches the specific UID
        assert len(matches) > 0

        # Verify that we actually have an instance with None position
        # Get the series object
        # s_idx = matches[0]
        # Accessing private attribute to verify (or via public API if available)
        # db.get_series_df() doesn't give objects.
        # We need to access db.patients...
        # Instead, verify via dataframe? DataFrame has ImagePositionPatient column?
        # Yes.

        series_inst_df = db.get_instances_df()

        # Manual filtering again to avoid TypeError
        i_uids = series_inst_df["SeriesInstanceUID"].astype(str).tolist()
        # IPP is exploded into X, Y, Z columns
        i_positions_x = series_inst_df["ImagePositionPatient_X"].tolist()

        matching_indices = [i for i, uid in enumerate(i_uids) if uid == series_uid]
        assert len(matching_indices) == 2

        # Check positions for matching rows
        vals = [i_positions_x[i] for i in matching_indices]
        # One should be 0.0, one should be None (or NaN)
        # NaN check: x != x for floats
        assert any(v is None or (isinstance(v, float) and v != v) for v in vals)
