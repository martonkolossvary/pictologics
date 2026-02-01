"""
Test suite for DICOM SR Parser.

Tests cover all functionality including:
- SR file detection
- SRDocument creation and parsing
- SRMeasurement and SRMeasurementGroup dataclasses
- DataFrame/CSV/JSON export methods
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Disable JIT warmup for tests
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import pandas as pd
import pytest

from pictologics.utilities.sr_parser import (
    SRDocument,
    SRMeasurement,
    SRMeasurementGroup,
    _extract_numeric_measurement,
    _get_code_value,
    _parse_content_sequence,
    is_dicom_sr,
)

# ============================================================================
# Fixtures and Helpers
# ============================================================================


def create_mock_sr_dataset(
    sop_class: str = "1.2.840.10008.5.1.4.1.1.88.33",  # Comprehensive SR
    template_id: str = "1500",
    n_groups: int = 2,
    n_measurements_per_group: int = 3,
) -> MagicMock:
    """Create a mock DICOM SR dataset for testing."""
    mock_sr = MagicMock()

    mock_sr.SOPClassUID = sop_class
    mock_sr.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    mock_sr.PatientID = "TEST_PATIENT"
    mock_sr.StudyInstanceUID = "1.2.3.4.5.1"
    mock_sr.SeriesInstanceUID = "1.2.3.4.5.2"
    mock_sr.ContentDate = "20240101"
    mock_sr.ContentTime = "120000"

    # ConceptNameCodeSequence (document title)
    concept = MagicMock()
    concept.CodeMeaning = "Measurement Report"
    mock_sr.ConceptNameCodeSequence = [concept]

    # ContentTemplateSequence
    template = MagicMock()
    template.TemplateIdentifier = template_id
    mock_sr.ContentTemplateSequence = [template]

    # Build ContentSequence with groups and measurements
    content_items = []

    for group_idx in range(n_groups):
        # Create CONTAINER item for group
        container = MagicMock()
        container.RelationshipType = "CONTAINS"
        container.ValueType = "CONTAINER"

        group_concept = MagicMock()
        group_concept.CodeMeaning = f"Measurement Group {group_idx + 1}"
        container.ConceptNameCodeSequence = [group_concept]

        # Add nested measurements to container
        container.ContentSequence = []
        for meas_idx in range(n_measurements_per_group):
            num_item = MagicMock()
            num_item.RelationshipType = "CONTAINS"
            num_item.ValueType = "NUM"

            meas_concept = MagicMock()
            meas_concept.CodeMeaning = f"Measurement {meas_idx + 1}"
            num_item.ConceptNameCodeSequence = [meas_concept]

            # MeasuredValueSequence
            mv = MagicMock()
            mv.NumericValue = float(100 + meas_idx * 10)

            unit = MagicMock()
            unit.CodeValue = "mm"
            mv.MeasurementUnitsCodeSequence = [unit]

            num_item.MeasuredValueSequence = [mv]
            container.ContentSequence.append(num_item)

        content_items.append(container)

    mock_sr.ContentSequence = content_items
    return mock_sr


# ============================================================================
# Test is_dicom_sr
# ============================================================================


class TestIsDicomSR:
    """Tests for is_dicom_sr helper function."""

    def test_is_dicom_sr_comprehensive(self) -> None:
        """Test detection of Comprehensive SR."""
        with patch("pydicom.dcmread") as mock_read:
            mock_dcm = MagicMock()
            mock_dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.33"
            mock_read.return_value = mock_dcm

            result = is_dicom_sr("/path/to/file.dcm")
            assert result is True

    def test_is_dicom_sr_basic_text(self) -> None:
        """Test detection of Basic Text SR."""
        with patch("pydicom.dcmread") as mock_read:
            mock_dcm = MagicMock()
            mock_dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"
            mock_read.return_value = mock_dcm

            result = is_dicom_sr("/path/to/file.dcm")
            assert result is True

    def test_is_dicom_sr_false_for_ct(self) -> None:
        """Test non-SR DICOM returns False."""
        with patch("pydicom.dcmread") as mock_read:
            mock_dcm = MagicMock()
            mock_dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
            mock_read.return_value = mock_dcm

            result = is_dicom_sr("/path/to/file.dcm")
            assert result is False

    def test_is_dicom_sr_exception(self) -> None:
        """Test exception handling returns False."""
        with patch("pydicom.dcmread") as mock_read:
            mock_read.side_effect = Exception("Read error")

            result = is_dicom_sr("/path/to/file.dcm")
            assert result is False


# ============================================================================
# Test SRMeasurement
# ============================================================================


class TestSRMeasurement:
    """Tests for SRMeasurement dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating measurement with required fields."""
        meas = SRMeasurement(name="Volume", value=123.45, unit="mL")
        assert meas.name == "Volume"
        assert meas.value == 123.45
        assert meas.unit == "mL"
        assert meas.finding_type is None
        assert meas.metadata == {}

    def test_creation_full(self) -> None:
        """Test creating measurement with all fields."""
        meas = SRMeasurement(
            name="Agatston Score",
            value=256.0,
            unit="1",
            finding_type="Calcification",
            finding_site="Coronary Artery",
            derivation="Automatic",
            tracking_id="TRACK001",
            metadata={"algorithm": "v2.0"},
        )
        assert meas.finding_type == "Calcification"
        assert meas.finding_site == "Coronary Artery"
        assert meas.tracking_id == "TRACK001"


# ============================================================================
# Test SRMeasurementGroup
# ============================================================================


class TestSRMeasurementGroup:
    """Tests for SRMeasurementGroup dataclass."""

    def test_creation_empty(self) -> None:
        """Test creating empty group."""
        group = SRMeasurementGroup()
        assert group.group_id is None
        assert group.measurements == []

    def test_creation_with_measurements(self) -> None:
        """Test creating group with measurements."""
        measurements = [
            SRMeasurement("M1", 10.0, "mm"),
            SRMeasurement("M2", 20.0, "mm"),
        ]
        group = SRMeasurementGroup(
            group_id="G1",
            finding_type="Lesion",
            finding_site="Liver",
            measurements=measurements,
        )
        assert group.group_id == "G1"
        assert len(group.measurements) == 2


# ============================================================================
# Test SRDocument
# ============================================================================


class TestSRDocument:
    """Tests for SRDocument dataclass."""

    def test_creation(self) -> None:
        """Test creating SRDocument."""
        doc = SRDocument(
            file_path=Path("/test/sr.dcm"),
            sop_instance_uid="1.2.3.4",
            template_id="1500",
            document_title="Test Report",
        )
        assert doc.file_path == Path("/test/sr.dcm")
        assert doc.template_id == "1500"
        assert doc.measurement_groups == []

    def test_from_file_not_found(self) -> None:
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="SR file not found"):
            SRDocument.from_file("/nonexistent/path.dcm")

    def test_from_file_invalid_sop_class(self, tmp_path: Path) -> None:
        """Test ValueError for non-SR file."""
        dummy_file = tmp_path / "ct.dcm"
        dummy_file.write_bytes(b"dummy")

        with patch("pydicom.dcmread") as mock_read:
            mock_dcm = MagicMock()
            mock_dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
            mock_read.return_value = mock_dcm

            with pytest.raises(ValueError, match="not a DICOM SR document"):
                SRDocument.from_file(str(dummy_file))

    def test_from_file_success(self, tmp_path: Path) -> None:
        """Test successful SR file loading."""
        dummy_file = tmp_path / "sr.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_sr = create_mock_sr_dataset()

        with patch("pydicom.dcmread") as mock_read:
            mock_read.return_value = mock_sr

            doc = SRDocument.from_file(str(dummy_file))

            assert doc.sop_instance_uid == "1.2.3.4.5.6.7.8.9"
            assert doc.patient_id == "TEST_PATIENT"
            assert doc.template_id == "1500"
            assert doc.document_title == "Measurement Report"

    def test_get_measurements_df(self) -> None:
        """Test DataFrame export."""
        measurements = [
            SRMeasurement("M1", 10.0, "mm"),
            SRMeasurement("M2", 20.0, "mm"),
        ]
        group = SRMeasurementGroup(
            group_id="G1",
            finding_type="Lesion",
            measurements=measurements,
        )
        doc = SRDocument(
            file_path=Path("/test/sr.dcm"),
            sop_instance_uid="1.2.3",
            measurement_groups=[group],
        )

        df = doc.get_measurements_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "measurement_name" in df.columns
        assert "value" in df.columns
        assert "unit" in df.columns
        assert df.iloc[0]["measurement_name"] == "M1"
        assert df.iloc[0]["value"] == 10.0

    def test_get_summary(self) -> None:
        """Test summary generation."""
        measurements = [SRMeasurement("M1", 10.0, "mm")]
        group = SRMeasurementGroup(group_id="G1", measurements=measurements)
        doc = SRDocument(
            file_path=Path("/test/sr.dcm"),
            sop_instance_uid="1.2.3",
            template_id="1500",
            document_title="Test",
            measurement_groups=[group],
        )

        summary = doc.get_summary()

        assert summary["num_groups"] == 1
        assert summary["num_measurements"] == 1
        assert summary["template_id"] == "1500"

    def test_export_csv(self, tmp_path: Path) -> None:
        """Test CSV export."""
        measurements = [SRMeasurement("M1", 10.0, "mm")]
        group = SRMeasurementGroup(group_id="G1", measurements=measurements)
        doc = SRDocument(
            file_path=Path("/test/sr.dcm"),
            sop_instance_uid="1.2.3",
            measurement_groups=[group],
        )

        csv_path = tmp_path / "output.csv"
        result = doc.export_csv(str(csv_path))

        assert result == csv_path
        assert csv_path.exists()

        # Verify content
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["measurement_name"] == "M1"

    def test_export_json(self, tmp_path: Path) -> None:
        """Test JSON export."""
        measurements = [SRMeasurement("M1", 10.0, "mm")]
        group = SRMeasurementGroup(group_id="G1", measurements=measurements)
        doc = SRDocument(
            file_path=Path("/test/sr.dcm"),
            sop_instance_uid="1.2.3",
            template_id="1500",
            measurement_groups=[group],
        )

        json_path = tmp_path / "output.json"
        result = doc.export_json(str(json_path))

        assert result == json_path
        assert json_path.exists()

        # Verify content
        with open(json_path) as f:
            data = json.load(f)

        assert data["sop_instance_uid"] == "1.2.3"
        assert data["template_id"] == "1500"
        assert len(data["measurement_groups"]) == 1
        assert len(data["measurement_groups"][0]["measurements"]) == 1


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Tests for SR parser helper functions."""

    def test_extract_numeric_measurement(self) -> None:
        """Test extracting numeric measurement from SR item."""
        mock_item = MagicMock()

        mv = MagicMock()
        mv.NumericValue = 42.5

        unit = MagicMock()
        unit.CodeValue = "mm"
        mv.MeasurementUnitsCodeSequence = [unit]

        mock_item.MeasuredValueSequence = [mv]

        result = _extract_numeric_measurement(mock_item, "Test Measurement")

        assert result is not None
        assert result.name == "Test Measurement"
        assert result.value == 42.5
        assert result.unit == "mm"

    def test_extract_numeric_measurement_no_value(self) -> None:
        """Test returns None when no measured value."""
        mock_item = MagicMock()
        mock_item.MeasuredValueSequence = []

        result = _extract_numeric_measurement(mock_item, "Test")
        assert result is None

    def test_get_code_value(self) -> None:
        """Test extracting code value from CODE item."""
        mock_item = MagicMock()

        code = MagicMock()
        code.CodeMeaning = "Liver"
        code.CodeValue = "LIV001"
        mock_item.ConceptCodeSequence = [code]

        result = _get_code_value(mock_item)
        assert result == "Liver"

    def test_get_code_value_no_sequence(self) -> None:
        """Test returns None when no ConceptCodeSequence."""
        mock_item = MagicMock(spec=[])

        result = _get_code_value(mock_item)
        assert result is None

    def test_parse_content_sequence_empty(self) -> None:
        """Test parse with no ContentSequence."""
        mock_dcm = MagicMock(spec=[])

        result = _parse_content_sequence(mock_dcm)
        assert result == []


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Integration tests for SR parser."""

    def test_import_from_utilities(self) -> None:
        """Test importing SR classes from utilities package."""
        from pictologics.utilities import (
            SRDocument,
            SRMeasurement,
            SRMeasurementGroup,
            is_dicom_sr,
        )

        assert SRDocument is not None
        assert SRMeasurement is not None
        assert SRMeasurementGroup is not None
        assert is_dicom_sr is not None


# ============================================================================
# Additional Edge Case Tests for Coverage
# ============================================================================


class TestSREdgeCases:
    """Tests for edge cases and error paths in SR parser."""

    def test_from_file_read_error(self, tmp_path: Path) -> None:
        """Test ValueError for DICOM read error."""
        dummy_file = tmp_path / "bad.dcm"
        dummy_file.write_bytes(b"dummy")

        with patch("pydicom.dcmread") as mock_read:
            mock_read.side_effect = Exception("Read error")

            with pytest.raises(ValueError, match="Failed to read DICOM file"):
                SRDocument.from_file(str(dummy_file))

    def test_from_file_no_content_datetime(self, tmp_path: Path) -> None:
        """Test loading SR without ContentDate."""
        dummy_file = tmp_path / "sr.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_sr = create_mock_sr_dataset()
        mock_sr.ContentDate = None
        mock_sr.ContentTime = None

        with patch("pydicom.dcmread") as mock_read:
            mock_read.return_value = mock_sr

            doc = SRDocument.from_file(str(dummy_file))
            assert doc.content_datetime is None

    def test_from_file_with_private_tags(self, tmp_path: Path) -> None:
        """Test extracting private tags when enabled."""
        dummy_file = tmp_path / "sr.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_sr = create_mock_sr_dataset()

        # Add a mock private element
        private_elem = MagicMock()
        private_elem.tag = MagicMock()
        private_elem.tag.is_private = True
        private_elem.keyword = "PrivateTag"
        private_elem.value = "PrivateValue"
        mock_sr.__iter__ = MagicMock(return_value=iter([private_elem]))

        with patch("pydicom.dcmread") as mock_read:
            mock_read.return_value = mock_sr

            doc = SRDocument.from_file(str(dummy_file), extract_private_tags=True)
            assert "PrivateTag" in doc.metadata

    def test_extract_numeric_measurement_null_value(self) -> None:
        """Test returns None when NumericValue is None."""
        mock_item = MagicMock()
        mv = MagicMock()
        mv.NumericValue = None
        mock_item.MeasuredValueSequence = [mv]

        result = _extract_numeric_measurement(mock_item, "Test")
        assert result is None

    def test_extract_numeric_measurement_invalid_value(self) -> None:
        """Test returns None for non-numeric value."""
        mock_item = MagicMock()
        mv = MagicMock()
        mv.NumericValue = "not a number"  # Invalid
        mock_item.MeasuredValueSequence = [mv]

        result = _extract_numeric_measurement(mock_item, "Test")
        assert result is None

    def test_parse_content_sequence_with_code_items(self) -> None:
        """Test parsing CODE value type for finding site."""
        mock_dcm = MagicMock()

        # Create CONTAINER with CODE item
        container = MagicMock()
        container.RelationshipType = "CONTAINS"
        container.ValueType = "CONTAINER"
        container.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Group")]
        container.ContentSequence = []

        # Create CODE item with "site" in concept name
        code_item = MagicMock()
        code_item.RelationshipType = "CONTAINS"
        code_item.ValueType = "CODE"

        code_concept = MagicMock()
        code_concept.CodeMeaning = "Finding Site"
        code_item.ConceptNameCodeSequence = [code_concept]

        code_value = MagicMock()
        code_value.CodeMeaning = "Liver"
        code_item.ConceptCodeSequence = [code_value]

        # Create NUM item for measurement
        num_item = MagicMock()
        num_item.RelationshipType = "CONTAINS"
        num_item.ValueType = "NUM"
        num_concept = MagicMock()
        num_concept.CodeMeaning = "Volume"
        num_item.ConceptNameCodeSequence = [num_concept]

        mv = MagicMock()
        mv.NumericValue = 50.0
        unit = MagicMock()
        unit.CodeValue = "mL"
        mv.MeasurementUnitsCodeSequence = [unit]
        num_item.MeasuredValueSequence = [mv]

        mock_dcm.ContentSequence = [container, code_item, num_item]

        result = _parse_content_sequence(mock_dcm)
        assert len(result) >= 1

    def test_parse_content_sequence_with_text_items(self) -> None:
        """Test parsing TEXT value type for tracking ID."""
        mock_dcm = MagicMock()

        # Container
        container = MagicMock()
        container.RelationshipType = "CONTAINS"
        container.ValueType = "CONTAINER"
        container.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Group")]
        container.ContentSequence = []

        # TEXT item with tracking
        text_item = MagicMock()
        text_item.RelationshipType = "CONTAINS"
        text_item.ValueType = "TEXT"

        text_concept = MagicMock()
        text_concept.CodeMeaning = "Tracking ID"
        text_item.ConceptNameCodeSequence = [text_concept]
        text_item.TextValue = "TRACK123"

        # NUM item for measurement
        num_item = MagicMock()
        num_item.RelationshipType = "CONTAINS"
        num_item.ValueType = "NUM"
        num_concept = MagicMock()
        num_concept.CodeMeaning = "Score"
        num_item.ConceptNameCodeSequence = [num_concept]

        mv = MagicMock()
        mv.NumericValue = 100.0
        mv.MeasurementUnitsCodeSequence = []
        num_item.MeasuredValueSequence = [mv]

        mock_dcm.ContentSequence = [container, text_item, num_item]

        result = _parse_content_sequence(mock_dcm)
        assert len(result) >= 1

    def test_parse_content_sequence_orphan_measurements(self) -> None:
        """Test orphan measurements without container create default group."""
        mock_dcm = MagicMock()

        # Just NUM items without any container
        num_item = MagicMock()
        num_item.RelationshipType = "CONTAINS"
        num_item.ValueType = "NUM"
        num_concept = MagicMock()
        num_concept.CodeMeaning = "Value"
        num_item.ConceptNameCodeSequence = [num_concept]

        mv = MagicMock()
        mv.NumericValue = 42.0
        mv.MeasurementUnitsCodeSequence = []
        num_item.MeasuredValueSequence = [mv]

        mock_dcm.ContentSequence = [num_item]

        result = _parse_content_sequence(mock_dcm)
        assert len(result) == 1
        assert result[0].group_id == "default"

    def test_from_file_private_tag_exception(self, tmp_path: Path) -> None:
        """Test private tag extraction handles exceptions (lines 217-218)."""
        dummy_file = tmp_path / "sr.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_sr = create_mock_sr_dataset()

        # Create private element that raises exception on value access
        private_elem = MagicMock()
        private_elem.tag = MagicMock()
        private_elem.tag.is_private = True
        private_elem.keyword = None  # Will use str(tag)
        # Make value property raise an exception
        type(private_elem).value = property(
            lambda self: (_ for _ in ()).throw(Exception("Cannot read"))
        )
        mock_sr.__iter__ = MagicMock(return_value=iter([private_elem]))

        with patch("pydicom.dcmread") as mock_read:
            mock_read.return_value = mock_sr

            # Should not raise, just skip the problematic private tag
            doc = SRDocument.from_file(str(dummy_file), extract_private_tags=True)
            assert doc is not None

    def test_parse_content_sequence_container_saves_prior_group(self) -> None:
        """Test that encountering a new CONTAINER saves prior group (lines 409-411)."""
        mock_dcm = MagicMock()

        # First container with measurements
        container1 = MagicMock()
        container1.RelationshipType = "CONTAINS"
        container1.ValueType = "CONTAINER"
        container1.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Group1")]
        container1.ContentSequence = []

        # A NUM item that comes between containers (will be in first group)
        num_item1 = MagicMock()
        num_item1.RelationshipType = "CONTAINS"
        num_item1.ValueType = "NUM"
        num_item1.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Measure1")]
        mv1 = MagicMock()
        mv1.NumericValue = 10.0
        mv1.MeasurementUnitsCodeSequence = []
        num_item1.MeasuredValueSequence = [mv1]

        # Second container - should trigger saving of first group
        container2 = MagicMock()
        container2.RelationshipType = "CONTAINS"
        container2.ValueType = "CONTAINER"
        container2.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Group2")]
        container2.ContentSequence = []

        # Measurement for second group
        num_item2 = MagicMock()
        num_item2.RelationshipType = "CONTAINS"
        num_item2.ValueType = "NUM"
        num_item2.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Measure2")]
        mv2 = MagicMock()
        mv2.NumericValue = 20.0
        mv2.MeasurementUnitsCodeSequence = []
        num_item2.MeasuredValueSequence = [mv2]

        # Order: container1, num_item1, container2,  num_item2
        mock_dcm.ContentSequence = [container1, num_item1, container2, num_item2]

        result = _parse_content_sequence(mock_dcm)

        # Should have at least 2 groups (one saved when container2 encountered, one at end)
        assert len(result) >= 2

    def test_parse_content_sequence_finding_type_code(self) -> None:
        """Test parsing CODE with 'type' in concept name (lines 436-437)."""
        mock_dcm = MagicMock()

        # Container first (to establish current_group)
        container = MagicMock()
        container.RelationshipType = "CONTAINS"
        container.ValueType = "CONTAINER"
        container.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Group")]
        container.ContentSequence = []

        # CODE item with "type" in concept name
        code_item = MagicMock()
        code_item.RelationshipType = "CONTAINS"
        code_item.ValueType = "CODE"

        type_concept = MagicMock()
        type_concept.CodeMeaning = "Finding Type"
        code_item.ConceptNameCodeSequence = [type_concept]

        code_value = MagicMock()
        code_value.CodeMeaning = "Lesion"
        code_item.ConceptCodeSequence = [code_value]

        # NUM item for measurement (to create a group with measurements)
        num_item = MagicMock()
        num_item.RelationshipType = "CONTAINS"
        num_item.ValueType = "NUM"
        num_item.ConceptNameCodeSequence = [MagicMock(CodeMeaning="Size")]
        mv = MagicMock()
        mv.NumericValue = 15.0
        mv.MeasurementUnitsCodeSequence = []
        num_item.MeasuredValueSequence = [mv]

        mock_dcm.ContentSequence = [container, code_item, num_item]

        result = _parse_content_sequence(mock_dcm)

        # Should have parsed the finding type
        assert len(result) >= 1


# ============================================================================
# SRBatch and from_folders Tests
# ============================================================================


class TestSRBatch:
    """Tests for SRBatch class and SRDocument.from_folders()."""

    def test_srbatch_empty(self) -> None:
        """Test SRBatch with empty documents."""
        from pictologics.utilities.sr_parser import SRBatch

        batch = SRBatch(documents=[], processing_log=[], output_dir=None)
        assert len(batch.documents) == 0
        df = batch.get_combined_measurements_df()
        assert len(df) == 0

    def test_srbatch_get_combined_measurements_df(self) -> None:
        """Test SRBatch.get_combined_measurements_df()."""
        from pictologics.utilities.sr_parser import SRBatch

        # Create mock documents with measurements
        doc1 = SRDocument(
            file_path=Path("/test/sr1.dcm"),
            sop_instance_uid="1.2.3.4",
            patient_id="PATIENT1",
            study_instance_uid="1.2.3.5",
            measurement_groups=[
                SRMeasurementGroup(
                    finding_type="Lesion",
                    finding_site="Liver",
                    measurements=[
                        SRMeasurement(name="Volume", value=100.0, unit="mm3"),
                        SRMeasurement(name="Diameter", value=10.0, unit="mm"),
                    ],
                )
            ],
        )
        doc2 = SRDocument(
            file_path=Path("/test/sr2.dcm"),
            sop_instance_uid="1.2.3.6",
            patient_id="PATIENT2",
            study_instance_uid="1.2.3.7",
            measurement_groups=[
                SRMeasurementGroup(
                    measurements=[
                        SRMeasurement(name="Score", value=50.0, unit="1"),
                    ]
                )
            ],
        )

        batch = SRBatch(documents=[doc1, doc2], processing_log=[], output_dir=None)
        df = batch.get_combined_measurements_df()

        assert len(df) == 3  # 2 from doc1 + 1 from doc2
        assert "sop_instance_uid" in df.columns
        assert "patient_id" in df.columns
        assert "measurement_name" in df.columns
        assert df["value"].sum() == 160.0  # 100 + 10 + 50

    def test_srbatch_export_combined_csv(self) -> None:
        """Test SRBatch.export_combined_csv()."""
        from pictologics.utilities.sr_parser import SRBatch

        doc = SRDocument(
            file_path=Path("/test/sr1.dcm"),
            sop_instance_uid="1.2.3.4",
            measurement_groups=[
                SRMeasurementGroup(
                    measurements=[SRMeasurement(name="Test", value=1.0, unit="mm")]
                )
            ],
        )
        batch = SRBatch(documents=[doc], processing_log=[], output_dir=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "combined.csv"
            result = batch.export_combined_csv(output_path)
            assert result.exists()
            df = pd.read_csv(result)
            assert len(df) == 1

    def test_srbatch_export_log(self) -> None:
        """Test SRBatch.export_log()."""
        from pictologics.utilities.sr_parser import SRBatch

        log = [
            {"file_path": "/test/sr1.dcm", "status": "success", "num_measurements": 5},
            {
                "file_path": "/test/sr2.dcm",
                "status": "error",
                "error_message": "Bad file",
            },
        ]
        batch = SRBatch(documents=[], processing_log=log, output_dir=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "log.csv"
            result = batch.export_log(output_path)
            assert result.exists()
            df = pd.read_csv(result)
            assert len(df) == 2
            assert df.iloc[0]["status"] == "success"
            assert df.iloc[1]["status"] == "error"


class TestSRDocumentFromFolders:
    """Tests for SRDocument.from_folders() class method."""

    def test_from_folders_empty_folder(self) -> None:
        """Test from_folders with empty folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = SRDocument.from_folders([tmpdir], show_progress=False)
            assert len(batch.documents) == 0
            assert len(batch.processing_log) == 0

    def test_from_folders_nonexistent_folder(self) -> None:
        """Test from_folders with nonexistent folder."""
        batch = SRDocument.from_folders(["/nonexistent/path"], show_progress=False)
        assert len(batch.documents) == 0

    @patch("pictologics.utilities.sr_parser.is_dicom_sr")
    @patch("pictologics.utilities.sr_parser.SRDocument.from_file")
    def test_from_folders_with_sr_files(
        self, mock_from_file: MagicMock, mock_is_sr: MagicMock
    ) -> None:
        """Test from_folders discovers and processes SR files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake SR files
            sr1 = Path(tmpdir) / "sr1.dcm"
            sr2 = Path(tmpdir) / "sr2.dcm"
            sr1.touch()
            sr2.touch()

            # Mock is_dicom_sr to return True
            mock_is_sr.return_value = True

            # Mock from_file to return mock documents
            mock_doc = SRDocument(
                file_path=sr1,
                sop_instance_uid="1.2.3.4",
                measurement_groups=[],
            )
            mock_from_file.return_value = mock_doc

            batch = SRDocument.from_folders(
                [tmpdir], show_progress=False, num_workers=1
            )

            assert len(batch.documents) == 2
            assert mock_from_file.call_count == 2

    @patch("pictologics.utilities.sr_parser.is_dicom_sr")
    @patch("pictologics.utilities.sr_parser.SRDocument.from_file")
    def test_from_folders_with_exports(
        self, mock_from_file: MagicMock, mock_is_sr: MagicMock
    ) -> None:
        """Test from_folders with export_csv and export_json enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            sr1 = input_dir / "sr1.dcm"
            sr1.touch()

            mock_is_sr.return_value = True

            mock_doc = MagicMock(spec=SRDocument)
            mock_doc.sop_instance_uid = "1_2_3_4"
            mock_doc.patient_id = "PATIENT"
            mock_doc.study_instance_uid = "1.2.3.5"
            mock_doc.measurement_groups = []
            mock_from_file.return_value = mock_doc

            SRDocument.from_folders(
                [str(input_dir)],
                output_dir=str(output_dir),
                export_csv=True,
                export_json=True,
                show_progress=False,
                num_workers=1,
            )

            assert output_dir.exists()
            # Check that export methods were called
            mock_doc.export_csv.assert_called()
            mock_doc.export_json.assert_called()

    @patch("pictologics.utilities.sr_parser.is_dicom_sr")
    @patch("pictologics.utilities.sr_parser.SRDocument.from_file")
    def test_from_folders_handles_errors(
        self, mock_from_file: MagicMock, mock_is_sr: MagicMock
    ) -> None:
        """Test from_folders handles errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr1 = Path(tmpdir) / "bad_sr.dcm"
            sr1.touch()

            mock_is_sr.return_value = True
            mock_from_file.side_effect = ValueError("Bad SR file")

            batch = SRDocument.from_folders(
                [tmpdir], show_progress=False, num_workers=1
            )

            assert len(batch.documents) == 0
            assert len(batch.processing_log) == 1
            assert batch.processing_log[0]["status"] == "error"
            assert "Bad SR file" in batch.processing_log[0]["error_message"]

    @patch("pictologics.utilities.sr_parser.is_dicom_sr")
    def test_from_folders_with_single_file_path(self, mock_is_sr: MagicMock) -> None:
        """Test from_folders with a file path instead of folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr1 = Path(tmpdir) / "sr1.dcm"
            sr1.touch()

            # File exists but is not SR
            mock_is_sr.return_value = False

            batch = SRDocument.from_folders([str(sr1)], show_progress=False)
            assert len(batch.documents) == 0

    @patch("pictologics.utilities.sr_parser.is_dicom_sr")
    @patch("pictologics.utilities.sr_parser.SRDocument.from_file")
    def test_from_folders_single_file_is_sr(
        self, mock_from_file: MagicMock, mock_is_sr: MagicMock
    ) -> None:
        """Test from_folders with a single file path that IS an SR file (line 435)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr1 = Path(tmpdir) / "sr1.dcm"
            sr1.touch()

            # File exists AND is SR
            mock_is_sr.return_value = True

            mock_doc = SRDocument(
                file_path=sr1,
                sop_instance_uid="1.2.3.4",
                measurement_groups=[],
            )
            mock_from_file.return_value = mock_doc

            batch = SRDocument.from_folders(
                [str(sr1)], show_progress=False, num_workers=1
            )

            assert len(batch.documents) == 1
            assert mock_from_file.call_count == 1

    @patch("pictologics.utilities.sr_parser.is_dicom_sr")
    def test_from_folders_parallel_processing(self, mock_is_sr: MagicMock) -> None:
        """Test from_folders with parallel processing (num_workers > 1, lines 473-485).

        Note: Mocks don't work across process boundaries with ProcessPoolExecutor,
        so we verify the parallel path is exercised even if files fail to parse.
        The successful document append is marked pragma: no cover for this reason.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple fake SR files
            for i in range(3):
                sr_file = Path(tmpdir) / f"sr_{i}.dcm"
                sr_file.touch()

            mock_is_sr.return_value = True

            # Run with num_workers=2 to trigger parallel branch
            batch = SRDocument.from_folders(
                [tmpdir], show_progress=False, num_workers=2
            )

            # Verify parallel processing was attempted (files were found and processed)
            assert len(batch.processing_log) == 3
            # All should fail because these are empty files, but the parallel branch ran
            assert all(log["status"] == "error" for log in batch.processing_log)


class TestProcessSRFileWorker:
    """Tests for _process_sr_file_worker function."""

    def test_worker_success(self) -> None:
        """Test worker with successful processing."""
        from pictologics.utilities.sr_parser import _process_sr_file_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            sr_file = Path(tmpdir) / "test.dcm"
            sr_file.touch()

            with patch("pictologics.utilities.sr_parser.SRDocument.from_file") as mock:
                mock_doc = SRDocument(
                    file_path=sr_file,
                    sop_instance_uid="1.2.3.4",
                    patient_id="TEST",
                    study_instance_uid="1.2.3.5",
                    measurement_groups=[
                        SRMeasurementGroup(
                            measurements=[
                                SRMeasurement(name="Test", value=1.0, unit="mm")
                            ]
                        )
                    ],
                )
                mock.return_value = mock_doc

                result = _process_sr_file_worker((sr_file, False, None, False, False))

                assert result["document"] is not None
                assert result["log"]["status"] == "success"
                assert result["log"]["num_measurements"] == 1

    def test_worker_with_exports(self) -> None:
        """Test worker with exports enabled."""
        from pictologics.utilities.sr_parser import _process_sr_file_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            sr_file = Path(tmpdir) / "test.dcm"
            sr_file.touch()
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            mock_doc = MagicMock(spec=SRDocument)
            mock_doc.sop_instance_uid = "1.2.3.4"
            mock_doc.patient_id = "TEST"
            mock_doc.study_instance_uid = "1.2.3.5"
            mock_doc.measurement_groups = []

            with patch("pictologics.utilities.sr_parser.SRDocument.from_file") as mock:
                mock.return_value = mock_doc

                result = _process_sr_file_worker(
                    (sr_file, False, output_dir, True, True)
                )

                assert result["log"]["status"] == "success"
                mock_doc.export_csv.assert_called_once()
                mock_doc.export_json.assert_called_once()

    def test_worker_failure(self) -> None:
        """Test worker with processing failure."""
        from pictologics.utilities.sr_parser import _process_sr_file_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            sr_file = Path(tmpdir) / "bad.dcm"
            sr_file.touch()

            with patch("pictologics.utilities.sr_parser.SRDocument.from_file") as mock:
                mock.side_effect = Exception("Parse error")

                result = _process_sr_file_worker((sr_file, False, None, False, False))

                assert result["document"] is None
                assert result["log"]["status"] == "error"
                assert "Parse error" in result["log"]["error_message"]
