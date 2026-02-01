"""
Test suite for DICOM SEG Loader.

Tests cover all functionality including:
- SEG file detection
- Loading SEG files as Image objects
- Single and multi-segment handling
- Geometry extraction
- Reference image alignment
- Segment info extraction
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Disable JIT warmup for tests to prevent NumPy reload warning and speed up collection
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import numpy as np
import pytest

from pictologics.loader import Image, _is_dicom_seg
from pictologics.loaders.seg_loader import (
    _align_to_reference,
    _extract_combined_segments,
    _extract_seg_geometry,
    _extract_single_segment,
    get_segment_info,
    load_seg,
)

# ============================================================================
# Fixtures and Helpers
# ============================================================================


def create_mock_seg_dataset(
    rows: int = 64,
    cols: int = 64,
    n_frames: int = 10,
    n_segments: int = 2,
    pixel_spacing: tuple[float, float] = (0.5, 0.5),
    slice_thickness: float = 2.5,
) -> MagicMock:
    """Create a mock DICOM SEG dataset for testing."""
    mock_seg = MagicMock()
    mock_seg.Rows = rows
    mock_seg.Columns = cols

    # Create pixel array with shape (frames, rows, cols)
    # For n_segments and n_slices, total frames = n_segments * n_slices
    pixel_array = np.zeros((n_frames, rows, cols), dtype=np.uint8)

    # Fill with some test data - each segment gets different values
    for frame_idx in range(n_frames):
        # Add some non-zero region
        pixel_array[frame_idx, 20:40, 20:40] = 1

    mock_seg.pixel_array = pixel_array

    # Create SegmentSequence
    mock_seg.SegmentSequence = []
    for i in range(n_segments):
        segment = MagicMock()
        segment.SegmentNumber = i + 1
        segment.SegmentLabel = f"Segment {i + 1}"
        segment.SegmentDescription = f"Test segment {i + 1}"
        segment.SegmentAlgorithmType = "AUTOMATIC"
        mock_seg.SegmentSequence.append(segment)

    # Create SharedFunctionalGroupsSequence
    shared_fg = MagicMock()

    # PixelMeasuresSequence
    pm = MagicMock()
    pm.PixelSpacing = list(pixel_spacing)
    pm.SliceThickness = slice_thickness
    shared_fg.PixelMeasuresSequence = [pm]

    # PlaneOrientationSequence
    po = MagicMock()
    po.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    shared_fg.PlaneOrientationSequence = [po]

    mock_seg.SharedFunctionalGroupsSequence = [shared_fg]

    # Create PerFrameFunctionalGroupsSequence
    mock_seg.PerFrameFunctionalGroupsSequence = []
    for frame_idx in range(n_frames):
        frame_fg = MagicMock()

        # SegmentIdentificationSequence
        seg_id = MagicMock()
        seg_id.ReferencedSegmentNumber = (frame_idx % n_segments) + 1
        frame_fg.SegmentIdentificationSequence = [seg_id]

        # FrameContentSequence with DimensionIndexValues
        fc = MagicMock()
        slice_idx = frame_idx // n_segments + 1  # 1-indexed
        fc.DimensionIndexValues = [slice_idx, (frame_idx % n_segments) + 1]
        frame_fg.FrameContentSequence = [fc]

        # PlanePositionSequence
        pp = MagicMock()
        pp.ImagePositionPatient = [0.0, 0.0, float(slice_idx - 1) * slice_thickness]
        frame_fg.PlanePositionSequence = [pp]

        mock_seg.PerFrameFunctionalGroupsSequence.append(frame_fg)

    return mock_seg


# ============================================================================
# Test _is_dicom_seg
# ============================================================================


class TestIsDicomSeg:
    """Tests for _is_dicom_seg helper function."""

    def test_is_dicom_seg_true(self) -> None:
        """Test detection of DICOM SEG file."""
        with patch("pydicom.dcmread") as mock_read:
            mock_dcm = MagicMock()
            mock_dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
            mock_read.return_value = mock_dcm

            result = _is_dicom_seg("/path/to/file.dcm")
            assert result is True
            mock_read.assert_called_once_with(
                "/path/to/file.dcm", stop_before_pixels=True
            )

    def test_is_dicom_seg_false_different_sop(self) -> None:
        """Test non-SEG DICOM file returns False."""
        with patch("pydicom.dcmread") as mock_read:
            mock_dcm = MagicMock()
            mock_dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
            mock_read.return_value = mock_dcm

            result = _is_dicom_seg("/path/to/file.dcm")
            assert result is False

    def test_is_dicom_seg_false_no_sop(self) -> None:
        """Test file without SOPClassUID returns False."""
        with patch("pydicom.dcmread") as mock_read:
            mock_dcm = MagicMock(spec=[])
            mock_read.return_value = mock_dcm

            result = _is_dicom_seg("/path/to/file.dcm")
            assert result is False

    def test_is_dicom_seg_false_exception(self) -> None:
        """Test file read exception returns False."""
        with patch("pydicom.dcmread") as mock_read:
            mock_read.side_effect = Exception("Read error")

            result = _is_dicom_seg("/path/to/file.dcm")
            assert result is False


# ============================================================================
# Test _extract_seg_geometry
# ============================================================================


class TestExtractSegGeometry:
    """Tests for _extract_seg_geometry helper function."""

    def test_extract_geometry_full(self) -> None:
        """Test extracting geometry with all attributes present."""
        mock_seg = create_mock_seg_dataset(
            pixel_spacing=(0.5, 0.5),
            slice_thickness=2.5,
        )

        spacing, origin, direction = _extract_seg_geometry(mock_seg)

        # Spacing should be (col, row, slice) = (0.5, 0.5, 2.5)
        assert spacing[0] == pytest.approx(0.5)
        assert spacing[1] == pytest.approx(0.5)
        # Origin should be from first frame
        assert origin == (0.0, 0.0, 0.0)
        # Direction should be identity-like for standard orientation
        assert direction is not None
        assert direction.shape == (3, 3)

    def test_extract_geometry_missing_shared_fg(self) -> None:
        """Test fallback when SharedFunctionalGroupsSequence is missing."""
        mock_seg = MagicMock()
        mock_seg.SharedFunctionalGroupsSequence = None
        mock_seg.PerFrameFunctionalGroupsSequence = None

        spacing, origin, direction = _extract_seg_geometry(mock_seg)

        # Should return defaults
        assert spacing == (1.0, 1.0, 1.0)
        assert origin == (0.0, 0.0, 0.0)
        assert direction is None


# ============================================================================
# Test _extract_combined_segments
# ============================================================================


class TestExtractCombinedSegments:
    """Tests for _extract_combined_segments helper function."""

    def test_combine_segments(self) -> None:
        """Test combining multiple segments into single label array."""
        mock_seg = create_mock_seg_dataset(n_frames=10, n_segments=2)

        result = _extract_combined_segments(
            mock_seg,
            mock_seg.pixel_array,
            target_segments=[1, 2],
            n_frames=10,
        )

        # Result should be 3D
        assert result.ndim == 3
        # Should have values for segments
        unique_values = np.unique(result)
        assert 0 in unique_values  # Background

    def test_combine_single_segment(self) -> None:
        """Test extracting only one segment."""
        mock_seg = create_mock_seg_dataset(n_frames=10, n_segments=2)

        result = _extract_combined_segments(
            mock_seg,
            mock_seg.pixel_array,
            target_segments=[1],
            n_frames=10,
        )

        # Should only have segment 1 values
        unique_values = np.unique(result)
        assert 2 not in unique_values


# ============================================================================
# Test _extract_single_segment
# ============================================================================


class TestExtractSingleSegment:
    """Tests for _extract_single_segment helper function."""

    def test_extract_single_segment(self) -> None:
        """Test extracting a single segment as binary mask."""
        mock_seg = create_mock_seg_dataset(n_frames=10, n_segments=2)

        result = _extract_single_segment(
            mock_seg,
            mock_seg.pixel_array,
            segment_number=1,
            n_frames=10,
        )

        # Result should be 3D
        assert result.ndim == 3
        # Should be binary
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 1})


# ============================================================================
# Test load_seg
# ============================================================================


class TestLoadSeg:
    """Tests for load_seg function."""

    def test_load_seg_file_not_found(self) -> None:
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="SEG file not found"):
            load_seg("/nonexistent/path/to/file.dcm")

    def test_load_seg_invalid_seg(self, tmp_path: Path) -> None:
        """Test ValueError for non-SEG file."""
        # Create a dummy file
        dummy_file = tmp_path / "dummy.dcm"
        dummy_file.write_bytes(b"not a dicom file")

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.side_effect = Exception("Invalid SEG")

            with pytest.raises(ValueError, match="Failed to load DICOM SEG"):
                load_seg(str(dummy_file))

    def test_load_seg_combined(self, tmp_path: Path) -> None:
        """Test loading SEG with combine_segments=True."""
        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset()

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            result = load_seg(str(dummy_file), combine_segments=True)

            assert isinstance(result, Image)
            assert result.modality == "SEG"
            assert result.array.ndim == 3

    def test_load_seg_separate(self, tmp_path: Path) -> None:
        """Test loading SEG with combine_segments=False."""
        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset(n_segments=2)

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            result = load_seg(str(dummy_file), combine_segments=False)

            assert isinstance(result, dict)
            assert 1 in result
            assert 2 in result
            for _seg_num, img in result.items():
                assert isinstance(img, Image)
                assert img.modality == "SEG"

    def test_load_seg_specific_segments(self, tmp_path: Path) -> None:
        """Test loading specific segments."""
        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset(n_segments=3)

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            result = load_seg(
                str(dummy_file),
                segment_numbers=[1, 3],
                combine_segments=False,
            )

            assert isinstance(result, dict)
            assert 1 in result
            assert 3 in result
            assert 2 not in result

    def test_load_seg_invalid_segment_number(self, tmp_path: Path) -> None:
        """Test error for invalid segment number."""
        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset(n_segments=2)

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            with pytest.raises(ValueError, match="Segment 99 not found"):
                load_seg(str(dummy_file), segment_numbers=[99])


# ============================================================================
# Test _align_to_reference
# ============================================================================


class TestAlignToReference:
    """Tests for _align_to_reference helper function."""

    def test_align_to_reference(self) -> None:
        """Test aligning mask to reference geometry."""
        # Create a small mask
        mask = Image(
            array=np.ones((32, 32, 10), dtype=np.uint8),
            spacing=(1.0, 1.0, 2.0),
            origin=(10.0, 10.0, 0.0),
            modality="SEG",
        )

        # Create a larger reference
        reference = Image(
            array=np.zeros((64, 64, 20), dtype=np.float64),
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
            modality="CT",
        )

        result = _align_to_reference(mask, reference)

        assert result.array.shape == reference.array.shape
        assert result.spacing == reference.spacing
        assert result.origin == reference.origin


# ============================================================================
# Test get_segment_info
# ============================================================================


class TestGetSegmentInfo:
    """Tests for get_segment_info function."""

    def test_get_segment_info(self, tmp_path: Path) -> None:
        """Test extracting segment information."""
        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset(n_segments=3)

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            info = get_segment_info(str(dummy_file))

            assert len(info) == 3
            assert info[0]["segment_number"] == 1
            assert info[0]["segment_label"] == "Segment 1"
            assert info[1]["segment_number"] == 2
            assert info[2]["segment_number"] == 3

    def test_get_segment_info_file_not_found(self) -> None:
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="SEG file not found"):
            get_segment_info("/nonexistent/path/to/file.dcm")

    def test_get_segment_info_invalid_seg(self, tmp_path: Path) -> None:
        """Test ValueError for non-SEG file."""
        dummy_file = tmp_path / "dummy.dcm"
        dummy_file.write_bytes(b"not a dicom file")

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.side_effect = Exception("Invalid SEG")

            with pytest.raises(ValueError, match="Failed to load DICOM SEG"):
                get_segment_info(str(dummy_file))


# ============================================================================
# Test Integration with load_image
# ============================================================================


class TestLoadImageIntegration:
    """Tests for SEG auto-detection in load_image."""

    def test_load_image_detects_seg(self, tmp_path: Path) -> None:
        """Test that load_image auto-detects and routes SEG files."""
        from pictologics.loader import load_image

        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset()

        with (
            patch("pictologics.loader._is_dicom_seg") as mock_is_seg,
            patch("highdicom.seg.segread") as mock_read,
        ):
            mock_is_seg.return_value = True
            mock_read.return_value = mock_seg

            result = load_image(str(dummy_file))

            assert isinstance(result, Image)
            assert result.modality == "SEG"

    def test_load_image_non_seg_dicom(self, tmp_path: Path) -> None:
        """Test that non-SEG DICOM files are loaded normally."""
        # This test would need real DICOM files or more extensive mocking
        # Skip for now as it requires more complex setup
        pass


# ============================================================================
# Additional Edge Case Tests for Coverage
# ============================================================================


class TestLoadSegWithReference:
    """Tests for load_seg with reference_image parameter."""

    def test_load_seg_combined_with_reference(self, tmp_path: Path) -> None:
        """Test loading SEG with reference alignment (combine_segments=True)."""
        from pictologics.loader import Image

        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset()

        # Create a larger reference image
        reference = Image(
            array=np.zeros((128, 128, 20), dtype=np.float64),
            spacing=(0.5, 0.5, 2.5),
            origin=(0.0, 0.0, 0.0),
            modality="CT",
        )

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            result = load_seg(
                str(dummy_file),
                combine_segments=True,
                reference_image=reference,
            )

            assert isinstance(result, Image)
            assert result.array.shape == reference.array.shape
            assert result.origin == reference.origin

    def test_load_seg_separate_with_reference(self, tmp_path: Path) -> None:
        """Test loading SEG with reference alignment (combine_segments=False)."""
        from pictologics.loader import Image

        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = create_mock_seg_dataset(n_segments=2)

        reference = Image(
            array=np.zeros((128, 128, 20), dtype=np.float64),
            spacing=(0.5, 0.5, 2.5),
            origin=(0.0, 0.0, 0.0),
            modality="CT",
        )

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            result = load_seg(
                str(dummy_file),
                combine_segments=False,
                reference_image=reference,
            )

            assert isinstance(result, dict)
            for img in result.values():
                assert img.array.shape == reference.array.shape


class TestEdgeCases:
    """Tests for edge cases and error paths."""

    def test_load_seg_no_segment_sequence(self, tmp_path: Path) -> None:
        """Test error when SEG lacks SegmentSequence."""
        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = MagicMock()
        # Remove SegmentSequence attribute
        del mock_seg.SegmentSequence

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            with pytest.raises(ValueError, match="not a valid DICOM SEG"):
                load_seg(str(dummy_file))

    def test_get_segment_info_no_segment_sequence(self, tmp_path: Path) -> None:
        """Test error when getting info from SEG lacking SegmentSequence."""
        dummy_file = tmp_path / "seg.dcm"
        dummy_file.write_bytes(b"dummy")

        mock_seg = MagicMock()
        del mock_seg.SegmentSequence

        with patch("highdicom.seg.segread") as mock_read:
            mock_read.return_value = mock_seg

            with pytest.raises(ValueError, match="not a valid DICOM SEG"):
                get_segment_info(str(dummy_file))

    def test_extract_geometry_with_slice_spacing_calculation(self) -> None:
        """Test geometry extraction calculates slice spacing from positions."""
        mock_seg = create_mock_seg_dataset(
            n_frames=10,
            n_segments=2,
            slice_thickness=3.0,
        )

        spacing, origin, direction = _extract_seg_geometry(mock_seg)

        # Should have calculated spacing from frame positions
        assert len(spacing) == 3
        assert spacing[2] > 0  # Slice spacing should be positive

    def test_extract_combined_segments_2d_pixel_array(self) -> None:
        """Test handling of 2D pixel array (single frame)."""
        mock_seg = MagicMock()
        mock_seg.Rows = 64
        mock_seg.Columns = 64
        mock_seg.SegmentSequence = [MagicMock()]
        mock_seg.SegmentSequence[0].SegmentNumber = 1
        mock_seg.PerFrameFunctionalGroupsSequence = []

        # 2D array (single frame)
        pixel_array = np.ones((64, 64), dtype=np.uint8)

        result = _extract_combined_segments(
            mock_seg,
            pixel_array,
            target_segments=[1],
            n_frames=1,
        )

        assert result.ndim == 3

    def test_extract_single_segment_2d_pixel_array(self) -> None:
        """Test single segment extraction with 2D pixel array."""
        mock_seg = MagicMock()
        mock_seg.Rows = 64
        mock_seg.Columns = 64
        mock_seg.SegmentSequence = [MagicMock()]
        mock_seg.SegmentSequence[0].SegmentNumber = 1
        mock_seg.PerFrameFunctionalGroupsSequence = []

        pixel_array = np.ones((64, 64), dtype=np.uint8)

        result = _extract_single_segment(
            mock_seg,
            pixel_array,
            segment_number=1,
            n_frames=1,
        )

        assert result.ndim == 3

    def test_extract_geometry_positive_median_spacing(self) -> None:
        """Test that positive median spacing is applied correctly (line 269)."""
        mock_seg = MagicMock()

        # SharedFunctionalGroupsSequence with pixel measures
        shared_fg = MagicMock()
        pm = MagicMock()
        pm.PixelSpacing = [0.5, 0.5]
        pm.SliceThickness = 1.0
        shared_fg.PixelMeasuresSequence = [pm]
        shared_fg.PlaneOrientationSequence = []
        mock_seg.SharedFunctionalGroupsSequence = [shared_fg]

        # Create distinct positions with positive spacing
        mock_seg.PerFrameFunctionalGroupsSequence = []
        for i in range(5):
            frame_fg = MagicMock()
            pp = MagicMock()
            # Position at z = i * 2.5 (positive spacing of 2.5)
            pp.ImagePositionPatient = [0.0, 0.0, float(i * 2.5)]
            frame_fg.PlanePositionSequence = [pp]
            mock_seg.PerFrameFunctionalGroupsSequence.append(frame_fg)

        spacing, origin, direction = _extract_seg_geometry(mock_seg)

        # Should have calculated the median spacing = 2.5
        assert spacing[2] == pytest.approx(2.5, rel=0.1)

    def test_extract_combined_segments_slice_bounds_check(self) -> None:
        """Test that slice_idx >= n_slices is handled correctly (line 346).

        When FrameContentSequence is missing, frame_to_slice stays empty.
        n_slices is estimated as n_frames // n_segments.
        slice_idx defaults to frame_idx. If n_frames > n_slices, bounds check triggers.
        """
        mock_seg = MagicMock()
        mock_seg.Rows = 32
        mock_seg.Columns = 32
        # 3 segments leads to n_slices = 6 frames // 3 segments = 2 slices
        mock_seg.SegmentSequence = [MagicMock(), MagicMock(), MagicMock()]
        mock_seg.SegmentSequence[0].SegmentNumber = 1
        mock_seg.SegmentSequence[1].SegmentNumber = 2
        mock_seg.SegmentSequence[2].SegmentNumber = 3

        # 6 frames but no FrameContentSequence
        mock_seg.PerFrameFunctionalGroupsSequence = []
        for i in range(6):
            frame_fg = MagicMock()
            seg_id = MagicMock()
            seg_id.ReferencedSegmentNumber = (i % 3) + 1
            frame_fg.SegmentIdentificationSequence = [seg_id]
            # No FrameContentSequence - frame_to_slice stays empty
            frame_fg.FrameContentSequence = []
            mock_seg.PerFrameFunctionalGroupsSequence.append(frame_fg)

        # 6 frames
        pixel_array = np.ones((6, 32, 32), dtype=np.uint8)

        # n_slices = 6 // 3 = 2
        # frames 0,1 ok; frames 2,3,4,5 have slice_idx >= 2, should be skipped
        result = _extract_combined_segments(
            mock_seg,
            pixel_array,
            target_segments=[1],
            n_frames=6,
        )

        assert result.ndim == 3
        # Output array should have shape (2, 32, 32)
        assert result.shape[0] == 2

    def test_extract_single_segment_slice_bounds_check(self) -> None:
        """Test that slice_idx >= n_slices is handled correctly (line 424).

        Similar setup: empty FrameContentSequence leads to frame_idx as slice_idx,
        and when frame_idx >= n_slices, the frame should be skipped.
        """
        mock_seg = MagicMock()
        mock_seg.Rows = 32
        mock_seg.Columns = 32
        # 2 segments leads to n_slices = 6 frames // 2 segments = 3 slices
        mock_seg.SegmentSequence = [MagicMock(), MagicMock()]
        mock_seg.SegmentSequence[0].SegmentNumber = 1
        mock_seg.SegmentSequence[1].SegmentNumber = 2

        # 6 frames but no FrameContentSequence
        mock_seg.PerFrameFunctionalGroupsSequence = []
        for i in range(6):
            frame_fg = MagicMock()
            seg_id = MagicMock()
            seg_id.ReferencedSegmentNumber = (i % 2) + 1
            frame_fg.SegmentIdentificationSequence = [seg_id]
            # No FrameContentSequence
            frame_fg.FrameContentSequence = []
            mock_seg.PerFrameFunctionalGroupsSequence.append(frame_fg)

        pixel_array = np.ones((6, 32, 32), dtype=np.uint8)

        # n_slices = 6 // 2 = 3
        # frames 0,1,2 ok for segment 1 (indices 0,2,4); frames 3,4,5 have slice_idx >= 3
        result = _extract_single_segment(
            mock_seg,
            pixel_array,
            segment_number=1,
            n_frames=6,
        )

        assert result.ndim == 3
        # Output array should have shape (3, 32, 32)
        assert result.shape[0] == 3
