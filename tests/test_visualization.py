"""
Test suite for Mask Visualization Utilities.

Tests cover all functionality including:
- RGBA overlay creation
- Slice selection parsing
- Batch export of overlay images
- Interactive viewer (mocked)
"""

import tempfile
import os
from pathlib import Path

# Disable JIT warmup for tests to prevent NumPy reload warning and speed up collection
os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

import numpy as np
import pytest

from unittest.mock import MagicMock, patch

from pictologics.loader import Image
from pictologics.utilities.mask_visualization import (
    COLORMAPS,
    _create_overlay_rgba,
    _get_colormap_colors,
    _normalize_image,
    _parse_slice_selection,
    save_mask_overlay_slices,
    visualize_mask_overlay,
)

# ============================================================================
# Test Helper Functions
# ============================================================================


class TestNormalizeImage:
    """Tests for _normalize_image."""

    def test_normalize_constant_array(self) -> None:
        """Test normalization of constant array."""
        arr = np.ones((10, 10)) * 100
        result = _normalize_image(arr)
        assert result.dtype == np.uint8
        assert np.all(result == 0)  # Constant becomes 0

    def test_normalize_range_array(self) -> None:
        """Test normalization of array with range."""
        arr = np.array([[0, 50], [100, 200]])
        result = _normalize_image(arr)
        assert result.dtype == np.uint8
        assert np.min(result) == 0
        assert np.max(result) == 255

    def test_normalize_negative_values(self) -> None:
        """Test normalization with negative values."""
        arr = np.array([[-100, 0], [100, 200]])
        result = _normalize_image(arr)
        assert result.dtype == np.uint8
        assert np.min(result) == 0
        assert np.max(result) == 255


class TestGetColormapColors:
    """Tests for _get_colormap_colors."""

    def test_get_tab10(self) -> None:
        """Test getting tab10 colormap."""
        colors = _get_colormap_colors("tab10")
        assert len(colors) == 10
        assert all(len(c) == 3 for c in colors)

    def test_get_tab20(self) -> None:
        """Test getting tab20 colormap."""
        colors = _get_colormap_colors("tab20")
        assert len(colors) == 20

    def test_get_unknown_returns_tab20(self) -> None:
        """Test unknown colormap defaults to tab20."""
        colors = _get_colormap_colors("unknown_colormap")
        assert colors == COLORMAPS["tab20"]


class TestCreateOverlayRgba:
    """Tests for _create_overlay_rgba."""

    def test_basic_overlay(self) -> None:
        """Test basic overlay creation."""
        img = np.random.rand(64, 64) * 1000
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 1

        result = _create_overlay_rgba(img, mask, alpha=0.5)
        assert result.shape == (64, 64, 4)
        assert result.dtype == np.uint8
        assert np.min(result[..., 3]) == 255  # Fully opaque

    def test_multi_label_mask(self) -> None:
        """Test overlay with multiple mask labels."""
        img = np.zeros((64, 64))
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2
        mask[50:60, 50:60] = 3

        result = _create_overlay_rgba(img, mask, alpha=0.5)
        assert result.shape == (64, 64, 4)

    def test_background_unchanged(self) -> None:
        """Test that background (label 0) is unchanged."""
        img = np.ones((10, 10)) * 128
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[5:, 5:] = 1

        result = _create_overlay_rgba(img, mask, alpha=0.5)
        # Check background pixel (should be gray)
        assert result[0, 0, 0] == result[0, 0, 1] == result[0, 0, 2]

    def test_different_colormaps(self) -> None:
        """Test overlay with different colormaps."""
        img = np.random.rand(32, 32) * 100
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[10:20, 10:20] = 1

        for cmap in ["tab10", "tab20", "Set1", "Set2", "Paired"]:
            result = _create_overlay_rgba(img, mask, colormap=cmap)
            assert result.shape == (32, 32, 4)


# ============================================================================
# Test Slice Selection Parsing
# ============================================================================


class TestParseSliceSelection:
    """Tests for _parse_slice_selection."""

    def test_integer_selection(self) -> None:
        """Test single integer slice selection."""
        result = _parse_slice_selection(5, 100)
        assert result == [5]

    def test_list_selection(self) -> None:
        """Test list of slice indices."""
        result = _parse_slice_selection([0, 10, 50, 99], 100)
        assert result == [0, 10, 50, 99]

    def test_list_filters_out_of_range(self) -> None:
        """Test that out-of-range indices are filtered."""
        result = _parse_slice_selection([0, 50, 150], 100)
        assert result == [0, 50]

    def test_percentage_selection(self) -> None:
        """Test percentage-based slice selection."""
        result = _parse_slice_selection("10%", 100)
        assert len(result) == 10
        assert 0 in result

    def test_every_n_selection(self) -> None:
        """Test every N slice selection."""
        result = _parse_slice_selection("every_10", 100)
        assert result == list(range(0, 100, 10))

    def test_numeric_string_selection(self) -> None:
        """Test numeric string as every N."""
        result = _parse_slice_selection("5", 50)
        assert result == list(range(0, 50, 5))

    def test_invalid_percentage(self) -> None:
        """Test invalid percentage defaults to [0]."""
        result = _parse_slice_selection("abc%", 100)
        assert result == [0]

    def test_zero_percentage(self) -> None:
        """Test 0% returns [0]."""
        result = _parse_slice_selection("0%", 100)
        assert result == [0]

    def test_zero_every_n(self) -> None:
        """Test every_0 returns [0]."""
        result = _parse_slice_selection("every_0", 100)
        assert result == [0]

    def test_invalid_every_n(self) -> None:
        """Test invalid every_N returns [0]."""
        # Covers the ValueError block in _parse_slice_selection
        result = _parse_slice_selection("every_invalid", 100)
        assert result == [0]

    def test_invalid_type_selection(self) -> None:
        """Test invalid type returns [0]."""
        # Covers the final return [0] block
        result = _parse_slice_selection(None, 100)  # type: ignore
        assert result == [0]
        result = _parse_slice_selection(3.14, 100)  # type: ignore
        assert result == [0]


# ============================================================================
# Test Save Overlay Slices
# ============================================================================


class TestSaveOverlaySlices:
    """Tests for save_mask_overlay_slices."""

    @pytest.fixture
    def synthetic_image(self) -> Image:
        """Create synthetic image."""
        arr = np.random.rand(32, 32, 20).astype(np.float32) * 1000
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    @pytest.fixture
    def synthetic_mask(self) -> Image:
        """Create synthetic mask with multiple labels."""
        arr = np.zeros((32, 32, 20), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        arr[15:25, 15:25, 5:15] = 2
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    def test_save_png_slices(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving PNG slices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_mask_overlay_slices(
                synthetic_image,
                synthetic_mask,
                tmpdir,
                slice_selection=[0, 10, 19],
                format="png",
            )
            assert len(files) == 3
            for f in files:
                assert Path(f).exists()
                assert f.endswith(".png")

    def test_save_jpeg_slices(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving JPEG slices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_mask_overlay_slices(
                synthetic_image,
                synthetic_mask,
                tmpdir,
                slice_selection="every_5",
                format="jpeg",
            )
            assert len(files) > 0
            for f in files:
                assert Path(f).exists()
                assert f.endswith(".jpg")

    def test_save_tiff_slices(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving TIFF slices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_mask_overlay_slices(
                synthetic_image,
                synthetic_mask,
                tmpdir,
                slice_selection="50%",
                format="tiff",
            )
            assert len(files) > 0
            for f in files:
                assert Path(f).exists()
                assert f.endswith(".tiff")

    def test_save_with_dpi(self, synthetic_image: Image, synthetic_mask: Image) -> None:
        """Test saving with different DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_mask_overlay_slices(
                synthetic_image,
                synthetic_mask,
                tmpdir,
                slice_selection=[10],
                dpi=300,
            )
            assert len(files) == 1
            assert Path(files[0]).exists()

    def test_save_with_dpi_72(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving with 72 DPI (scale factor 1.0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_mask_overlay_slices(
                synthetic_image,
                synthetic_mask,
                tmpdir,
                slice_selection=[10],
                dpi=72,
            )
            assert len(files) == 1
            assert Path(files[0]).exists()

    def test_different_axes(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test slicing along different axes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for axis in [0, 1, 2]:
                files = save_mask_overlay_slices(
                    synthetic_image,
                    synthetic_mask,
                    tmpdir,
                    slice_selection=[0],
                    axis=axis,
                    filename_prefix=f"axis{axis}",
                )
                assert len(files) == 1

    def test_shape_mismatch_error(self) -> None:
        """Test error on shape mismatch."""
        img = Image(
            array=np.zeros((32, 32, 20)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        mask = Image(
            array=np.zeros((32, 32, 10)),  # Different shape
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="does not match"):
                save_mask_overlay_slices(img, mask, tmpdir)

    def test_invalid_format_defaults_to_png(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test invalid format defaults to PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_mask_overlay_slices(
                synthetic_image,
                synthetic_mask,
                tmpdir,
                slice_selection=[0],
                format="invalid",
            )
            assert files[0].endswith(".png")

    def test_jpg_alias(self, synthetic_image: Image, synthetic_mask: Image) -> None:
        """Test jpg is aliased to jpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_mask_overlay_slices(
                synthetic_image,
                synthetic_mask,
                tmpdir,
                slice_selection=[0],
                format="jpg",
            )
            assert files[0].endswith(".jpg")


# ============================================================================
# Test Interactive Viewer (Mocked)
# ============================================================================


class TestVisualizeOverlay:
    """Tests for visualize_mask_overlay (mocked matplotlib)."""

    @pytest.fixture
    def synthetic_image(self) -> Image:
        """Create synthetic image."""
        arr = np.random.rand(32, 32, 20).astype(np.float32) * 1000
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    @pytest.fixture
    def synthetic_mask(self) -> Image:
        """Create synthetic mask."""
        arr = np.zeros((32, 32, 20), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    def test_visualize_shape_mismatch_error(self) -> None:
        """Test error on shape mismatch."""
        img = Image(
            array=np.zeros((32, 32, 20)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        mask = Image(
            array=np.zeros((32, 32, 10)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        with pytest.raises(ValueError, match="does not match"):
            visualize_mask_overlay(img, mask)

    def test_visualize_interactive_elements(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """
        Test interactive elements of visualization using mocks.

        This verifies that:
        1. Slider update triggers image update
        2. Scroll events trigger slider update
        3. Initial slice is set correctly
        """
        # Create mocks
        mock_figure = MagicMock()
        mock_axes = MagicMock()
        mock_slider_axes = MagicMock()
        mock_slider = MagicMock()
        mock_image_display = MagicMock()

        # Setup mock returns
        mock_figure.canvas = MagicMock()
        mock_figure.canvas.mpl_connect = MagicMock()
        mock_slider.val = 0  # Initial slider value

        # Store callbacks to call them later
        stored_callbacks = {}

        def capture_connect(event_name, callback):
            stored_callbacks[event_name] = callback

        mock_figure.canvas.mpl_connect.side_effect = capture_connect

        # Patch everything
        with patch(
            "matplotlib.pyplot.subplots", return_value=(mock_figure, mock_axes)
        ), patch("matplotlib.pyplot.axes", return_value=mock_slider_axes), patch(
            "matplotlib.widgets.Slider", return_value=mock_slider
        ), patch(
            "matplotlib.pyplot.show"
        ), patch(
            "matplotlib.pyplot.subplots_adjust"
        ):

            # Setup image display mock
            mock_axes.imshow.return_value = mock_image_display

            # Run visualization
            visualize_mask_overlay(
                synthetic_image, synthetic_mask, initial_slice=5, axis=2
            )

            # 1. Verify initialization
            mock_axes.imshow.assert_called_once()
            mock_axes.set_title.assert_called()

            # Check slider initialization
            mock_slider.on_changed.assert_called_once()
            update_callback = mock_slider.on_changed.call_args[0][0]

            # 2. Test Slider Update
            # Call the update function as if dragging the slider
            update_callback(10.0)

            # Verify image data was updated
            mock_image_display.set_data.assert_called()
            # Verify title was updated with new slice index
            assert "10/" in mock_axes.set_title.call_args_list[-1][0][0]

            # 3. Test Scroll Wheel
            assert "scroll_event" in stored_callbacks
            scroll_callback = stored_callbacks["scroll_event"]

            # Test scroll UP
            mock_event_up = MagicMock()
            mock_event_up.button = "up"
            mock_slider.val = 5

            scroll_callback(mock_event_up)
            # Should increment slider
            mock_slider.set_val.assert_called_with(6)

            # Test scroll DOWN
            mock_event_down = MagicMock()
            mock_event_down.button = "down"
            mock_slider.val = 5

            scroll_callback(mock_event_down)
            # Should decrement slider
            mock_slider.set_val.assert_called_with(4)

            # Test scroll limits (max)
            mock_slider.val = 19  # Max is 19 for 20 slices
            scroll_callback(mock_event_up)
            mock_slider.set_val.assert_called_with(19)

            # Test scroll limits (min)
            mock_slider.val = 0
            scroll_callback(mock_event_down)
            mock_slider.set_val.assert_called_with(0)

    def test_visualize_axis_support(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test visualization initialization for different axes."""
        mock_figure = MagicMock()
        mock_axes = MagicMock()

        with patch(
            "matplotlib.pyplot.subplots", return_value=(mock_figure, mock_axes)
        ), patch("matplotlib.pyplot.axes"), patch("matplotlib.widgets.Slider"), patch(
            "matplotlib.pyplot.show"
        ), patch(
            "matplotlib.pyplot.subplots_adjust"
        ):

            # Test Coronal (axis 1)
            visualize_mask_overlay(synthetic_image, synthetic_mask, axis=1)
            mock_axes.imshow.assert_called()

            # Test Sagittal (axis 0)
            visualize_mask_overlay(synthetic_image, synthetic_mask, axis=0)
            mock_axes.imshow.assert_called()
