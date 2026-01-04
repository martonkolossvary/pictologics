"""
Test suite for Visualization Utilities.

Tests cover all functionality including:
- RGBA display creation (image-only, mask-only, overlay)
- Window/level normalization
- Slice selection parsing
- Batch export of images
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
from pictologics.utilities.visualization import (
    COLORMAPS,
    _apply_window_level,
    _create_display_rgba,
    _get_colormap_colors,
    _get_reference_array,
    _normalize_image,
    _parse_slice_selection,
    save_slices,
    visualize_slices,
)

# ============================================================================
# Test Helper Functions
# ============================================================================


class TestApplyWindowLevel:
    """Tests for _apply_window_level."""

    def test_basic_window_level(self) -> None:
        """Test basic window/level application."""
        arr = np.array([[0, 100], [200, 400]])
        result = _apply_window_level(arr, center=200, width=200)
        assert result.dtype == np.uint8
        # center=200, width=200 -> range [100, 300]
        # 100 -> 0, 200 -> 127, 300 -> 255
        assert result[0, 0] == 0  # 0 clipped to 0
        assert result[0, 1] == 0  # 100 is at min
        assert result[1, 0] == 127  # 200 is at center
        assert result[1, 1] == 255  # 400 clipped to 255

    def test_narrow_window(self) -> None:
        """Test narrow window clips values."""
        arr = np.array([[0, 50], [100, 200]])
        result = _apply_window_level(arr, center=100, width=50)
        # range [75, 125]
        assert result[0, 0] == 0  # clipped to min
        assert result[1, 1] == 255  # clipped to max


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

    def test_normalize_with_window_level(self) -> None:
        """Test normalization with window/level parameters."""
        arr = np.array([[0, 100], [200, 400]])
        result = _normalize_image(arr, window_center=200, window_width=200)
        assert result.dtype == np.uint8
        # Should use window/level instead of min-max


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


class TestCreateDisplayRgba:
    """Tests for _create_display_rgba."""

    def test_overlay_mode(self) -> None:
        """Test overlay mode (image + mask)."""
        img = np.random.rand(64, 64) * 1000
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 1

        result = _create_display_rgba(img, mask, alpha=0.5)
        assert result.shape == (64, 64, 4)
        assert result.dtype == np.uint8
        assert np.min(result[..., 3]) == 255  # Fully opaque

    def test_image_only_mode(self) -> None:
        """Test image-only mode (no mask)."""
        img = np.random.rand(64, 64) * 1000

        result = _create_display_rgba(img, None)
        assert result.shape == (64, 64, 4)
        assert result.dtype == np.uint8
        # Should be grayscale (R=G=B)
        assert np.allclose(result[..., 0], result[..., 1])
        assert np.allclose(result[..., 1], result[..., 2])

    def test_mask_only_colormap_mode(self) -> None:
        """Test mask-only mode with colormap."""
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2

        result = _create_display_rgba(None, mask, mask_as_colormap=True)
        assert result.shape == (64, 64, 4)
        # Background should be black
        assert np.all(result[0, 0, :3] == 0)
        # Labels should have color
        assert not np.all(result[15, 15, :3] == 0)

    def test_mask_only_grayscale_mode(self) -> None:
        """Test mask-only mode with grayscale."""
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 128
        mask[30:40, 30:40] = 255

        result = _create_display_rgba(None, mask, mask_as_colormap=False)
        assert result.shape == (64, 64, 4)
        # Should be grayscale
        assert np.allclose(result[..., 0], result[..., 1])

    def test_neither_provided_raises(self) -> None:
        """Test that providing neither raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            _create_display_rgba(None, None)

    def test_multi_label_mask(self) -> None:
        """Test multi-label mask overlay."""
        img = np.ones((64, 64)) * 100
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2
        mask[50:60, 50:60] = 3

        result = _create_display_rgba(img, mask, alpha=0.5)
        assert result.shape == (64, 64, 4)


class TestParseSliceSelection:
    """Tests for _parse_slice_selection."""

    def test_single_int(self) -> None:
        """Test single integer selection."""
        result = _parse_slice_selection(50, 100)
        assert result == [50]

    def test_list_of_ints(self) -> None:
        """Test list of integers."""
        result = _parse_slice_selection([0, 25, 50, 75], 100)
        assert result == [0, 25, 50, 75]

    def test_list_filters_out_of_range(self) -> None:
        """Test list filters out-of-range values."""
        result = _parse_slice_selection([0, 50, 150], 100)
        assert result == [0, 50]

    def test_percentage_string(self) -> None:
        """Test percentage string."""
        result = _parse_slice_selection("10%", 100)
        assert len(result) == 10
        assert 0 in result

    def test_every_n_string(self) -> None:
        """Test every_N string."""
        result = _parse_slice_selection("every_10", 100)
        assert len(result) == 10

    def test_plain_number_string(self) -> None:
        """Test plain number string."""
        result = _parse_slice_selection("5", 100)
        assert len(result) == 20

    def test_invalid_percentage(self) -> None:
        """Test invalid percentage returns [0]."""
        result = _parse_slice_selection("-5%", 100)
        assert result == [0]

    def test_invalid_every_n(self) -> None:
        """Test invalid every_N returns [0]."""
        result = _parse_slice_selection("every_0", 100)
        assert result == [0]

    def test_invalid_type(self) -> None:
        """Test invalid type returns [0]."""
        result = _parse_slice_selection({"invalid": "type"}, 100)  # type: ignore
        assert result == [0]


class TestSaveSlices:
    """Tests for save_slices."""

    @pytest.fixture
    def synthetic_image(self) -> Image:
        """Create a synthetic image for testing."""
        arr = np.random.rand(64, 64, 20).astype(np.float32) * 1000
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    @pytest.fixture
    def synthetic_mask(self) -> Image:
        """Create a synthetic mask for testing."""
        arr = np.zeros((64, 64, 20), dtype=np.uint8)
        arr[20:40, 20:40, 5:15] = 1
        arr[10:30, 10:30, 8:12] = 2
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    def test_save_overlay_mode(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving with both image and mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0, 10],
            )
            assert len(files) == 2
            for f in files:
                assert Path(f).exists()

    def test_save_image_only(self, synthetic_image: Image) -> None:
        """Test saving image without mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                slice_selection=[0, 10],
            )
            assert len(files) == 2
            for f in files:
                assert Path(f).exists()

    def test_save_mask_only_colormap(self, synthetic_mask: Image) -> None:
        """Test saving mask only with colormap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                mask=synthetic_mask,
                slice_selection=[0, 10],
                mask_as_colormap=True,
            )
            assert len(files) == 2

    def test_save_mask_only_grayscale(self, synthetic_mask: Image) -> None:
        """Test saving mask only as grayscale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                mask=synthetic_mask,
                slice_selection=[0, 10],
                mask_as_colormap=False,
            )
            assert len(files) == 2

    def test_save_jpeg_format(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving as JPEG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0],
                format="jpeg",
            )
            assert len(files) == 1
            assert files[0].endswith(".jpg")

    def test_save_with_window_level(self, synthetic_image: Image) -> None:
        """Test saving with window/level normalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                slice_selection=[0],
                window_center=500,
                window_width=1000,
            )
            assert len(files) == 1

    def test_save_shape_mismatch_raises(self, synthetic_image: Image) -> None:
        """Test shape mismatch raises ValueError."""
        bad_mask = Image(
            array=np.zeros((32, 32, 20)),
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="shape"):
                save_slices(
                    output_dir=tmpdir,
                    image=synthetic_image,
                    mask=bad_mask,
                )

    def test_save_neither_provided_raises(self) -> None:
        """Test that providing neither raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="At least one"):
                save_slices(output_dir=tmpdir)

    def test_save_with_dpi_72(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving with 72 DPI (scale_factor=1.0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0],
                dpi=72,
            )
            assert len(files) == 1


class TestVisualizeSlices:
    """Tests for visualize_slices (mocked matplotlib)."""

    @pytest.fixture
    def synthetic_image(self) -> Image:
        """Create a synthetic image for testing."""
        arr = np.random.rand(64, 64, 20).astype(np.float32) * 1000
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    @pytest.fixture
    def synthetic_mask(self) -> Image:
        """Create a synthetic mask for testing."""
        arr = np.zeros((64, 64, 20), dtype=np.uint8)
        arr[20:40, 20:40, 5:15] = 1
        return Image(
            array=arr,
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )

    def test_visualize_overlay_mode(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test interactive viewer with overlay."""
        with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
            "matplotlib.pyplot.axes"
        ) as mock_axes, patch("matplotlib.widgets.Slider") as mock_slider_class, patch(
            "matplotlib.pyplot.show"
        ) as mock_show, patch(
            "matplotlib.pyplot.subplots_adjust"
        ):

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_slider = MagicMock()
            mock_slider.val = 10
            mock_slider_class.return_value = mock_slider

            visualize_slices(image=synthetic_image, mask=synthetic_mask)

            mock_show.assert_called_once()

    def test_visualize_image_only(self, synthetic_image: Image) -> None:
        """Test interactive viewer with image only."""
        with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
            "matplotlib.pyplot.axes"
        ), patch("matplotlib.widgets.Slider"), patch(
            "matplotlib.pyplot.show"
        ) as mock_show, patch(
            "matplotlib.pyplot.subplots_adjust"
        ):

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            visualize_slices(image=synthetic_image)

            mock_show.assert_called_once()

    def test_visualize_mask_only(self, synthetic_mask: Image) -> None:
        """Test interactive viewer with mask only."""
        with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
            "matplotlib.pyplot.axes"
        ), patch("matplotlib.widgets.Slider"), patch(
            "matplotlib.pyplot.show"
        ) as mock_show, patch(
            "matplotlib.pyplot.subplots_adjust"
        ):

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            visualize_slices(mask=synthetic_mask)

            mock_show.assert_called_once()

    def test_visualize_neither_raises(self) -> None:
        """Test that providing neither raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            visualize_slices()

    def test_visualize_shape_mismatch_raises(self, synthetic_image: Image) -> None:
        """Test shape mismatch raises ValueError."""
        bad_mask = Image(
            array=np.zeros((32, 32, 20)),
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError, match="shape"):
            visualize_slices(image=synthetic_image, mask=bad_mask)

    def test_visualize_with_different_axes(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test visualize with axis=0 and axis=1."""
        with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
            "matplotlib.pyplot.axes"
        ), patch("matplotlib.widgets.Slider") as mock_slider_class, patch(
            "matplotlib.pyplot.show"
        ), patch(
            "matplotlib.pyplot.subplots_adjust"
        ):
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_slider = MagicMock()
            mock_slider.val = 10
            mock_slider_class.return_value = mock_slider

            # Test axis=0 (sagittal)
            visualize_slices(image=synthetic_image, mask=synthetic_mask, axis=0)
            # Test axis=1 (coronal)
            visualize_slices(image=synthetic_image, mask=synthetic_mask, axis=1)

    def test_visualize_callbacks(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test that slider update and scroll callbacks work."""
        with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
            "matplotlib.pyplot.axes"
        ), patch("matplotlib.widgets.Slider") as mock_slider_class, patch(
            "matplotlib.pyplot.show"
        ), patch(
            "matplotlib.pyplot.subplots_adjust"
        ):
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_slider = MagicMock()
            mock_slider.val = 10
            mock_slider_class.return_value = mock_slider

            # Capture callbacks
            update_callback = None
            scroll_callback = None

            def capture_on_changed(cb):
                nonlocal update_callback
                update_callback = cb

            mock_slider.on_changed = capture_on_changed

            def capture_mpl_connect(event_name, cb):
                nonlocal scroll_callback
                if event_name == "scroll_event":
                    scroll_callback = cb

            mock_fig.canvas.mpl_connect = capture_mpl_connect

            visualize_slices(image=synthetic_image, mask=synthetic_mask)

            # Test update callback
            assert update_callback is not None
            update_callback(5)  # Call with a slice index

            # Test scroll callback
            assert scroll_callback is not None
            mock_event_up = MagicMock()
            mock_event_up.button = "up"
            scroll_callback(mock_event_up)

            mock_event_down = MagicMock()
            mock_event_down.button = "down"
            scroll_callback(mock_event_down)


class TestSliceSelectionAdditional:
    """Additional tests for _parse_slice_selection edge cases."""

    def test_invalid_percentage_value(self) -> None:
        """Test invalid percentage string (non-numeric)."""
        result = _parse_slice_selection("abc%", 100)
        assert result == [0]

    def test_invalid_every_n_value(self) -> None:
        """Test invalid every_N string (non-numeric)."""
        result = _parse_slice_selection("every_abc", 100)
        assert result == [0]


class TestGetReferenceArray:
    """Tests for _get_reference_array helper."""

    def test_get_reference_array_neither_raises(self) -> None:
        """Test that providing neither image nor mask raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            _get_reference_array(None, None)


class TestSaveSlicesAdditional:
    """Additional tests for save_slices edge cases."""

    @pytest.fixture
    def synthetic_image(self) -> Image:
        arr = np.random.rand(64, 64, 20).astype(np.float32) * 1000
        return Image(array=arr, spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))

    @pytest.fixture
    def synthetic_mask(self) -> Image:
        arr = np.zeros((64, 64, 20), dtype=np.uint8)
        arr[20:40, 20:40, 5:15] = 1
        return Image(array=arr, spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))

    def test_save_jpg_format_alias(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving with 'jpg' format alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0],
                format="jpg",  # Should be treated as jpeg
            )
            assert len(files) == 1
            assert files[0].endswith(".jpg")

    def test_save_invalid_format_fallback(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving with invalid format falls back to png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0],
                format="invalid_format",
            )
            assert len(files) == 1
            assert files[0].endswith(".png")

    def test_save_tiff_format(
        self, synthetic_image: Image, synthetic_mask: Image
    ) -> None:
        """Test saving as TIFF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0],
                format="tiff",
            )
            assert len(files) == 1
            assert files[0].endswith(".tiff")

    def test_save_axis_0(self, synthetic_image: Image, synthetic_mask: Image) -> None:
        """Test saving with axis=0 (sagittal)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0, 10],
                axis=0,
            )
            assert len(files) == 2

    def test_save_axis_1(self, synthetic_image: Image, synthetic_mask: Image) -> None:
        """Test saving with axis=1 (coronal)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_slices(
                output_dir=tmpdir,
                image=synthetic_image,
                mask=synthetic_mask,
                slice_selection=[0, 10],
                axis=1,
            )
            assert len(files) == 2
