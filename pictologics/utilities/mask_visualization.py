"""
Mask Visualization Module
=========================

This module provides utilities for visualizing mask overlays on medical images.
It supports interactive slice scrolling and batch export of overlay images.

Key Features:
- Interactive slice viewer with matplotlib
- Multi-label mask support (up to 20+ labels with distinct colors)
- Configurable output formats (PNG, JPEG, TIFF)
- Flexible slice selection for batch export
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image as PILImage

from pictologics.loader import Image

# Colormap definitions for mask labels (RGB tuples, 0-255)
# Based on matplotlib's tab20 colormap
COLORMAPS: dict[str, list[tuple[int, int, int]]] = {
    "tab10": [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ],
    "tab20": [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229),
    ],
    "Set1": [
        (228, 26, 28),
        (55, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (255, 255, 51),
        (166, 86, 40),
        (247, 129, 191),
        (153, 153, 153),
    ],
    "Set2": [
        (102, 194, 165),
        (252, 141, 98),
        (141, 160, 203),
        (231, 138, 195),
        (166, 216, 84),
        (255, 217, 47),
        (229, 196, 148),
        (179, 179, 179),
    ],
    "Paired": [
        (166, 206, 227),
        (31, 120, 180),
        (178, 223, 138),
        (51, 160, 44),
        (251, 154, 153),
        (227, 26, 28),
        (253, 191, 111),
        (255, 127, 0),
        (202, 178, 214),
        (106, 61, 154),
        (255, 255, 153),
        (177, 89, 40),
    ],
}


def _normalize_image(image_array: np.ndarray) -> np.ndarray:
    """Normalize image array to 0-255 uint8."""
    arr = image_array.astype(np.float64)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def _get_colormap_colors(colormap: str) -> list[tuple[int, int, int]]:
    """Get color list for the specified colormap."""
    if colormap in COLORMAPS:
        return COLORMAPS[colormap]
    # Default to tab20
    return COLORMAPS["tab20"]


def _create_overlay_rgba(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "tab20",
) -> np.ndarray:
    """
    Create an RGBA overlay of mask on image slice.

    This function handles the axis convention difference between medical imaging
    libraries and matplotlib:

    - **Pictologics convention**: Arrays are stored in (X, Y, Z) order to match
      ITK/SimpleITK conventions, where X=columns, Y=rows, Z=slices.
    - **Matplotlib imshow convention**: Expects (height, width) = (rows, cols) = (Y, X)

    This function internally transposes 2D slices from (X, Y) to (Y, X) before
    rendering, ensuring correct visual orientation without modifying the underlying
    data storage format.

    Args:
        image_slice: 2D grayscale image array in (X, Y) format.
        mask_slice: 2D mask array with integer labels (0=background) in (X, Y) format.
        alpha: Transparency of mask overlay (0-1).
        colormap: Name of colormap for mask labels.

    Returns:
        RGBA array (H, W, 4) as uint8, ready for matplotlib imshow.
    """
    # Transpose from (X, Y) to (Y, X) for proper display with imshow
    # Pictologics stores arrays in (X, Y, Z) format to match ITK/SimpleITK conventions
    # Matplotlib imshow expects (height, width) = (rows, columns) = (Y, X)
    image_slice = np.transpose(image_slice)
    mask_slice = np.transpose(mask_slice)

    # Normalize image to grayscale
    gray = _normalize_image(image_slice)

    # Create RGB base from grayscale
    rgb = np.stack([gray, gray, gray], axis=-1)

    # Create alpha channel (fully opaque)
    rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255

    # Get colormap colors
    colors = _get_colormap_colors(colormap)
    num_colors = len(colors)

    # Apply mask colors
    unique_labels = np.unique(mask_slice)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        # Get color (cycle if more labels than colors)
        color_idx = (int(label) - 1) % num_colors
        color = colors[color_idx]

        # Create mask for this label
        label_mask = mask_slice == label

        # Blend colors
        for i in range(3):
            rgba[..., i][label_mask] = np.clip(
                (1 - alpha) * rgba[..., i][label_mask] + alpha * color[i],
                0,
                255,
            ).astype(np.uint8)

    return rgba


def _parse_slice_selection(
    selection: Union[str, int, list[int]],
    num_slices: int,
) -> list[int]:
    """
    Parse slice selection specification.

    Args:
        selection: One of:
            - "every_N" or "N": Every Nth slice
            - "N%": Slices at each N% interval
            - int: Single slice index
            - list[int]: Specific slice indices
        num_slices: Total number of slices.

    Returns:
        List of slice indices.
    """
    if isinstance(selection, int):
        return [selection]

    if isinstance(selection, list):
        return [i for i in selection if 0 <= i < num_slices]

    if isinstance(selection, str):
        selection = selection.strip()

        # Percentage-based: "10%" means every 10%
        if selection.endswith("%"):
            try:
                pct = float(selection[:-1])
                if pct <= 0:
                    return [0]
                step = max(1, int(num_slices * pct / 100))
                return list(range(0, num_slices, step))
            except ValueError:
                return [0]

        # Every N: "every_10" or just "10"
        try:
            if selection.startswith("every_"):
                n = int(selection[6:])
            else:
                n = int(selection)
            if n <= 0:
                return [0]
            return list(range(0, num_slices, n))
        except ValueError:
            return [0]

    return [0]


def save_mask_overlay_slices(
    image: Image,
    mask: Image,
    output_dir: str,
    slice_selection: Union[str, int, list[int]] = "10%",
    format: str = "png",
    dpi: int = 300,
    alpha: float = 0.25,
    colormap: str = "tab20",
    axis: int = 2,
    filename_prefix: str = "slice",
) -> list[str]:
    """
    Save mask overlay images for selected slices.

    Args:
        image: Pictologics Image object containing the image data.
        mask: Pictologics Image object containing the mask data.
        output_dir: Directory to save output images.
        slice_selection: Slice selection specification:
            - "every_N" or "N": Every Nth slice
            - "N%": Slices at each N% interval (e.g., "10%" = 10 images)
            - int: Single slice index
            - list[int]: Specific slice indices
        format: Output format ("png", "jpeg", "tiff").
        dpi: Output resolution in dots per inch.
        alpha: Transparency of mask overlay (0-1).
        colormap: Colormap for mask labels. Options:
            - "tab10": 10 distinct colors
            - "tab20": 20 distinct colors (default)
            - "Set1": 9 bold colors
            - "Set2": 8 pastel colors
            - "Paired": 12 paired colors
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        filename_prefix: Prefix for output filenames.

    Returns:
        List of paths to saved files.

    Raises:
        ValueError: If image and mask shapes don't match.

    Example:
        ```python
        from pictologics import load_image
        from pictologics.utilities import save_mask_overlay_slices
        img = load_image("scan.nii.gz")
        mask = load_image("segmentation.nii.gz")
        files = save_mask_overlay_slices(img, mask, "output/", "10%")
        ```
    """
    # Validate shapes
    if image.array.shape != mask.array.shape:
        raise ValueError(
            f"Image shape {image.array.shape} does not match "
            f"mask shape {mask.array.shape}"
        )

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get number of slices along axis
    num_slices = image.array.shape[axis]

    # Parse slice selection
    slice_indices = _parse_slice_selection(slice_selection, num_slices)

    # Validate format
    format = format.lower()
    if format == "jpg":
        format = "jpeg"
    if format not in ("png", "jpeg", "tiff"):
        format = "png"

    # Calculate pixel size based on DPI
    # Assume 1 pixel = 1 unit at 72 DPI, scale accordingly
    scale_factor = dpi / 72.0

    saved_files = []

    for idx in slice_indices:
        # Extract slices
        if axis == 0:
            img_slice = image.array[idx, :, :]
            mask_slice = mask.array[idx, :, :]
        elif axis == 1:
            img_slice = image.array[:, idx, :]
            mask_slice = mask.array[:, idx, :]
        else:  # axis == 2
            img_slice = image.array[:, :, idx]
            mask_slice = mask.array[:, :, idx]

        # Create overlay
        rgba = _create_overlay_rgba(img_slice, mask_slice, alpha, colormap)

        # Scale if needed for DPI
        if scale_factor != 1.0:
            h, w = rgba.shape[:2]
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            pil_img = PILImage.fromarray(rgba)
            pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
        else:
            pil_img = PILImage.fromarray(rgba)

        # Convert to RGB for JPEG (no alpha support)
        if format == "jpeg":
            pil_img = pil_img.convert("RGB")

        # Save
        ext = {"png": ".png", "jpeg": ".jpg", "tiff": ".tiff"}[format]
        filename = f"{filename_prefix}_{idx:04d}{ext}"
        filepath = out_path / filename
        pil_img.save(filepath, dpi=(dpi, dpi))
        saved_files.append(str(filepath))

    return saved_files


def visualize_mask_overlay(
    image: Image,
    mask: Image,
    alpha: float = 0.25,
    colormap: str = "tab20",
    axis: int = 2,
    initial_slice: Optional[int] = None,
    window_title: str = "Mask Overlay Viewer",
) -> None:
    """
    Display interactive mask overlay viewer with slice scrolling.

    Args:
        image: Pictologics Image object containing the image data.
        mask: Pictologics Image object containing the mask data.
        alpha: Transparency of mask overlay (0-1).
        colormap: Colormap for mask labels. Options:
            - "tab10": 10 distinct colors
            - "tab20": 20 distinct colors (default)
            - "Set1": 9 bold colors
            - "Set2": 8 pastel colors
            - "Paired": 12 paired colors
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        initial_slice: Initial slice to display (default: middle).
        window_title: Title for the viewer window.

    Raises:
        ValueError: If image and mask shapes don't match.

    Example:
        ```python
        from pictologics import load_image
        from pictologics.utilities import visualize_mask_overlay
        img = load_image("scan.nii.gz")
        mask = load_image("segmentation.nii.gz")
        visualize_mask_overlay(img, mask)
        ```
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # Validate shapes
    if image.array.shape != mask.array.shape:
        raise ValueError(
            f"Image shape {image.array.shape} does not match "
            f"mask shape {mask.array.shape}"
        )

    # Get number of slices
    num_slices = image.array.shape[axis]

    # Set initial slice
    if initial_slice is None:
        initial_slice = num_slices // 2

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15)

    # Get initial slice data
    def get_slice(idx: int) -> tuple[np.ndarray, np.ndarray]:
        if axis == 0:
            return image.array[idx, :, :], mask.array[idx, :, :]
        elif axis == 1:
            return image.array[:, idx, :], mask.array[:, idx, :]
        else:
            return image.array[:, :, idx], mask.array[:, :, idx]

    img_slice, mask_slice = get_slice(initial_slice)
    rgba = _create_overlay_rgba(img_slice, mask_slice, alpha, colormap)

    # Display
    im = ax.imshow(rgba, aspect="equal")
    ax.set_title(f"Slice {initial_slice}/{num_slices - 1}")
    ax.axis("off")

    # Add slider
    ax_slider = plt.axes((0.15, 0.05, 0.7, 0.03))
    slider = Slider(
        ax=ax_slider,
        label="Slice",
        valmin=0,
        valmax=num_slices - 1,
        valinit=initial_slice,
        valstep=1,
    )

    def update(val: float) -> None:
        idx = int(val)
        img_slice, mask_slice = get_slice(idx)
        rgba = _create_overlay_rgba(img_slice, mask_slice, alpha, colormap)
        im.set_data(rgba)
        ax.set_title(f"Slice {idx}/{num_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Add scroll wheel support
    def on_scroll(event) -> None:  # type: ignore[no-untyped-def]
        if event.button == "up":
            new_val = min(slider.val + 1, num_slices - 1)
        else:
            new_val = max(slider.val - 1, 0)
        slider.set_val(new_val)

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    fig.suptitle(window_title)
    plt.show()
