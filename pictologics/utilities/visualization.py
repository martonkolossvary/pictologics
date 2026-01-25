"""
Visualization Module
====================

This module provides utilities for visualizing medical images and segmentation masks.
It supports interactive slice scrolling and batch export of images.

Key Features
------------
- **Interactive slice viewer** with matplotlib
- **Flexible display modes**: image-only, mask-only, or overlay
- **Multi-label mask support** (up to 20+ labels with distinct colors)
- **Window/Level normalization** for CT/MR viewing
- **Configurable output formats** (PNG, JPEG, TIFF)
- **Flexible slice selection** for batch export

Display Modes
-------------
The visualization functions support three display modes based on which inputs are provided:

1. **Image + Mask (Overlay Mode)**:
   Both `image` and `mask` are provided. The mask is overlaid on the grayscale image
   with the specified transparency (alpha) and colormap.

2. **Image Only**:
   Only `image` is provided (`mask=None`). The image is displayed as grayscale,
   optionally with window/level normalization applied.

3. **Mask Only**:
   Only `mask` is provided (`image=None`). The mask can be displayed either:
   - As a **colormap visualization** (`mask_as_colormap=True`, default): Each unique
     label value gets a distinct color from the specified colormap.
   - As **grayscale** (`mask_as_colormap=False`): Values are normalized to 0-255.

Window/Level Normalization
--------------------------
For medical imaging (CT, MR), window/level controls are essential for proper visualization.
When `window_center` and `window_width` are specified:

- **window_center** (Level): The center value of the display window (default: 200 HU for soft tissue)
- **window_width** (Width): The range of values displayed (default: 600 HU)

Values outside [center - width/2, center + width/2] are clipped to black/white.

Common presets:
- Soft tissue: Center=40, Width=400
- Bone: Center=400, Width=1800
- Lung: Center=-600, Width=1500
- Brain: Center=40, Width=80
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy import typing as npt
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

# Default window/level values (suitable for soft tissue CT)
DEFAULT_WINDOW_CENTER = 200.0
DEFAULT_WINDOW_WIDTH = 600.0


def _apply_window_level(
    arr: npt.NDArray[np.floating[Any]],
    center: float,
    width: float,
) -> npt.NDArray[np.floating[Any]]:
    """
    Apply window/level normalization to an image array.

    This is the standard method for adjusting contrast in medical imaging,
    particularly for CT and MR images.

    Args:
        arr: Input image array (any numeric dtype).
        center: Window center (level) value.
        width: Window width value.

    Returns:
        Normalized array as uint8 (0-255).
    """
    arr = arr.astype(np.float64)
    min_val = center - width / 2
    max_val = center + width / 2
    arr = np.clip(arr, min_val, max_val)
    arr = (arr - min_val) / (max_val - min_val) * 255
    return arr.astype(np.uint8)


def _normalize_image(
    image_array: npt.NDArray[np.floating[Any]],
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Normalize image array to 0-255 uint8.

    If window/level parameters are provided, uses window/level normalization.
    Otherwise, uses min-max normalization.

    Args:
        image_array: Input image array.
        window_center: Optional window center (level).
        window_width: Optional window width.

    Returns:
        Normalized array as uint8.
    """
    if window_center is not None and window_width is not None:
        return _apply_window_level(image_array, window_center, window_width)

    # Default: min-max normalization
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


def _create_display_rgba(
    image_slice: Optional[npt.NDArray[np.floating[Any]]],
    mask_slice: Optional[npt.NDArray[np.floating[Any]]],
    alpha: float = 0.25,
    colormap: str = "tab20",
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    mask_as_colormap: bool = True,
) -> npt.NDArray[np.floating[Any]]:
    """
    Create an RGBA image for display.

    Supports three modes:
    1. Image + Mask: Overlay mask on grayscale image
    2. Image only: Grayscale image
    3. Mask only: Colormap or grayscale mask

    Args:
        image_slice: 2D grayscale image array in (X, Y) format, or None.
        mask_slice: 2D mask array with integer labels in (X, Y) format, or None.
        alpha: Transparency of mask overlay (0-1).
        colormap: Name of colormap for mask labels.
        window_center: Optional window center for image normalization.
        window_width: Optional window width for image normalization.
        mask_as_colormap: If True and mask-only, display with colormap. If False, grayscale.

    Returns:
        RGBA array (H, W, 4) as uint8, ready for matplotlib imshow.

    Raises:
        ValueError: If both image_slice and mask_slice are None.
    """
    if image_slice is None and mask_slice is None:
        raise ValueError("At least one of image_slice or mask_slice must be provided.")

    # Transpose from (X, Y) to (Y, X) for proper display with imshow
    if image_slice is not None:
        image_slice = np.transpose(image_slice)
    if mask_slice is not None:
        mask_slice = np.transpose(mask_slice)

    # Determine shape from whichever slice is provided
    if image_slice is not None:
        shape = image_slice.shape
    else:
        assert mask_slice is not None  # For mypy: we know at least one is not None
        shape = mask_slice.shape

    # --- Mode 1: Image only ---
    if mask_slice is None:
        assert image_slice is not None  # For mypy
        gray = _normalize_image(image_slice, window_center, window_width)
        rgba = np.zeros((*shape, 4), dtype=np.uint8)
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = 255
        return rgba  # type: ignore[return-value]

    # --- Mode 2: Mask only ---
    if image_slice is None:
        if mask_as_colormap:
            # Create colormap visualization
            colors = _get_colormap_colors(colormap)
            num_colors = len(colors)
            rgba = np.zeros((*shape, 4), dtype=np.uint8)
            rgba[..., 3] = 255  # Fully opaque

            # Background stays black
            unique_labels = np.unique(mask_slice)
            for label in unique_labels:
                if label == 0:
                    continue
                color_idx = (int(label) - 1) % num_colors
                color = colors[color_idx]
                label_mask = mask_slice == label
                rgba[..., 0][label_mask] = color[0]
                rgba[..., 1][label_mask] = color[1]
                rgba[..., 2][label_mask] = color[2]
            return rgba  # type: ignore[return-value]
        else:
            # Grayscale mask
            gray = _normalize_image(mask_slice, window_center, window_width)
            rgba = np.zeros((*shape, 4), dtype=np.uint8)
            rgba[..., 0] = gray
            rgba[..., 1] = gray
            rgba[..., 2] = gray
            rgba[..., 3] = 255
            return rgba  # type: ignore[return-value]

    # --- Mode 3: Overlay (image + mask) ---
    gray = _normalize_image(image_slice, window_center, window_width)

    # Create RGB base from grayscale
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = 255

    # Get colormap colors
    colors = _get_colormap_colors(colormap)
    num_colors = len(colors)

    # Apply mask colors with blending
    unique_labels = np.unique(mask_slice)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        color_idx = (int(label) - 1) % num_colors
        color = colors[color_idx]
        label_mask = mask_slice == label

        for i in range(3):
            rgba[..., i][label_mask] = np.clip(
                (1 - alpha) * rgba[..., i][label_mask] + alpha * color[i],
                0,
                255,
            ).astype(np.uint8)

    return rgba  # type: ignore[return-value]


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


def _get_reference_array(
    image: Optional[Image],
    mask: Optional[Image],
) -> npt.NDArray[np.floating[Any]]:
    """Get the reference array for shape/slicing operations."""
    if image is not None:
        return image.array
    if mask is not None:
        return mask.array
    raise ValueError("At least one of image or mask must be provided.")


def save_slices(
    output_dir: str,
    image: Optional[Image] = None,
    mask: Optional[Image] = None,
    slice_selection: Union[str, int, list[int]] = "10%",
    format: str = "png",
    dpi: int = 300,
    alpha: float = 0.25,
    colormap: str = "tab20",
    axis: int = 2,
    filename_prefix: str = "slice",
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    mask_as_colormap: bool = True,
) -> list[str]:
    """
    Save image slices to files.

    This function supports three display modes:

    1. **Image + Mask (Overlay Mode)**: Both `image` and `mask` are provided.
       The mask is overlaid on the grayscale image with transparency.

    2. **Image Only**: Only `image` is provided. Saves grayscale slices,
       optionally with window/level normalization.

    3. **Mask Only**: Only `mask` is provided. Saves mask visualization
       using either a colormap or grayscale display.

    Args:
        output_dir: Directory to save output images.
        image: Optional Pictologics Image object containing the image data.
        mask: Optional Pictologics Image object containing the mask data.
        slice_selection: Slice selection specification:
            - "every_N" or "N": Every Nth slice
            - "N%": Slices at each N% interval (e.g., "10%" = ~10 images)
            - int: Single slice index
            - list[int]: Specific slice indices
        format: Output format ("png", "jpeg", "tiff").
        dpi: Output resolution in dots per inch.
        alpha: Transparency of mask overlay (0-1). Only used in overlay mode.
        colormap: Colormap for mask labels. Options:
            - "tab10": 10 distinct colors
            - "tab20": 20 distinct colors (default)
            - "Set1": 9 bold colors
            - "Set2": 8 pastel colors
            - "Paired": 12 paired colors
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        filename_prefix: Prefix for output filenames.
        window_center: Window center (level) for normalization. Default: None (min-max).
        window_width: Window width for normalization. Default: None (min-max).
        mask_as_colormap: If True and mask-only mode, display with colormap.
            If False, display as grayscale.

    Returns:
        List of paths to saved files.

    Raises:
        ValueError: If neither image nor mask is provided, or if shapes don't match
            when both are provided.

    Example:
        Save image slices with and without mask overlay:

        ```python
        from pictologics import load_image
        from pictologics.utilities import save_slices

        # Save image with mask overlay
        img = load_image("scan.nii.gz")
        mask = load_image("segmentation.nii.gz")
        files = save_slices("output/", image=img, mask=mask, slice_selection="10%")

        # Save image only (no mask)
        files = save_slices("output/", image=img, slice_selection="10%")

        # Save mask only with colormap
        files = save_slices("output/", mask=mask, slice_selection="10%")
        ```
    """
    if image is None and mask is None:
        raise ValueError("At least one of image or mask must be provided.")

    # Validate shapes if both provided
    if image is not None and mask is not None:
        if image.array.shape != mask.array.shape:
            raise ValueError(
                f"Image shape {image.array.shape} does not match "
                f"mask shape {mask.array.shape}"
            )

    # Get reference array for shape
    ref_array = _get_reference_array(image, mask)

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get number of slices along axis
    num_slices = ref_array.shape[axis]

    # Parse slice selection
    slice_indices = _parse_slice_selection(slice_selection, num_slices)

    # Validate format
    format = format.lower()
    if format == "jpg":
        format = "jpeg"
    if format not in ("png", "jpeg", "tiff"):
        format = "png"

    # Calculate pixel size based on DPI
    scale_factor = dpi / 72.0

    saved_files = []

    for idx in slice_indices:
        # Extract slices
        img_slice = None
        mask_slice = None

        if image is not None:
            if axis == 0:
                img_slice = image.array[idx, :, :]
            elif axis == 1:
                img_slice = image.array[:, idx, :]
            else:
                img_slice = image.array[:, :, idx]

        if mask is not None:
            if axis == 0:
                mask_slice = mask.array[idx, :, :]
            elif axis == 1:
                mask_slice = mask.array[:, idx, :]
            else:
                mask_slice = mask.array[:, :, idx]

        # Create display RGBA
        rgba = _create_display_rgba(
            img_slice,
            mask_slice,
            alpha,
            colormap,
            window_center,
            window_width,
            mask_as_colormap,
        )

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


def visualize_slices(
    image: Optional[Image] = None,
    mask: Optional[Image] = None,
    alpha: float = 0.25,
    colormap: str = "tab20",
    axis: int = 2,
    initial_slice: Optional[int] = None,
    window_title: str = "Slice Viewer",
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    mask_as_colormap: bool = True,
) -> None:
    """
    Display interactive slice viewer with scrolling.

    This function supports three display modes:

    1. **Image + Mask (Overlay Mode)**: Both `image` and `mask` are provided.
       The mask is overlaid on the grayscale image with transparency.

    2. **Image Only**: Only `image` is provided. Displays grayscale slices,
       optionally with window/level normalization.

    3. **Mask Only**: Only `mask` is provided. Displays mask visualization
       using either a colormap or grayscale display.

    Args:
        image: Optional Pictologics Image object containing the image data.
        mask: Optional Pictologics Image object containing the mask data.
        alpha: Transparency of mask overlay (0-1). Only used in overlay mode.
        colormap: Colormap for mask labels. Options:
            - "tab10": 10 distinct colors
            - "tab20": 20 distinct colors (default)
            - "Set1": 9 bold colors
            - "Set2": 8 pastel colors
            - "Paired": 12 paired colors
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        initial_slice: Initial slice to display (default: middle).
        window_title: Title for the viewer window.
        window_center: Window center (level) for normalization. Default: None (min-max).
        window_width: Window width for normalization. Default: None (min-max).
        mask_as_colormap: If True and mask-only mode, display with colormap.
            If False, display as grayscale.

    Raises:
        ValueError: If neither image nor mask is provided, or if shapes don't match
            when both are provided.

    Example:
        Visualise slices interactively:

        ```python
        from pictologics import load_image
        from pictologics.utilities import visualize_slices

        # View image with mask overlay
        img = load_image("scan.nii.gz")
        mask = load_image("segmentation.nii.gz")
        visualize_slices(image=img, mask=mask)

        # View image only
        visualize_slices(image=img, window_center=40, window_width=400)

        # View mask only with colormap
        visualize_slices(mask=mask)
        ```
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if image is None and mask is None:
        raise ValueError("At least one of image or mask must be provided.")

    # Validate shapes if both provided
    if image is not None and mask is not None:
        if image.array.shape != mask.array.shape:
            raise ValueError(
                f"Image shape {image.array.shape} does not match "
                f"mask shape {mask.array.shape}"
            )

    # Get reference array for shape
    ref_array = _get_reference_array(image, mask)

    # Get number of slices
    num_slices = ref_array.shape[axis]

    # Set initial slice
    if initial_slice is None:
        initial_slice = num_slices // 2

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15)

    # Get slice data
    def get_slice(
        idx: int,
    ) -> tuple[
        Optional[npt.NDArray[np.floating[Any]]], Optional[npt.NDArray[np.floating[Any]]]
    ]:
        img_slice = None
        mask_slice = None

        if image is not None:
            if axis == 0:
                img_slice = image.array[idx, :, :]
            elif axis == 1:
                img_slice = image.array[:, idx, :]
            else:
                img_slice = image.array[:, :, idx]

        if mask is not None:
            if axis == 0:
                mask_slice = mask.array[idx, :, :]
            elif axis == 1:
                mask_slice = mask.array[:, idx, :]
            else:
                mask_slice = mask.array[:, :, idx]

        return img_slice, mask_slice

    img_slice, mask_slice = get_slice(initial_slice)
    rgba = _create_display_rgba(
        img_slice,
        mask_slice,
        alpha,
        colormap,
        window_center,
        window_width,
        mask_as_colormap,
    )

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
        rgba = _create_display_rgba(
            img_slice,
            mask_slice,
            alpha,
            colormap,
            window_center,
            window_width,
            mask_as_colormap,
        )
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
