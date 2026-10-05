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

- **window_center** (Level): The center value of the display window (for example 40 HU for soft tissue)
- **window_width** (Width): The range of values displayed (for example 400 HU)

Without them, all slices share one gray scale: the minimum and maximum of the volume.

Values outside [center - width/2, center + width/2] are clipped to black/white.

Common presets:
- Soft tissue: Center=40, Width=400
- Bone: Center=400, Width=1800
- Lung: Center=-600, Width=1500
- Brain: Center=40, Width=80
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numba import get_num_threads
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
    np.nan_to_num(arr, copy=False, nan=min_val)  # NaN shows as the window minimum
    arr = np.clip(arr, min_val, max_val)
    arr = (arr - min_val) / (max_val - min_val) * 255
    return arr.astype(np.uint8)


def _check_window_width(window_width: Optional[float]) -> None:
    """A window width of 0 divides by zero, and a negative width makes every pixel white."""
    if window_width is not None and window_width <= 0:
        raise ValueError(f"window_width must be more than 0, not {window_width}")


def _normalize_image(
    image_array: npt.NDArray[np.floating[Any]],
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    value_range: Optional[tuple[float, float]] = None,
) -> npt.NDArray[np.floating[Any]]:
    """
    Normalize image array to 0-255 uint8.

    If window/level parameters are provided, uses window/level normalization.
    Otherwise, uses min-max normalization over `value_range`, or over the array
    itself when no range is given.

    Args:
        image_array: Input image array.
        window_center: Optional window center (level).
        window_width: Optional window width.
        value_range: Optional (minimum, maximum) for min-max normalization, for
            example those of the whole volume, so that all slices share one scale.

    Returns:
        Normalized array as uint8.
    """
    if window_center is not None and window_width is not None:
        return _apply_window_level(image_array, window_center, window_width)

    # Default: min-max normalization
    arr = image_array.astype(np.float64)
    if value_range is not None:
        arr_min, arr_max = value_range
        np.nan_to_num(arr, copy=False, nan=arr_min)  # NaN shows as the minimum
    else:
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
    value_range: Optional[tuple[float, float]] = None,
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
        value_range: Optional (minimum, maximum) for the min-max normalization of the
            slice shown in gray (the image, or a grayscale mask), used without a window.

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
        gray = _normalize_image(image_slice, window_center, window_width, value_range)
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

            # Background stays black; each label takes its color in one lookup
            _, color_idx = _label_colors(mask_slice, num_colors)
            rgba[..., :3] = _color_table(colors, np.uint8)[color_idx]
            return rgba  # type: ignore[return-value]
        else:
            # Grayscale mask
            gray = _normalize_image(mask_slice, window_center, window_width, value_range)
            rgba = np.zeros((*shape, 4), dtype=np.uint8)
            rgba[..., 0] = gray
            rgba[..., 1] = gray
            rgba[..., 2] = gray
            rgba[..., 3] = 255
            return rgba  # type: ignore[return-value]

    # --- Mode 3: Overlay (image + mask) ---
    gray = _normalize_image(image_slice, window_center, window_width, value_range)

    # Create RGB base from grayscale
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = 255

    # Get colormap colors
    colors = _get_colormap_colors(colormap)
    num_colors = len(colors)

    # Apply mask colors with blending, for all labelled pixels at once
    labelled, color_idx = _label_colors(mask_slice, num_colors)
    pixels = np.flatnonzero(labelled)
    rgb = rgba.reshape(-1, 4)  # a view: one row per pixel
    rgb[pixels, :3] = np.clip(
        (1 - alpha) * rgb[pixels, :3]
        + alpha * _color_table(colors, np.float64)[color_idx.ravel()[pixels]],
        0,
        255,
    ).astype(np.uint8)

    return rgba  # type: ignore[return-value]


def _label_colors(
    mask_slice: npt.NDArray[Any], num_colors: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """The labelled pixels, and per pixel the color index: (label - 1) mod the colors,
    with the label truncated to an integer, or `num_colors` (black) for background 0. A
    NaN in a float mask is background."""
    labelled = mask_slice != 0
    if mask_slice.dtype.kind == "f":
        labelled &= ~np.isnan(mask_slice)
        mask_slice = np.nan_to_num(mask_slice, nan=0.0)
    index = np.where(labelled, (mask_slice.astype(np.int64) - 1) % num_colors, num_colors)
    return labelled, index


def _color_table(colors: list[tuple[int, int, int]], dtype: Any) -> npt.NDArray[Any]:
    """The colors as rows of a table, with a last black row for the background."""
    table = np.zeros((len(colors) + 1, 3), dtype=dtype)
    table[: len(colors)] = colors
    return table


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

    Raises:
        ValueError: If a single index is outside 0 to num_slices - 1.
    """
    if isinstance(selection, (int, np.integer)):
        if not 0 <= selection < num_slices:
            raise ValueError(
                f"Slice {selection} is out of range: the axis has {num_slices} slices "
                f"(0 to {num_slices - 1})."
            )
        return [int(selection)]

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


def _gray_range(
    image: Optional[Image],
    mask: Optional[Image],
    window_center: Optional[float],
    window_width: Optional[float],
    mask_as_colormap: bool,
) -> Optional[tuple[float, float]]:
    """(minimum, maximum) of the whole volume shown in gray (the image, or a mask shown
    in grayscale) when no window is set, so that all slices share one gray scale. NaN
    values are left out."""
    if window_center is not None and window_width is not None:
        return None
    gray = image if image is not None else (None if mask_as_colormap else mask)
    if gray is None:
        return None
    return float(np.nanmin(gray.array)), float(np.nanmax(gray.array))


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
    output_dir: str | Path,
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
            - int: Single slice index (0 to the number of slices - 1)
            - list[int]: Specific slice indices (indices out of range are skipped)
        format: Output format ("png", "jpeg", "tiff").
        dpi: The resolution tag written into the file, in dots per inch. The image has
            one pixel per voxel; viewers and printers use the tag to scale it.
        alpha: Transparency of mask overlay (0-1). Only used in overlay mode.
        colormap: Colormap for mask labels. Options:
            - "tab10": 10 distinct colors
            - "tab20": 20 distinct colors (default)
            - "Set1": 9 bold colors
            - "Set2": 8 pastel colors
            - "Paired": 12 paired colors
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        filename_prefix: Prefix for output filenames.
        window_center: Window center (level) for normalization. Default: None (the
            minimum and maximum of the whole volume, the same for all slices).
        window_width: Window width for normalization. Default: None (the minimum and
            maximum of the whole volume, the same for all slices).
        mask_as_colormap: If True and mask-only mode, display with colormap.
            If False, display as grayscale.

    Returns:
        List of paths to saved files.

    Raises:
        ValueError: If neither image nor mask is provided, if shapes don't match
            when both are provided, if a single slice index is out of range, or if
            window_width is 0 or less.

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
    _check_window_width(window_width)

    # Validate shapes if both provided
    if image is not None and mask is not None:
        if image.array.shape != mask.array.shape:
            raise ValueError(
                f"Image shape {image.array.shape} does not match mask shape {mask.array.shape}"
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
    value_range = _gray_range(image, mask, window_center, window_width, mask_as_colormap)

    # Validate format
    format = format.lower()
    if format == "jpg":
        format = "jpeg"
    if format not in ("png", "jpeg", "tiff"):
        format = "png"

    ext = {"png": ".png", "jpeg": ".jpg", "tiff": ".tiff"}[format]
    # PNG compression level 3: 2.1x faster than the default 6, for files 4 % larger
    # (40 slices of a CT with a mask).
    save_options: dict[str, Any] = {"compress_level": 3} if format == "png" else {}

    def save_slice(idx: int) -> str:
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
            value_range,
        )

        # The overlay is mixed into the colors and the alpha channel is always 255, so
        # the files are RGB (the same pixels as RGBA, a quarter less data to encode).
        pil_img = PILImage.fromarray(np.ascontiguousarray(rgba[..., :3]))

        # Save: one pixel per voxel, with dpi as the resolution tag of the file
        filename = f"{filename_prefix}_{idx:04d}{ext}"
        filepath = out_path / filename
        pil_img.save(filepath, dpi=(dpi, dpi), **save_options)
        return str(filepath)

    # Slices in threads: PIL releases the GIL while it encodes. Numba's thread count, and
    # not more threads than slices.
    workers = min(get_num_threads(), max(1, len(slice_indices)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(save_slice, slice_indices))


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

    The viewer needs Matplotlib, the optional extra "viz": ``pip install "pictologics[viz]"``.

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
        window_center: Window center (level) for normalization. Default: None (the
            minimum and maximum of the whole volume, the same for all slices).
        window_width: Window width for normalization. Default: None (the minimum and
            maximum of the whole volume, the same for all slices).
        mask_as_colormap: If True and mask-only mode, display with colormap.
            If False, display as grayscale.

    Raises:
        ImportError: If Matplotlib is not installed.
        ValueError: If neither image nor mask is provided, if shapes don't match
            when both are provided, if initial_slice is out of range, or if
            window_width is 0 or less.

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
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError as error:  # Matplotlib is the optional extra "viz"
        raise ImportError(
            'visualize_slices needs Matplotlib: pip install "pictologics[viz]"'
        ) from error

    if image is None and mask is None:
        raise ValueError("At least one of image or mask must be provided.")
    _check_window_width(window_width)

    # Validate shapes if both provided
    if image is not None and mask is not None:
        if image.array.shape != mask.array.shape:
            raise ValueError(
                f"Image shape {image.array.shape} does not match mask shape {mask.array.shape}"
            )

    # Get reference array for shape
    ref_array = _get_reference_array(image, mask)

    # Get number of slices
    num_slices = ref_array.shape[axis]

    # Set initial slice
    if initial_slice is None:
        initial_slice = num_slices // 2
    elif not 0 <= initial_slice < num_slices:
        raise ValueError(
            f"initial_slice {initial_slice} is out of range: the axis has {num_slices} "
            f"slices (0 to {num_slices - 1})."
        )
    value_range = _gray_range(image, mask, window_center, window_width, mask_as_colormap)

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15)

    # Get slice data
    def get_slice(
        idx: int,
    ) -> tuple[Optional[npt.NDArray[np.floating[Any]]], Optional[npt.NDArray[np.floating[Any]]]]:
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
        value_range,
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
            value_range,
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
