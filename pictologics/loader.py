"""
Image Loading Module
====================

This module handles the loading of medical images from various formats (NIfTI, DICOM)
into a standardized `Image` class. It abstracts away file format differences to provide
a consistent interface for the rest of the library.

Key Features:
-------------
- **Unified Image Class**: Stores 3D data, spacing, origin, direction, and modality.
- **Format Support**:
    - NIfTI (.nii, .nii.gz) via `nibabel`.
    - DICOM Series (directory of DICOM files) via `pydicom`.
    - Single DICOM files.
- **Automatic Detection**: `load_image` automatically detects format and dimensionality.
- **Robust DICOM Sorting**: Sorts slices based on spatial position and orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pydicom
from numpy.typing import DTypeLike


@dataclass
class Image:
    """
    A standardized container for 3D medical image data and metadata.

    This class serves as the common interface for all image processing operations
    in the library, abstracting away the differences between file formats like
    DICOM and NIfTI.

    Attributes:
        array (np.ndarray): The 3D image data with shape (x, y, z).
        spacing (tuple[float, float, float]): Voxel spacing in millimeters (mm)
            along the (x, y, z) axes.
        origin (tuple[float, float, float]): World coordinates of the image origin
            (center of the first voxel) in millimeters (mm).
        direction (Optional[np.ndarray]): 3x3 direction cosine matrix defining the
            orientation of the image axes in world space. Defaults to identity matrix.
        modality (str): The imaging modality (e.g., 'CT', 'MR', 'PT'). Defaults to 'Unknown'.
    """

    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: Optional[np.ndarray] = None
    modality: str = "Unknown"


def create_full_mask(reference_image: Image, dtype: DTypeLike = np.uint8) -> Image:
    """Create a whole-image ROI mask matching a reference image.

    This utility is primarily used when a user does not provide a segmentation mask.
    The returned mask has the same geometry (shape, spacing, origin, direction) as
    the reference image and contains a value of 1 for every voxel.

    Args:
        reference_image: Image whose geometry should be copied.
        dtype: Numpy dtype to use for the mask array. Defaults to `np.uint8`.

    Returns:
        An `Image` mask with `array == 1` everywhere.

    Raises:
        ValueError: If the reference image does not have a valid 3D array.
    """
    if reference_image.array.ndim != 3:
        raise ValueError(
            f"reference_image.array must be 3D, got shape {reference_image.array.shape}"
        )

    mask_array = np.ones(reference_image.array.shape, dtype=dtype)
    return Image(
        array=mask_array,
        spacing=reference_image.spacing,
        origin=reference_image.origin,
        direction=reference_image.direction,
        modality="mask",
    )


def _find_best_dicom_series_dir(root: Path) -> Path:
    """Recursively find the subdirectory with the most DICOM files."""
    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")

    best_dir = None
    best_count = -1

    # Include root itself in the search
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]

    found_any = False

    for d in candidates:
        try:
            # Count DICOMs using pydicom's robust check
            count = sum(
                1 for f in d.iterdir() if f.is_file() and pydicom.misc.is_dicom(f)
            )
            if count > 0:
                found_any = True

            if count > best_count:
                best_count = count
                best_dir = d
        except OSError:
            continue

    if not found_any or best_dir is None or best_count == 0:
        raise ValueError(f"No DICOM files found in {root} or its subdirectories.")

    return best_dir


def load_image(path: str, dataset_index: int = 0, recursive: bool = False) -> Image:
    """
    Load a medical image from a file path or directory.

    This is the main entry point for loading data. It automatically detects whether
    the input is a NIfTI file or a DICOM directory/file (single DICOM or series)
    and standardizes it into an `Image` object.

    The resulting image array is always 3D with dimensions (x, y, z).

    Args:
        path (str): The absolute or relative path to the image file (e.g., .nii.gz,
            .dcm or file with no extension) or the directory containing DICOM files.
        dataset_index (int, optional): For 4D datasets (like fMRI or dynamic PET),
            this specifies which volume (time point) to extract. Defaults to 0 (the first volume).
        recursive (bool, optional): If True and `path` is a directory, recursively searches
            subdirectories and loads the DICOM series from the folder containing the most
            DICOM files. Defaults to False.

    Returns:
        Image: An `Image` object containing the 3D numpy array and metadata (spacing, origin, etc.).

    Raises:
        ValueError: If the path does not exist, the file format is not supported,
            or the file is corrupt/unreadable.

    Examples:
        **Loading a NIfTI file:**
        ```python
        from pictologics.loader import load_image

        # Load a standard brain scan
        img = load_image("data/brain.nii.gz")
        print(f"Image shape: {img.array.shape}")
        # Output: Image shape: (256, 256, 128)
        ```

        **Loading a DICOM series:**
        ```python
        # Load a CT scan from a folder of DICOM files
        img_ct = load_image("data/patients/001/CT_scan/")
        print(f"Voxel spacing: {img_ct.spacing}")
        # Output: Voxel spacing: (0.97, 0.97, 2.5)
        ```

        **Loading a single DICOM file:**
        ```python
        # Load a single DICOM file (even without .dcm extension)
        img_slice = load_image("data/slice_001")
        print(f"Modality: {img_slice.modality}")
        ```

        **Recursive DICOM loading:**
        ```python
        # Finds the deep subfolder with actual DICOM files
        img = load_image("data/patients/001/", recursive=True)
        ```

        **Loading a specific volume from a 4D file:**
        ```python
        # Load the 5th time point from a 4D fMRI file
        fmri_vol = load_image("data/fmri.nii.gz", dataset_index=4)
        ```
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"The specified path does not exist: {path}")

    try:
        if path_obj.is_dir():
            target_path = path_obj
            if recursive:
                target_path = _find_best_dicom_series_dir(path_obj)
            return _load_dicom_series(target_path)
        elif path.lower().endswith((".nii", ".nii.gz")):

            return _load_nifti(path, dataset_index)
        else:
            # Attempt to load as a single DICOM file if extension is not NIfTI
            try:
                return _load_dicom_file(path)
            except Exception:
                raise ValueError(
                    f"Unsupported file format or unable to read file: {path}"
                ) from None
    except Exception as e:
        # Re-raise ValueErrors directly, wrap others
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to load image from '{path}': {e}") from e


def load_and_merge_images(
    image_paths: list[str],
    reference_image: Optional[Image] = None,
    conflict_resolution: str = "max",
    dataset_index: int = 0,
    recursive: bool = False,
    binarize: bool | int | list[int] | tuple[int, int] | None = None,
) -> Image:
    """
    Load multiple images (e.g., masks or partial scans) and merge them into a single image.

    This function loads images from the provided paths, validates that they all share
    the same geometry (dimensions, spacing, origin, direction), and merges them
    according to the specified conflict resolution strategy.

    **Use Cases:**
    - Merging multiple segmentation masks into a single ROI.
    - Merging split image volumes (though typically less common than mask merging).

    **Format & Path Support:**
    Since this function uses `load_image` internally for each path, it supports:
    - **NIfTI files** (.nii, .nii.gz).
    - **DICOM series** (directories containing DICOM files).
    - **Single DICOM files** (with or without .dcm extension).
    - **Nested directories** (if paths point to folders containing DICOMs).

    Args:
        image_paths (list[str]): List of absolute or relative paths to the images.
            These can be file paths or directory paths.
        reference_image (Optional[Image]): An optional reference image (e.g., the scan
            corresponding to the masks). If provided, the merged image is validated
            against this image's geometry.
        conflict_resolution (str): Strategy to resolve voxel values when multiple images
            have non-zero values at the same location. Options:
            - 'max': Use the maximum value (default).
            - 'min': Use the minimum value.
            - 'first': Keep the value from the first image encountered (earlier in list).
            - 'last': Overwrite with the value from the last image encountered (later in list).
        dataset_index (int, optional): For 4D datasets, this specifies which volume
            (time point) to extract for all images. Defaults to 0.
        recursive (bool, optional): If True, recursively searches subdirectories
            for each path in `image_paths`. Defaults to False.
        binarize (bool | int | list[int] | tuple[int, int] | None, optional):
            Rules for binarizing the merged image.
            - `None` (default): No binarization.
            - `True`: Sets all voxels > 0 to 1, others to 0.
            - `int` (e.g., 2): Sets voxels == value to 1, others to 0.
            - `list[int]` (e.g., [1, 2]): Sets voxels in list to 1, others to 0.
            - `tuple[int, int]` (e.g., (1, 10)): Sets voxels in inclusive range to 1, others to 0.

    **Note on Filtering:**
    The `binarize` parameter is intended for **mask filtering** (e.g., selecting specific ROI labels).
    To filter image intensity values (e.g., HU ranges), use the preprocessing steps in the
    radiomics pipeline configuration instead.

    Returns:
        Image: A new `Image` object containing the merged data.

    Raises:
        ValueError: If `image_paths` is empty, if an invalid `conflict_resolution` is provided,
            or if the images (or reference) have mismatched geometries.
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty.")

    valid_strategies = {"max", "min", "first", "last"}
    if conflict_resolution not in valid_strategies:
        raise ValueError(
            f"Invalid conflict_resolution '{conflict_resolution}'. "
            f"Must be one of {valid_strategies}."
        )

    # Load the first image to serve as the consensus geometry
    try:
        consensus_image = load_image(
            image_paths[0], dataset_index=dataset_index, recursive=recursive
        )
    except Exception as e:
        raise ValueError(f"Failed to load first image '{image_paths[0]}': {e}") from e

    merged_array = consensus_image.array.copy()

    # Geometry validation helper
    def _validate_geometry(target: Image, ref: Image, name: str, ref_name: str) -> None:
        if target.array.shape != ref.array.shape:
            raise ValueError(
                f"Dimension mismatch between {name} {target.array.shape} "
                f"and {ref_name} {ref.array.shape}."
            )
        if not np.allclose(target.spacing, ref.spacing, atol=1e-5):
            raise ValueError(
                f"Spacing mismatch between {name} {target.spacing} "
                f"and {ref_name} {ref.spacing}."
            )
        if not np.allclose(target.origin, ref.origin, atol=1e-5):
            raise ValueError(
                f"Origin mismatch between {name} {target.origin} "
                f"and {ref_name} {ref.origin}."
            )
        if target.direction is not None and ref.direction is not None:
            if not np.allclose(target.direction, ref.direction, atol=1e-5):
                raise ValueError(f"Direction mismatch between {name} and {ref_name}.")

    # Iterate through remaining images
    for path in image_paths[1:]:
        try:
            current_image = load_image(
                path, dataset_index=dataset_index, recursive=recursive
            )
        except Exception as e:
            raise ValueError(f"Failed to load image '{path}': {e}") from e

        _validate_geometry(
            current_image, consensus_image, f"image '{path}'", "consensus image"
        )

        current_array = current_image.array

        # Identify regions
        # Overlap: non-zero in both
        overlap_mask = (merged_array != 0) & (current_array != 0)
        # New data: zero in merged, non-zero in current
        new_data_mask = (merged_array == 0) & (current_array != 0)

        # Apply new data (always acceptable)
        merged_array[new_data_mask] = current_array[new_data_mask]

        # Resolve conflicts in overlapping regions
        if np.any(overlap_mask):
            if conflict_resolution == "max":
                merged_array[overlap_mask] = np.maximum(
                    merged_array[overlap_mask], current_array[overlap_mask]
                )
            elif conflict_resolution == "min":
                merged_array[overlap_mask] = np.minimum(
                    merged_array[overlap_mask], current_array[overlap_mask]
                )
            elif conflict_resolution == "last":
                merged_array[overlap_mask] = current_array[overlap_mask]
            elif conflict_resolution == "first":
                pass  # Already have the 'first' value, do nothing

    # Apply binarization if requested
    if binarize is not None:
        mask_out = np.zeros_like(merged_array, dtype=np.uint8)
        if isinstance(binarize, bool) and binarize is True:
            mask_out[merged_array > 0] = 1
        elif isinstance(binarize, int) and not isinstance(binarize, bool):
            mask_out[merged_array == binarize] = 1
        elif isinstance(binarize, list):
            mask_out[np.isin(merged_array, binarize)] = 1
        elif isinstance(binarize, tuple) and len(binarize) == 2:
            mask_out[(merged_array >= binarize[0]) & (merged_array <= binarize[1])] = 1
        else:
            # Fallback for False or unknown types, though type hint restricts this
            if binarize is not False:
                raise ValueError(f"Unsupported binarize value: {binarize}")
            mask_out = merged_array  # return original as fallback

        if binarize is not False:
            merged_array = mask_out

    # Validate against reference image if provided
    if reference_image is not None:
        # Create a dummy image wrapping the merged array for validation
        final_merged_image = Image(
            array=merged_array,
            spacing=consensus_image.spacing,
            origin=consensus_image.origin,
            direction=consensus_image.direction,
            modality="Image",
        )
        _validate_geometry(
            final_merged_image, reference_image, "merged image", "reference image"
        )

    return Image(
        array=merged_array,
        spacing=consensus_image.spacing,
        origin=consensus_image.origin,
        direction=consensus_image.direction,
        modality="MergedImage",
    )


def _ensure_3d(array: np.ndarray, dataset_index: int = 0) -> np.ndarray:
    """
    Ensure the input array is strictly 3D (x, y, z).

    This helper function handles different input dimensionalities:
    - **2D (x, y)**: Promoted to 3D by adding a singleton dimension (x, y, 1).
    - **3D (x, y, z)**: Returned as is.
    - **4D (x, y, z, t)**: The volume at `dataset_index` is extracted.

    Args:
        array (np.ndarray): The input numpy array of arbitrary dimensions.
        dataset_index (int): The index of the volume to extract if the input is 4D.

    Returns:
        np.ndarray: A 3D numpy array.

    Raises:
        ValueError: If the array has an unsupported number of dimensions (not 2, 3, or 4)
            or if `dataset_index` is invalid for the 4D array.
    """
    ndim = array.ndim
    if ndim == 2:
        # (x, y) -> (x, y, 1)
        return array[..., np.newaxis]
    elif ndim == 3:
        return array
    elif ndim == 4:
        if dataset_index < 0 or dataset_index >= array.shape[3]:
            raise ValueError(
                f"Dataset index {dataset_index} is out of bounds for 4D image "
                f"with {array.shape[3]} volumes."
            )
        return array[..., dataset_index]
    else:
        raise ValueError(
            f"Unsupported array dimensionality: {ndim}. Expected 2, 3, or 4."
        )


def _load_nifti(path: str, dataset_index: int = 0) -> Image:
    """
    Load a NIfTI file (.nii or .nii.gz) using the nibabel library.

    This function extracts the image data, voxel spacing, origin, and direction
    from the NIfTI header.

    Args:
        path (str): Path to the NIfTI file.
        dataset_index (int): The volume index to load if the file is 4D.

    Returns:
        Image: A standardized `Image` object.

    Raises:
        ValueError: If nibabel fails to load the file (e.g., corrupt header).
    """
    try:
        nii_img = nib.load(path)  # type: ignore
    except Exception as e:
        raise ValueError(f"Could not load NIfTI file '{path}': {e}") from e

    # Load image data as float64 to preserve precision
    array = nii_img.get_fdata()  # type: ignore
    array = _ensure_3d(array, dataset_index)

    # Extract metadata
    header = nii_img.header  # type: ignore
    zooms = header.get_zooms()  # type: ignore

    # Ensure spacing has at least 3 dimensions (pad with 1.0 if needed)
    spacing_list = [float(z) for z in zooms]
    while len(spacing_list) < 3:
        spacing_list.append(1.0)
    spacing = (spacing_list[0], spacing_list[1], spacing_list[2])

    # Extract affine for origin and direction
    affine = nii_img.affine  # type: ignore
    origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
    direction = affine[:3, :3]

    return Image(
        array=array,
        spacing=spacing,
        origin=origin,
        direction=direction,
        modality="Nifti",
    )


def _load_dicom_series(path: str | Path) -> Image:
    """
    Load a DICOM series (a set of DICOM files) from a directory.

    This function reads all DICOM files in the directory, sorts them spatially
    to reconstruct the 3D volume, and extracts metadata.

    **Sorting Logic:**
    Slices are sorted based on the projection of their `ImagePositionPatient`
    onto the slice normal vector (derived from `ImageOrientationPatient`).
    This robustly handles axial, sagittal, coronal, and oblique acquisitions.
    If spatial tags are missing, it falls back to `InstanceNumber`.

    Args:
        path (str): Directory containing the DICOM files.

    Returns:
        Image: A standardized `Image` object.

    Raises:
        ValueError: If no DICOM files are found or if they cannot be read/sorted.
    """
    # List all DICOM files
    path_obj = Path(path)
    files = [p for p in path_obj.iterdir() if p.is_file() and pydicom.misc.is_dicom(p)]
    if not files:
        raise ValueError(f"No DICOM files found in directory: {path}")

    # Read all slices
    try:
        slices = [pydicom.dcmread(f) for f in files]
    except Exception as e:
        raise ValueError(f"Error reading DICOM files in '{path}': {e}") from e

    # Determine sorting direction
    # Calculate the normal vector of the slice plane
    ref = slices[0]
    try:
        orientation = np.array(ref.ImageOrientationPatient, dtype=float)
        row_cosines = orientation[:3]
        col_cosines = orientation[3:]
        slice_normal = np.cross(row_cosines, col_cosines)
    except (AttributeError, ValueError):
        # Fallback to simple Z-sorting if orientation is missing
        slice_normal = np.array([0, 0, 1.0])

    # Sort slices by projection of position onto the normal vector
    try:
        slices.sort(
            key=lambda s: np.dot(
                np.array(s.ImagePositionPatient, dtype=float), slice_normal
            )
        )
    except AttributeError:
        # Fallback to InstanceNumber if ImagePositionPatient is missing
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

    # Stack pixel data
    # pydicom pixel_array is (Rows, Columns) -> (Y, X)
    # We want (X, Y, Z)
    try:
        pixel_data = [s.pixel_array for s in slices]
    except Exception as e:
        raise ValueError("Failed to extract pixel arrays from DICOM slices.") from e

    volume = np.stack(pixel_data, axis=-1)  # Result: (Y, X, Z)
    volume = np.swapaxes(volume, 0, 1)  # Result: (X, Y, Z)
    volume = _ensure_3d(volume)

    # Extract metadata from the first slice (reference)
    ref = slices[0]

    # Spacing
    try:
        pixel_spacing = ref.PixelSpacing
        spacing_x = float(pixel_spacing[1])  # Column spacing (X)
        spacing_y = float(pixel_spacing[0])  # Row spacing (Y)

        # Slice thickness / spacing
        if hasattr(ref, "SpacingBetweenSlices"):
            spacing_z = float(ref.SpacingBetweenSlices)
        elif hasattr(ref, "SliceThickness"):
            spacing_z = float(ref.SliceThickness)
        else:
            # Estimate from position difference if multiple slices
            if len(slices) > 1:
                p1 = np.array(slices[0].ImagePositionPatient)
                p2 = np.array(slices[1].ImagePositionPatient)
                spacing_z = float(np.linalg.norm(p2 - p1))
            else:
                spacing_z = 1.0

        spacing = (spacing_x, spacing_y, spacing_z)
    except (AttributeError, IndexError):
        spacing = (1.0, 1.0, 1.0)

    # Origin
    try:
        origin = (
            float(ref.ImagePositionPatient[0]),
            float(ref.ImagePositionPatient[1]),
            float(ref.ImagePositionPatient[2]),
        )
    except AttributeError:
        origin = (0.0, 0.0, 0.0)

    # Direction
    try:
        orientation = np.array(ref.ImageOrientationPatient, dtype=float)
        row_cosines = orientation[:3]
        col_cosines = orientation[3:]
        slice_cosine = np.cross(row_cosines, col_cosines)
        direction = np.stack([row_cosines, col_cosines, slice_cosine], axis=1)
    except (AttributeError, ValueError):
        direction = np.eye(3)

    # Apply Rescale Slope and Intercept (Hounsfield Units conversion)
    slope = getattr(ref, "RescaleSlope", 1.0)
    intercept = getattr(ref, "RescaleIntercept", 0.0)

    if slope != 1.0 or intercept != 0.0:
        volume = volume * slope + intercept

    return Image(
        array=volume,
        spacing=spacing,
        origin=origin,
        direction=direction,
        modality=getattr(ref, "Modality", "DICOM"),
    )


def _load_dicom_file(path: str) -> Image:
    """
    Load a single DICOM file as a 3D image.

    This is useful for 2D X-rays or single slices. The resulting image will
    have a shape of (x, y, 1).

    Args:
        path (str): Path to the DICOM file.

    Returns:
        Image: A standardized `Image` object.

    Raises:
        ValueError: If the file is not a valid DICOM file.
    """
    try:
        dcm = pydicom.dcmread(path)
        data = dcm.pixel_array
    except Exception as e:
        raise ValueError(f"Corrupt or invalid DICOM file '{path}': {e}") from e

    # Handle dimensions
    # DICOM pixel_array is (Rows, Columns) -> (Y, X). Swap to (X, Y).
    if data.ndim == 2:
        data = np.swapaxes(data, 0, 1)

    data = _ensure_3d(data)

    # Metadata extraction (simplified for single file)
    try:
        ps = dcm.PixelSpacing
        spacing = (
            float(ps[1]),
            float(ps[0]),
            float(getattr(dcm, "SliceThickness", 1.0)),
        )
    except (AttributeError, IndexError):
        spacing = (1.0, 1.0, 1.0)

    try:
        ipp = dcm.ImagePositionPatient
        origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    except AttributeError:
        origin = (0.0, 0.0, 0.0)

    # Rescale
    slope = getattr(dcm, "RescaleSlope", 1.0)
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    if slope != 1.0 or intercept != 0.0:
        data = data * slope + intercept

    return Image(
        array=data,
        spacing=spacing,
        origin=origin,
        direction=np.eye(
            3
        ),  # Direction is ambiguous for single slice without orientation
        modality=getattr(dcm, "Modality", "DICOM"),
    )
