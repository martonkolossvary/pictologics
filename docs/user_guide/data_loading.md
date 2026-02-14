# Data Loading

This guide covers all aspects of loading medical imaging data into Pictologics. Whether you're working with NIfTI files, DICOM series, multi-phase acquisitions, or segmentation masks, this page will help you get your data into the `Image` class for radiomics analysis.

## The Image Class

All data in Pictologics is represented by the `Image` dataclass, which provides a standardized container for 3D medical image data. All data are stored as 3D numpy arrays, with additional metadata to describe the geometry of the data. If 2D data is provided, it is converted to a 3D numpy array with a singleton dimension.

```python
from pictologics import Image

# Image attributes:
# - array: numpy.ndarray (3D, in X, Y, Z order)
# - spacing: tuple[float, float, float] (voxel dimensions in mm)
# - origin: tuple[float, float, float] (world coordinates of first voxel)
# - direction: Optional[numpy.ndarray] (3x3 direction cosine matrix)
# - modality: str (e.g., "CT", "MR", "Unknown")
```

!!! note
    Pictologics uses **(X, Y, Z)** axis ordering to match ITK/SimpleITK conventions. This differs from raw DICOM (which uses Rows, Columns = Y, X) and matplotlib (which expects height, width = Y, X). All loaders handle these transformations automatically.

## Basic Loading with `load_image()`

The `load_image()` function is the primary entry point for loading data. It automatically detects the file format and handles the appropriate loading strategy.

### Loading NIfTI Files

```python
from pictologics import load_image

# Load a NIfTI file (.nii or .nii.gz)
image = load_image("path/to/scan.nii.gz")
mask = load_image("path/to/segmentation.nii.gz")

print(f"Shape: {image.array.shape}")
print(f"Spacing: {image.spacing}")
print(f"Origin: {image.origin}")
```

### Loading DICOM Series

For a directory containing DICOM files from a single series:

```python
# Load all DICOM files in a directory as a single volume
image = load_image("path/to/dicom_folder/")

# Pictologics automatically:
# - Finds all DICOM files in the directory
# - Sorts slices by spatial position
# - Extracts spacing, origin, and direction from headers
# - Stacks slices into a 3D volume
```

### Loading a Single DICOM File

Single DICOM files (e.g., enhanced DICOM, segmentation objects) are also supported:

```python
# Load a single DICOM file
image = load_image("path/to/image.dcm")
```

### DICOM Intensity Rescaling

By default, `load_image()` applies **RescaleSlope** and **RescaleIntercept** transformations to DICOM data, converting stored pixel values to real-world values (e.g., Hounsfield Units for CT). This matches the behavior of NIfTI loading, which always applies its scaling factors.

```python
# Default: values are converted (e.g., to Hounsfield Units)
ct = load_image("ct_scan/")
print(ct.array.min(), ct.array.max())  # e.g., -1024.0 to 3000.0

# If you need raw stored pixel values:
ct_raw = load_image("ct_scan/", apply_rescale=False)
print(ct_raw.array.min(), ct_raw.array.max())  # e.g., 0 to 4095
```

| Format | Rescaling Behavior |
|--------|-------------------|
| **NIfTI** | Always applies `scl_slope` and `scl_inter` from header |
| **DICOM** | Applies `RescaleSlope` and `RescaleIntercept` when `apply_rescale=True` (default) |

### Handling Sentinel (NA) Values

Medical imaging formats often use a **sentinel value** to represent missing or invalid data. Common examples:

| Modality | Common Sentinel Values |
|----------|----------------------|
| CT | -1024, -2048, -32768 (outside tissue HU range) |
| MR | 0 (often used for background/air) |
| PET | 0 or negative values |

Since DICOM uses integer storage and cannot represent `NaN`, these sentinel values are substituted for missing data. Pictologics offers two approaches for handling them:

#### Approach 1: Resegmentation (Simple)

Use the **`resegment`** preprocessing step to exclude sentinel values by restricting the ROI to a valid intensity range:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
pipeline.add_config("ct_analysis", [
    # Exclude sentinel values by filtering to valid HU range
    {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "texture"]}},
])
```

!!! warning "Limitations of resegmentation alone"
    Resegmentation removes sentinel voxels from the ROI **after** earlier pipeline steps have already
    run. This means:

    - **Resampling**: When the image is resampled to a new voxel spacing, the interpolation kernel
      reads neighboring voxels — including sentinel values. A valid voxel next to a -2048 sentinel
      will receive a blended value that is far below its true intensity, corrupting the resampled
      output.
    - **Filtering**: Convolution-based filters (e.g., LoG, Gabor, wavelets) sum intensities over
      a local neighborhood. Any sentinel voxels within the kernel window contribute their artificial
      values to the filter response, producing incorrect texture and edge features.

    This approach works well when the pipeline **only** discretises and extracts features (no
    resampling or filtering). For pipelines that include resampling or filtering, combine
    resegmentation with **Approach 2** (source masking) — see below.

#### Approach 2: Source Masking (Protects Resampling & Filtering)

When your images contain sentinel values that could **contaminate resampling and filtering** (e.g., pre-cropped lesion exports), use sentinel detection and source masking. This creates a *source mask* that ensures sentinel voxels are excluded from interpolation and convolution operations:

```python
from pictologics import load_image
from pictologics.preprocessing import detect_sentinel_value, create_source_mask_from_sentinel

image = load_image("lesion_export.nii.gz")

# Automatically detect sentinel value
sentinel = detect_sentinel_value(image)
print(f"Detected sentinel: {sentinel}")  # e.g., -2048.0

# Create source mask (1 = valid, 0 = sentinel)
source_mask = create_source_mask_from_sentinel(image, sentinel)
```

The source mask enables **masked interpolation** for resampling and **normalized convolution** for filters, preventing sentinel values from bleeding into valid regions.

!!! warning "Source masking does not replace resegmentation"
    The source mask protects **resampling and filtering only** — it does **not** change the ROI
    used for feature extraction. Sentinel voxels will still be included in intensity statistics,
    texture matrices, and all other computed features unless you also add a `resegment` step to
    restrict the ROI to a valid intensity range. In practice, you typically need **both**:

    - `source_mode="auto"` (or an explicit source mask) to protect preprocessing and **prevent memory exhaustion**.
    - `resegment` to define the correct ROI for feature extraction.

    **Critical Note:** Without `source_mode="auto"`, resampled background voxels (often 0) may fall within your `resegment` range (e.g., -100 to 3000). This causes the **entire image volume** to be included in the ROI, leading to huge memory usage and slow GLCM calculations. `source_mode="auto"` ensures these background voxels are excluded from the ROI.

    See the **[Quick Start](quick_start.md)** guide for a complete workflow and the
    **[Cookbook](cookbook.md)** for batch processing examples.

    #### Decision Guide: When to use `source_mode`?

    | Image Type | Example | Recommended Mode | Why? |
    | :--- | :--- | :--- | :--- |
    | **Full FOV Scan** | Standard CT/MRI (rectangular, includes air/background) | `"full_image"` (Default) | Entire image contains valid physical measurements (even air is approx -1000 HU). No artificial edges to protect. |
    | **Pre-processed / Cropped** | Skull-stripped brain, cardiac ROI crop, or image with applied mask (background = 0 or -2048) | `"auto"` | The background is *artificial*. Resampling near the tissue edge would blend valid tissue with invalid background (0), corrupting values. `auto` masking prevents this. |
    | **ROI Mask Provided?** | You have a separate segmentation file (e.g., `liver_mask.nii.gz`) | **Matches Image Type** | The *ROI mask* tells us *where* to extract features. The `source_mode` tells us *what pixel values are valid* for interpolation. Use `"auto"` if the *image itself* has invalid background; use `"full_image"` if it's a raw scan. |

    **Summary**:
    - **Raw Scan + Mask**: Use `source_mode="full_image"` (default).
    - **Masked/Cropped Image**: Use `source_mode="auto"`.

!!! tip
    Check your data's minimum value to identify potential sentinels:
    ```python
    image = load_image("scan.dcm")
    print(f"Min: {image.array.min()}, Max: {image.array.max()}")
    # If min is -1024 or -2048, those are likely sentinels
    ```

## Multi-Phase DICOM Series

Many clinical acquisitions contain multiple phases (e.g., cardiac CT with multiple timepoints). Pictologics can detect and load specific phases from datasets.

### Discovering Available Phases

Use `get_dicom_phases()` to explore what's available before loading:

```python
from pictologics.utilities import get_dicom_phases

# Discover phases in a multi-phase DICOM directory
phases = get_dicom_phases("path/to/cardiac_ct/")

print(f"Found {len(phases)} phases:")
for phase in phases:
    print(f"--- Phase {phase.index} ---")
    print(f"Label:       {phase.label}")
    print(f"Split Tag:   {phase.split_tag}")
    print(f"Split Value: {phase.split_value}")
    print(f"Num Slices:  {phase.num_slices}")
    # Show first file path as example
    print(f"Example File: {phase.file_paths[0].name}")
```

Example output:
```text
Found 10 phases:
--- Phase 0 ---
Label:       Phase 0%
Split Tag:   NominalPercentageOfCardiacPhase
Split Value: 0
Num Slices:  256
Example File: IM-0001-0001.dcm
--- Phase 1 ---
Label:       Phase 10%
Split Tag:   NominalPercentageOfCardiacPhase
Split Value: 10
Num Slices:  256
Example File: IM-0001-0225.dcm
...
```

Each `DicomPhaseInfo` object contains:

- `index`: Phase index (0, 1, 2, ...)
- `label`: Human-readable label (e.g., "CardiacPhase=0", "TemporalPosition=1")
- `num_slices`: Number of slices in this phase
- `split_tag`: The DICOM tag used for detection
- `split_value`: The actual tag value


### Loading a Specific Phase

Use the `dataset_index` parameter to load a particular phase:

```python
# Load the first phase (index 0)
phase_0 = load_image("path/to/cardiac_ct/", dataset_index=0)

# Load the second phase (index 1)
phase_1 = load_image("path/to/cardiac_ct/", dataset_index=1)
```

### Phase Detection Priority

Pictologics automatically detects phases using these DICOM tags (in order of priority):

1. **NominalPercentageOfCardiacPhase** - Cardiac phases (percentage)
2. **TemporalPositionIdentifier** - Temporal position index
3. **TriggerTime** - ECG trigger time
4. **AcquisitionNumber** - Acquisition sequence number
5. **EchoNumber** - Multi-echo MRI

## 4D NIfTI Files

NIfTI files can contain 4D data (3D + time/phase). Use `dataset_index` similarly:

```python
# Load a 4D NIfTI file - get the first volume
vol_0 = load_image("path/to/4d_data.nii.gz", dataset_index=0)

# Load the second volume
vol_1 = load_image("path/to/4d_data.nii.gz", dataset_index=1)
```

## DICOM Segmentation (SEG) Files

DICOM SEG files are specialized objects containing segmentation masks. Use `load_seg()` for full control, or let `load_image()` auto-detect them.

### Auto-Detection in load_image()

```python
# load_image() automatically detects DICOM SEG files
mask = load_image("path/to/segmentation.dcm")
```

### Detailed Control with load_seg()

```python
from pictologics import load_seg
from pictologics.loaders import get_segment_info
import numpy as np

# First, inspect what segments are available
segments = get_segment_info("path/to/segmentation.dcm")
for seg in segments:
    print(f"Segment {seg['segment_number']}: {seg['segment_label']}")

# Load all segments combined into a single label mask
# Each segment gets its numeric label (1, 2, 3, etc.)
combined_mask = load_seg("path/to/segmentation.dcm")
print(np.unique(combined_mask.array))  # [0, 1, 2, 3, ...]
# Background = 0, Segment 1 = 1, Segment 2 = 2, etc.

# Load only specific segments
liver_mask = load_seg(
    "path/to/segmentation.dcm",
    segment_numbers=[1, 2]  # Only segments 1 and 2
)

# Load segments separately (returns dict)
separate_masks = load_seg(
    "path/to/segmentation.dcm",
    combine_segments=False
)
# separate_masks = {1: Image(...), 2: Image(...), ...}
```

### Working with Separate Segments

When using `combine_segments=False`, you can iterate over segments for individual analysis:

```python
# Get each segment as a separate binary mask
masks = load_seg("seg.dcm", combine_segments=False)

# Iterate over segments
for seg_num, mask in masks.items():
    print(f"Segment {seg_num}: {mask.array.sum()} voxels")

# Process each segment for radiomics
for seg_num, mask in masks.items():
    features = pipeline.run(image=ct, mask=mask)
```

### Combining Specific Segments into a Binary Mask

To merge selected segments into a single binary mask:

```python
# Load specific segments separately
masks = load_seg("seg.dcm", segment_numbers=[1, 2], combine_segments=False)

# Combine into single binary mask using logical OR
combined = masks[1].array | masks[2].array
```

### Aligning SEG to a Reference Image

SEG files may have different geometry than the source image. Use `reference_image` to align:

```python
# Load the CT image
ct = load_image("path/to/ct_series/")

# Load and align the segmentation to CT geometry
mask = load_seg(
    "path/to/segmentation.dcm",
    reference_image=ct
)

# Now mask.array.shape == ct.array.shape
```

## Merging Multiple Images with `load_and_merge_images()`

When you have multiple segmentation masks (e.g., different organs, or masks split across files), use `load_and_merge_images()` to combine them.

### Basic Merging

```python
from pictologics import load_and_merge_images

# Merge multiple mask files into one
combined_mask = load_and_merge_images([
    "path/to/liver_mask.nii.gz",
    "path/to/kidney_mask.nii.gz",
    "path/to/spleen_mask.nii.gz"
])
```

### Relabeling Masks for Visualization

When merging binary masks, assign unique labels to each:

```python
# Each mask gets a unique label (1, 2, 3, ...)
combined = load_and_merge_images(
    ["mask1.nii.gz", "mask2.nii.gz", "mask3.nii.gz"],
    relabel_masks=True
)
# Result: voxels from mask1 = 1, mask2 = 2, mask3 = 3
```

### Merge Strategy Options

Control how overlapping voxels are handled:

```python
# "max" (default): Take the maximum value at each voxel
combined = load_and_merge_images(masks, conflict_resolution="max")

# "min": Take the minimum value at each voxel
combined = load_and_merge_images(masks, conflict_resolution="min")

# "first": Keep the first non-zero value
combined = load_and_merge_images(masks, conflict_resolution="first")

# "last": Keep the last non-zero value
combined = load_and_merge_images(masks, conflict_resolution="last")
```

## Handling Cropped Masks

Medical imaging software often stores segmentation masks as **cropped volumes** (bounding boxes around the region of interest) to minimize storage. When loading these cropped masks, they need to be repositioned into the original image's coordinate space for proper visualization and analysis.

The `pictologics` loader uses the spatial metadata (`ImagePositionPatient` for DICOM, affine matrix for NIfTI) to calculate where the cropped mask belongs in the full volume.


### Repositioning a Single Cropped Mask

```python
# Load the full CT image
ct = load_image("path/to/full_ct/")

# Load a cropped mask and reposition it
cropped_mask = load_image(
    "path/to/cropped_mask.nii.gz",
    reference_image=ct
)
# cropped_mask now has the same shape as ct
```

### Merging Multiple Cropped Masks

```python
# Load CT as reference
ct = load_image("path/to/ct/")

# Merge cropped masks into reference space
combined = load_and_merge_images(
    ["cropped_liver.nii.gz", "cropped_kidney.nii.gz"],
    reference_image=ct,
    reposition_to_reference=True,
    relabel_masks=True
)
```

### Handling Axis Transposition

If your masks have different axis ordering (e.g., from different software), specify the transformation:

```python
combined = load_and_merge_images(
    mask_paths,
    reference_image=ct,
    reposition_to_reference=True,
    transpose_axes=(1, 0, 2)  # Swap X and Y axes
)
```

### Error Handling

| Issue | Behavior |
|-------|----------|
| Spacing mismatch | `ValueError` raised (resampling not yet supported) |
| Orientation mismatch | Warning emitted, positioning continues |
| Mask outside reference bounds | Warning emitted, empty volume returned |
| Partial overlap | Valid region is positioned, rest is clipped |

!!! tip
    **Label Order**: When using `relabel_masks=True`, labels are assigned based on the order of files in `image_paths`. Use `sorted()` for consistent ordering, or specify the exact order you want.

## Creating a Full Mask

When you don't have a segmentation mask and want to analyze the entire image:

```python
from pictologics import create_full_mask

# Create a mask of all ones matching the image geometry
image = load_image("scan.nii.gz")
full_mask = create_full_mask(image)

# Now use full_mask for whole-image analysis
```

!!! tip
    If you pass `mask=None` to `RadiomicsPipeline.run()`, it automatically creates a full mask internally.



## Summary of Loading Functions

| Function | Purpose |
|----------|---------|
| `load_image()` | Main entry point - loads NIfTI, DICOM series, single DICOM, or DICOM SEG |
| `load_seg()` | Detailed DICOM SEG loading with segment selection and alignment |
| `get_segment_info()` | Inspect available segments in a DICOM SEG file |
| `load_and_merge_images()` | Combine multiple images/masks with various strategies |
| `create_full_mask()` | Create an all-ones mask matching image geometry |
| `get_dicom_phases()` | Discover available phases in multi-phase DICOM |

## Next Steps

- [Pipeline & Preprocessing](pipeline.md) - Configure and run the radiomics pipeline
- [Cookbook](cookbook.md) - End-to-end batch processing scripts
