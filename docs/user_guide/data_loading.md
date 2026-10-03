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
# - frame_of_reference_uid: Optional[str] (the DICOM FrameOfReferenceUID, or None)
```

!!! note
    Pictologics uses **(X, Y, Z)** axis ordering to match ITK/SimpleITK conventions. This differs from raw DICOM (which uses Rows, Columns = Y, X) and matplotlib (which expects height, width = Y, X). All loaders handle these transformations automatically.

!!! note "Physical Geometry"
    `Image.direction` stores unit direction cosines, while voxel sizes are stored separately in `Image.spacing`. NIfTI affine columns are normalized on load, and DICOM row/column orientation is converted to the same `(X, Y, Z)` convention.

!!! note "World Frame"
    `Image.origin` and `Image.direction` are in the LPS+ world frame (Left, Posterior, Superior) for every format, as in DICOM and ITK/SimpleITK. A NIfTI affine is in RAS+, so the loader changes the sign of its X and Y rows. Give an in-memory `Image` its geometry in LPS+ too.

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

### Loading NRRD and MetaImage Files

`load_image()` also reads NRRD files (`.nrrd`, and `.nhdr` headers with a detached data file) and MetaImage files (`.mha`, and `.mhd` headers with a `.raw` or `.zraw` data file), as 3D Slicer and ITK write them. Pictologics reads them with its own readers, so you do not need another package.

```python
image = load_image("path/to/scan.nrrd")
mask = load_image("path/to/segmentation.seg.nrrd", reference_image=image)
```

- The geometry is in the LPS+ frame, as for every format. The loader converts a NRRD file in the RAS or LAS space.
- A file with one more axis (a 4D image, or the layers of a `.seg.nrrd` file) gives the volume of `dataset_index`, as a 4D NIfTI file does.
- The readers take raw, gzip, bzip2 and text data (NRRD), and raw, compressed and text data (MetaImage). They do not take data in more than one file, or images with more than one channel.

A 3D Slicer `.seg.nrrd` file holds each segment as a label value in a layer. Overlapping segments need more than one layer. `get_segment_info()` lists the segments:

```python
from pictologics.loaders import get_segment_info

for segment in get_segment_info("path/to/segmentation.seg.nrrd"):
    print(segment["segment_label"], segment["label_value"], segment["layer"])

# The layer of a segment: its voxels hold its label value
layer = load_image("path/to/segmentation.seg.nrrd", reference_image=image, dataset_index=1)
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

### Compressed DICOM Data

Compressed pixel data loads like uncompressed data: RLE, JPEG Lossless, JPEG-LS (lossless and near-lossless), JPEG 2000 (lossless and lossy) and baseline JPEG. pydicom decodes it with python-gdcm and Pillow, which install with Pictologics. SEG files with compressed frames load the same way.

12-bit lossy JPEG (JPEG Extended) does not load: its only decoder, pylibjpeg-libjpeg, has a GPL-3.0 license. If you install `pylibjpeg` and `pylibjpeg-libjpeg` yourself, pydicom uses them.

### DICOM Intensity Rescaling

By default, `load_image()` applies **RescaleSlope** and **RescaleIntercept** transformations to DICOM data, converting stored pixel values to real-world values (e.g., Hounsfield Units for CT). For DICOM series this is applied **per slice**, so series with slice-specific rescale metadata are handled correctly. This matches the behavior of NIfTI loading, which always applies its scaling factors.

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
| **DICOM** | Applies each slice's `RescaleSlope` and `RescaleIntercept` when `apply_rescale=True` (default) |

### PET Standardized Uptake Values (SUV)

`suv` converts a DICOM PET image (Modality PT) from activity concentration (Bq/ml) to its standardized uptake value when it loads:

```python
pet = load_image("pet_series/", suv="bw")
```

| `suv` | Normalised by | Unit |
|-------|---------------|------|
| `"bw"` | The body weight (PatientWeight) | g/ml |
| `"lbm"` | The lean body mass by the Janmahasatian formula, from the weight, the height (PatientSize) and the sex (PatientSex) | g/ml |
| `"lbm_james"` | The lean body mass by the James formula (PERCIST 1.0, and many older programs), from the same values | g/ml |
| `"bsa"` | The body surface area by the Du Bois formula, from the weight and the height | cm²/ml |

The factor follows the [QIBA vendor-neutral pseudo-code](https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV)). The images must be attenuation and decay corrected (CorrectedImage with ATTN and DECY, DecayCorrection START), and the injected dose decays from the injection to the series start. For a post-processed series (a series time after the acquisition), the start is the GE private scan time, else the start from the frame times, else the earliest acquisition. Units CNTS take the Philips private SUV factor, and Units GML are SUVbw already. DecayCorrection ADMIN (decay corrected to the injection) takes the dose without decay. A missing, empty or zero attribute raises an error that names it: the loader never guesses a value. The `"lbm"` formula is the one that Tahari et al. (J Nucl Med 2014) recommend for SUL in place of the James formula of PERCIST 1.0; DICOM names it SUVlbm(Janma). In a pipeline, give `image_options={"suv": "bw"}` to `run()`, `run_rois()` or a `run_batch()` case.

Checked on the [QIBA FDG-PET/CT digital reference object](https://depts.washington.edu/petctdro/DROsuv_main.html) (female and male, 2013): `"bw"` gives its SUV values (0.00, 1.00, 4.00, 0.10, 0.90 and the test voxels 4.11 and -0.11) to within 6.4e-5. The object gives its SUVlbm values with the James formula: `"lbm_james"` gives them (0.750 and 0.771 times SUVbw for the female and the male object), and `"lbm"` (Janmahasatian) gives 0.660 and 0.758. The James formula fails for very obese patients (its lean body mass falls, and can drop below 0, with more weight), so Tahari et al. recommend Janmahasatian.

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
    The source mask protects **resampling and filtering** and, after resampling, keeps ROI masks
    inside the valid source extent. It does **not** define a clinical/intensity compartment by
    itself, and it does not change feature-extraction masks in pipelines with no spatial step.
    Add a `resegment` step to restrict the ROI to the valid intensity range you want to measure.
    In practice, you typically need **both**:

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

In a pipeline, `image_options={"dataset_index": 1}` in `run()` loads that phase from the folder path.

### Phase Detection Priority

Pictologics automatically detects phases using these DICOM tags (in order of priority):

1. **NominalPercentageOfCardiacPhase** - Cardiac phases (percentage)
2. **TemporalPositionIdentifier** - Temporal position index
3. **TriggerTime** - ECG trigger time
4. **AcquisitionNumber** - Acquisition sequence number
5. **EchoNumbers** - Multi-echo MRI

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

# When loading a SEG as a pipeline mask path, RadiomicsPipeline.run()
# passes the image as reference_image automatically.
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run("path/to/ct_series/", "path/to/segmentation.dcm")
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

When a mask contains multiple labels in one volume, `RadiomicsPipeline` treats
all nonzero labels as one combined ROI by default. Label values are never used
as numeric weights in volume or texture calculations. Add a `binarize_mask` step
when the run should use only selected label values:

```python
pipeline.add_config("segment_2_only", [
    {"step": "binarize_mask", "params": {"mask_values": 2}},
    {"step": "extract_features", "params": {"families": ["morphology", "intensity"]}},
])
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
    reference_image=ct,
    subvoxel_tolerance=0.05,
    min_overlap_fraction=0.5,
)

# Now mask.array.shape == ct.array.shape
```

`load_seg()` uses the same reference-alignment controls as `load_image()` and
`load_and_merge_images()`, including `transpose_axes`, `subvoxel_tolerance`,
`subvoxel_warning_threshold`, and `min_overlap_fraction`.

## DICOM RTSTRUCT Files

A DICOM RT Structure Set (RTSTRUCT) holds the contours of each ROI as polygons, not voxels. Pictologics fills them onto the grid of the image that the contours belong to, so the image is necessary:

```python
from pictologics import RadiomicsPipeline, load_image, load_rtstruct
from pictologics.loaders import get_segment_info

ct = load_image("path/to/ct_folder/")

# The ROIs: ROI Number, name and number of closed contours
for roi in get_segment_info("path/to/rtstruct.dcm"):
    print(roi["segment_number"], roi["segment_label"], roi["contour_count"])

# One label image: the label of each ROI is its ROI Number
labels = load_image("path/to/rtstruct.dcm", reference_image=ct)

# Binary masks by ROI name, which keep overlapping ROIs (such as a GTV in a PTV)
masks = load_rtstruct("path/to/rtstruct.dcm", ct, roi_names=["GTV", "PTV"], combine_rois=False)
results = RadiomicsPipeline().run(ct, masks["GTV"], config_names=["standard_fbn_32"])
```

- A voxel is in an ROI when its center lies inside an odd number of the contours of the ROI on its slice (the even-odd rule). So a contour inside another contour cuts a hole.
- A center on an edge is inside on one side of the edge only, so two ROIs that share an edge share no voxel.
- The plane of each contour must be a slice plane of the image. A contour up to `subvoxel_tolerance` voxels (default 0.5) away from a slice is filled on the nearest slice, with a warning above `subvoxel_warning_threshold` (default 0.01).
- In one label image, a later ROI wins where ROIs overlap. Use `combine_rois=False` for overlapping ROIs.
- `run(image, mask="rtstruct.dcm")` fills the RTSTRUCT onto the grid of the image, as one label image.

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

A mask can hold the grid of the image in another voxel order: axes flipped or swapped, as some converters write NIfTI files. The loader then turns the mask to the voxel order of the image first. So a DICOM image and a NIfTI mask on the same grid load together:

```python
ct = load_image("path/to/dicom_folder/")
mask = load_image("path/to/segmentation.nii.gz", reference_image=ct)
# mask.array has the shape and the voxel order of ct.array
```


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

The loader turns the axes itself when the geometry of the file tells the voxel order. Use `transpose_axes` only for a file with swapped axes that its geometry does not show:

```python
combined = load_and_merge_images(
    mask_paths,
    reference_image=ct,
    reposition_to_reference=True,
    transpose_axes=(1, 0, 2)  # Swap X and Y axes
)
```

### Sub-Voxel Alignment and Overlap Safeguards

DICOM software from different vendors sometimes stores mask origins with small floating-point imprecision, resulting in fractional-voxel offsets relative to the reference image grid. Pictologics handles this gracefully with two configurable thresholds:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `subvoxel_warning_threshold` | `0.01` | Drift above this (~1% of a voxel) emits a `UserWarning` but snaps to the nearest voxel and continues. |
| `subvoxel_tolerance` | `0.5` | Drift above this raises a `ValueError`. At the default of 0.5 (the mathematical maximum for rounding), valid masks never error. Lower this to detect suspiciously imprecise coordinates. |
| `min_overlap_fraction` | `0.5` | At least 50% of the mask's voxel volume must lie within the reference image space, otherwise a `ValueError` is raised. This prevents silently loading masks from the wrong patient. Set to `0.0` to disable. |

```python
# Dataset with known DICOM precision quirks — suppress sub-voxel warnings
# by keeping the default tolerance (0.5) but raising the warning threshold:
combined = load_and_merge_images(
    mask_paths,
    reference_image=ct,
    reposition_to_reference=True,
    subvoxel_warning_threshold=0.1,   # only warn if drift > 10% of a voxel
)

# Stricter project — flag any drift above 5% of a voxel as an error:
combined = load_and_merge_images(
    mask_paths,
    reference_image=ct,
    reposition_to_reference=True,
    subvoxel_tolerance=0.05,
)
```

### Error Handling

| Issue | Behavior |
|-------|----------|
| Spacing mismatch | `ValueError` raised (resampling not yet supported) |
| Orientation mismatch | `ValueError` raised when the mask axes are not the image axes in another order or with other signs; resample the mask to the image grid |
| Sub-voxel drift > `subvoxel_warning_threshold` | `UserWarning` emitted, nearest-voxel snapping applied |
| Sub-voxel drift > `subvoxel_tolerance` | `ValueError` raised |
| Overlap fraction < `min_overlap_fraction` | `ValueError` raised to prevent wrong-patient mask loading |
| Mask outside reference bounds (`min_overlap_fraction=0.0`) | `UserWarning` emitted, empty volume returned |
| Mask of another DICOM frame of reference | `UserWarning` emitted: a DICOM image, a SEG and an RTSTRUCT keep their `FrameOfReferenceUID`, and a mask with another UID can belong to another scan |
| Partial overlap | Valid region is positioned, rest is clipped |

!!! tip
    **Label Order**: When using `relabel_masks=True`, labels are assigned based on the order of files in `image_paths`. Use `sorted()` for consistent ordering, or specify the exact order you want.

## Saving Images

`save_image()` writes an image, a mask or a response map as a NIfTI file (`.nii` or `.nii.gz`). The geometry goes back to the RAS+ affine of NIfTI, so `load_image()` reads the same array and geometry, and other tools (3D Slicer, ITK) read the same grid:

```python
from pictologics import load_image, load_rtstruct, save_image

ct = load_image("path/to/ct_folder/")
masks = load_rtstruct("path/to/rtstruct.dcm", ct, roi_names=["GTV"], combine_rois=False)
save_image(masks["GTV"], "gtv.nii.gz")  # an RTSTRUCT ROI as a NIfTI mask
```

- A bool mask is saved as uint8. Other arrays keep their type (float64 images, float32 response maps, uint8 masks).
- NIfTI keeps the geometry in float32 (about 1e-5 mm).

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
| `load_image()` | Main entry point - loads NIfTI, NRRD (also `.seg.nrrd`), MetaImage, DICOM series, single DICOM, DICOM SEG, or DICOM RTSTRUCT (with `reference_image`) |
| `load_seg()` | Detailed DICOM SEG loading with segment selection and alignment |
| `load_rtstruct()` | DICOM RTSTRUCT contours filled onto a reference image, as one label image or masks by ROI name |
| `get_segment_info()` | Inspect available segments in a DICOM SEG, RTSTRUCT or `.seg.nrrd` file |
| `save_image()` | Save an image, mask or response map as NIfTI |
| `load_and_merge_images()` | Combine multiple images/masks with various strategies |
| `create_full_mask()` | Create an all-ones mask matching image geometry |
| `get_dicom_phases()` | Discover available phases in multi-phase DICOM |

## Next Steps

- [Pipeline & Preprocessing](pipeline.md) - Configure and run the radiomics pipeline
- [Cookbook](cookbook.md) - End-to-end batch processing scripts
