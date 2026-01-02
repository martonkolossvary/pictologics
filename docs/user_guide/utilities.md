# Utilities

Pictologics provides a set of powerful utilities to help with data wrangling and organization, particularly for handling complex DICOM datasets.

## DICOM Database Parser

The `DicomDatabase` class allows you to easily parse, organize, and query DICOM folders. It automatically structures your data into a Patient -> Study -> Series -> Instance hierarchy and extracts relevant metadata.

### Basic Usage

To parse a directory of DICOM files:

```python
from pictologics.utilities import DicomDatabase

# 1. Parse a folder (recursive by default)
db = DicomDatabase.from_folders(
    paths=["path/to/dicom/folder"], 
    num_workers=4  # Use parallel processing for speed
)

# 2. Get a summary DataFrame of all series
df_series = db.get_series_df()
print(df_series.head())

# 3. Get detailed DataFrame of all instances
df_instances = db.get_instances_df()
```

### Memory-Efficient Exports

By default, DataFrame exports exclude the large `InstanceSOPUIDs` and `InstanceFilePaths` columns to reduce memory usage. To include them:

```python
# Default: smaller DataFrames without instance lists
df_compact = db.get_series_df()

# Include full instance lists if needed
df_full = db.get_series_df(include_instance_lists=True)
```

This applies to `get_patients_df()`, `get_studies_df()`, and `get_series_df()`.

### Multi-Phase Series Splitting

Medical images often contain multiple phases (e.g., dynamic contrast-enhanced MRI, multiphase CT, cardiac phases) within a single "series" (sharing the same `SeriesInstanceUID`). `DicomDatabase` automatically detects and splits these into separate logical series for easier analysis.

By default, `split_multiseries=True`.

```python
# Automatically splits series based on:
# - Cardiac Phase
# - Acquisition Number
# - Temporal Position
# - Echo Number
# - Trigger Time
# - Duplicate Spatial Positions (fallback)

db = DicomDatabase.from_folders(["path/to/multiphase/data"])

# The resulting DataFrames will show split series with unique identifiers
# e.g., "1.2.3.4.5" -> "1.2.3.4.5" (if single phase)
# e.g., "1.2.3.4.5" -> "1.2.3.4.5.1", "1.2.3.4.5.2" (if multi-phase)
series_df = db.get_series_df()
```

### Exporting Data

You can export the structured data to standard formats:

```python
# Export all levels to CSV (default: without instance lists for smaller files)
db.export_csv("output", levels=["patients", "studies", "series", "instances"])

# Export with full instance UIDs and file paths
db.export_csv("output", include_instance_lists=True)

# Export hierarchical JSON (includes file paths by default)
db.export_json("dataset.json")

# Export JSON without file paths for smaller files
db.export_json("dataset.json", include_instance_lists=False)
```

### Accessing the Hierarchy Directly

You can also traverse the object hierarchy directly if you need fine-grained control:

```python
for patient in db.patients:
    print(f"Patient: {patient.patient_id}")
    for study in patient.studies:
        print(f"  Study: {study.study_date}")
        for series in study.series:
            print(f"    Series: {series.modality} ({len(series.instances)} images)")
            
            # Access instances
            # series.instances is a list of DicomInstance objects
```

---

## Mask Visualization

Pictologics provides utilities to visualize mask overlays on medical images, helpful for verifying segmentation alignment.

### Interactive Viewer

Scroll through slices with the mask overlay:

```python
from pictologics import load_image
from pictologics.utilities import visualize_mask_overlay

img = load_image("scan.nii.gz")
mask = load_image("segmentation.nii.gz")

visualize_mask_overlay(img, mask, alpha=0.4, colormap="tab20")
```

### Save Overlay Slices

Export selected slices as images:

```python
from pictologics.utilities import save_mask_overlay_slices

# Save every 10th slice
save_mask_overlay_slices(img, mask, "output/", slice_selection="every_10")

# Save at 10% intervals
save_mask_overlay_slices(img, mask, "output/", slice_selection="10%")

# Save specific slices
save_mask_overlay_slices(img, mask, "output/", slice_selection=[0, 50, 100])
```

### Colormap Options

| Colormap | Labels | Description |
|----------|--------|-------------|
| `tab10` | 10 | Distinct categorical colors |
| `tab20` | 20 | Default, 20 distinct colors |
| `Set1` | 9 | Bold qualitative colors |
| `Set2` | 8 | Pastel qualitative colors |
| `Paired` | 12 | Paired colors |

### Output Formats

Supported formats: `png` (default, transparent), `jpeg`, `tiff`

```python
save_mask_overlay_slices(img, mask, "output/", format="tiff", dpi=300)
```

---

## Cropped Image Repositioning

Medical imaging software often stores segmentation masks as **cropped volumes** (bounding boxes around the region of interest) to minimize storage. When loading these cropped masks, they need to be repositioned into the original image's coordinate space for proper visualization and analysis.

### Why is this needed?

Cropped masks have smaller dimensions than the full image and are stored with their own spatial origin. The `pictologics` loader uses the spatial metadata (`ImagePositionPatient` for DICOM, affine matrix for NIfTI) to calculate where the cropped mask belongs in the full volume.

### Loading Cropped Masks

#### Single Image with Reference

```python
from pictologics import load_image

# Load the main image first
main_image = load_image("ct_scan/", recursive=True)

# Load a cropped mask and automatically reposition it
mask = load_image(
    "cropped_mask.dcm",
    reference_image=main_image  # Repositions to match main_image's space
)

# mask now has the same shape as main_image
print(f"Main image shape: {main_image.array.shape}")
print(f"Mask shape: {mask.array.shape}")  # Same as main_image
```

#### Merging Multiple Cropped Masks

When you have multiple cropped segmentation files (e.g., one per anatomical region), use `load_and_merge_images` with `reposition_to_reference=True`:

```python
from pictologics import load_image, load_and_merge_images
from pathlib import Path

# Load the main image
main_image = load_image("patient/ct_scan/", recursive=True)

# Find all segmentation files
seg_folder = Path("patient/segmentations/")
seg_files = [str(f) for f in seg_folder.glob("*.dcm")]

# Load and merge all cropped masks
merged_mask = load_and_merge_images(
    image_paths=seg_files,
    reference_image=main_image,
    reposition_to_reference=True,  # Enable cropped mask repositioning
    conflict_resolution="max",      # How to handle overlapping regions
)

# Now visualize
from pictologics.utilities import visualize_mask_overlay
visualize_mask_overlay(main_image, merged_mask)
```

### Handling Axis Transposition

Some imaging software may store masks with different axis orderings. Use the `transpose_axes` parameter to correct this:

```python
# If the mask has Y and Z axes swapped
merged_mask = load_and_merge_images(
    image_paths=seg_files,
    reference_image=main_image,
    reposition_to_reference=True,
    transpose_axes=(0, 2, 1),  # Swap Y and Z before repositioning
)
```

### Error Handling

| Issue | Behavior |
|-------|----------|
| Spacing mismatch | `ValueError` raised (resampling not yet supported) |
| Orientation mismatch | Warning emitted, positioning continues |
| Mask outside reference bounds | Warning emitted, empty volume returned |
| Partial overlap | Valid region is positioned, rest is clipped |

### Multi-Label Masks with `relabel_masks`

When loading multiple binary mask files (each containing only 0 and 1), you often want to distinguish them visually with different colors. The `relabel_masks` parameter automatically assigns unique label values (1, 2, 3, ...) to each file:

```python
from pictologics import load_image, load_and_merge_images
from pictologics.utilities import visualize_mask_overlay
from pathlib import Path

# Load the main CT scan
main_image = load_image("patient/ct_scan/", recursive=True)

# Find all AHA segment mask files (17 files)
seg_folder = Path("patient/aha_segments/")
seg_files = sorted(seg_folder.glob("*.dcm"))  # Sorted for consistent labeling

# Merge with unique labels per file
merged_mask = load_and_merge_images(
    image_paths=[str(f) for f in seg_files],
    reference_image=main_image,
    reposition_to_reference=True,
    relabel_masks=True,  # Each file gets a unique label: 1, 2, 3, ...
    conflict_resolution="max",
)

# Result: merged_mask.array contains values [0, 1, 2, 3, ..., 17]
# where 0 = background, 1 = first file, 2 = second file, etc.

# Visualize with different colors per segment
visualize_mask_overlay(main_image, merged_mask, colormap="tab20")
```

!!! tip "Label Order"
    Labels are assigned based on the order of files in `image_paths`. Use `sorted()` for consistent ordering, or specify the exact order you want.

**Without vs With `relabel_masks`:**

| Parameter | Resulting Values | Use Case |
|-----------|-----------------|----------|
| `relabel_masks=False` (default) | `[0, 1]` (binary) | Single combined ROI |
| `relabel_masks=True` | `[0, 1, 2, ..., N]` (multi-label) | Visualizing distinct regions |
