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
