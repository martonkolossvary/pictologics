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

### Parallel Processing and Progress Bar

For large DICOM datasets, parsing can take significant time. `DicomDatabase.from_folders()` supports **parallel processing** to speed up the scan and displays a **progress bar** showing progress and estimated time remaining.

```python
# Parallel processing with all available cores
db = DicomDatabase.from_folders(
    paths=["large_dataset/"],
    num_workers=8,          # Use 8 parallel workers
    show_progress=True      # Show progress bar (default: True)
)

# Disable progress bar for silent operation
db = DicomDatabase.from_folders(
    paths=["data/"],
    show_progress=False
)
```

The progress bar shows:
- Number of files processed
- Percentage complete
- Elapsed time and estimated time remaining

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

## Visualization

Pictologics provides flexible utilities for visualizing medical images and segmentation masks. The visualization functions support three display modes:

| Mode | `image` | `mask` | Description |
|------|---------|--------|-------------|
| **Overlay** | ✓ | ✓ | Mask overlaid on grayscale image |
| **Image Only** | ✓ | ✗ | Grayscale image (with optional window/level) |
| **Mask Only** | ✗ | ✓ | Colormap or grayscale mask display |

### Interactive Viewer

Scroll through slices interactively:

```python
from pictologics import load_image
from pictologics.utilities import visualize_slices

img = load_image("scan.nii.gz")
mask = load_image("segmentation.nii.gz")

# Overlay mode (image + mask)
visualize_slices(image=img, mask=mask, alpha=0.4, colormap="tab20")

# Image only mode
visualize_slices(image=img)

# Mask only mode (with colormap)
visualize_slices(mask=mask)
```

### Save Slices to Files

Export selected slices as images:

```python
from pictologics.utilities import save_slices

# Save overlay slices
save_slices("output/", image=img, mask=mask, slice_selection="10%")

# Save every 10th slice
save_slices("output/", image=img, mask=mask, slice_selection="every_10")

# Save specific slices
save_slices("output/", image=img, slice_selection=[0, 50, 100])
```

### Window/Level Normalization

For CT and MR images, use window/level controls for proper contrast:

```python
# Soft tissue window (default: center=200, width=600)
visualize_slices(image=img, window_center=40, window_width=400)

# Bone window
visualize_slices(image=img, window_center=400, window_width=1800)

# Lung window
visualize_slices(image=img, window_center=-600, window_width=1500)
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

Supported formats: `png` (default), `jpeg`, `tiff`

```python
save_slices("output/", image=img, mask=mask, format="tiff", dpi=300)
```

### Parallel Batch Processing

For processing multiple images efficiently, use `concurrent.futures`:

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pictologics import load_image
from pictologics.utilities import save_slices

def process_case(args):
    """Process a single image/mask pair."""
    image_path, mask_path, output_dir = args
    
    # Load images
    img = load_image(image_path, recursive=True)
    mask = load_image(mask_path, recursive=True)
    
    # Save slices
    return save_slices(
        output_dir,
        image=img,
        mask=mask,
        slice_selection="10%",
        window_center=40,
        window_width=400
    )

# Prepare list of (image, mask, output) tuples
cases = [
    ("patient_001/ct/", "patient_001/seg.dcm", "output/patient_001/"),
    ("patient_002/ct/", "patient_002/seg.dcm", "output/patient_002/"),
    ("patient_003/ct/", "patient_003/seg.dcm", "output/patient_003/"),
]

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_case, cases))
    
print(f"Processed {len(results)} cases")
```

!!! note "Performance Notes"
    - Use `ProcessPoolExecutor` (not `ThreadPoolExecutor`) to avoid Python's GIL
    - Set `max_workers` to the number of CPU cores (4-8 is typically optimal)
    - Each worker loads one image at a time, so memory usage scales with `max_workers`

## DICOM Structured Reports (SR)

Parse DICOM Structured Reports to extract measurements and tabular data.

### Loading and Parsing SR

```python
from pictologics.utilities import SRDocument

sr = SRDocument.from_file("measurements.dcm")
print(f"Template: {sr.template_id}")
print(f"Groups: {len(sr.measurement_groups)}")
```

### Extracting Measurements

```python
# Get as DataFrame
df = sr.get_measurements_df()
print(df[["measurement_name", "value", "unit"]])

# Export to files
sr.export_csv("measurements.csv")
sr.export_json("measurements.json")
```

### Batch SR Processing

For processing multiple SR files from folders, use `SRDocument.from_folders()`:

```python
from pictologics.utilities import SRDocument

# Process all SR files in a folder (recursive by default)
batch = SRDocument.from_folders(
    paths=["dicom_data/"],
    num_workers=4,  # Parallel processing
    output_dir="sr_exports/",  # Auto-export each SR
    export_csv=True,
    export_json=True,
)

# Access results
print(f"Processed {len(batch.documents)} SR files")

# Get combined measurements from all SRs
df = batch.get_combined_measurements_df()
print(df.head())

# Export combined data
batch.export_combined_csv("sr_exports/all_measurements.csv")
batch.export_log("sr_exports/processing_log.csv")
```

### Parallel Processing and Progress Bar

For large collections of SR files, `SRDocument.from_folders()` supports **parallel processing** for faster parsing and displays a **progress bar** showing progress and estimated time remaining.

```python
# Parallel processing with 8 workers
batch = SRDocument.from_folders(
    paths=["large_dataset/"],
    num_workers=8,          # Use 8 parallel workers
    show_progress=True      # Show progress bar (default: True)
)

# Disable progress bar for silent operation
batch = SRDocument.from_folders(
    paths=["data/"],
    show_progress=False
)
```

The progress bar shows:
- Number of SR files processed
- Percentage complete
- Elapsed time and estimated time remaining

### Output Structure

When `output_dir` is specified:
```
sr_exports/
├── all_measurements.csv       # Combined measurements (optional)
├── processing_log.csv         # Log of all processed files
├── 1_2_3_4_5_6.csv           # Individual SR (if export_csv=True)
├── 1_2_3_4_5_6.json          # Individual SR (if export_json=True)
└── ...
```

### Processing Log

The processing log tracks each file:

| Column | Description |
|--------|-------------|
| file_path | Source SR file path |
| sop_instance_uid | SOP Instance UID |
| status | "success" or "error" |
| num_measurements | Count of measurements |
| csv_path | Path to exported CSV |
| json_path | Path to exported JSON |
| error_message | Error details if failed |

