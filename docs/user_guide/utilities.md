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
# Export series summary to CSV
db.export_csv("series_summary.csv", level="series")

# Export full instance details to CSV
db.export_csv("instances.csv", level="instance")

# Export hierarchical JSON
db.export_json("dataset.json")
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
