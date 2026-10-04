# Utilities

Pictologics has tools to sort DICOM data, to look at images and masks, and to read DICOM structured reports. They are in `pictologics.utilities`.

## DICOM Database

`DicomDatabase` reads DICOM folders and sorts them into patients, studies, series and instances, with their metadata.

```python
from pictologics.utilities import DicomDatabase

db = DicomDatabase.from_folders(paths=["dicom_archive/"], num_workers=4)  # recursive by default
series = db.get_series_df()       # one row for each series
instances = db.get_instances_df()  # one row for each file
print(series.head())
```

- **Workers**: each worker process imports Pictologics before it reads a file, so workers pay off only for large scans. `from_folders()` starts at most one worker per 1,000 DICOM files (`FILES_PER_WORKER`); a folder of a few series is read in the calling process, which is faster. `show_progress=False` turns the progress bar off.
- **Private tags**: `extract_private_tags=True` (default) keeps the vendor tags. A binary private value (for example a 60 KB CSA header) or a private sequence is stored as its size, such as `<OB, 60000 bytes>`.
- **Skipped files**: DICOMDIR files are not patient data, so they are skipped.
- **Completeness**: the series table tells whether a series is complete (`IsComplete`), whether its slices have gaps (`HasGaps`), and its slice spacing (`SpacingMM`).
- **Scouts**: the database lists every image, also a scout that `load_image()` leaves out of a series.

### Phases

`split_multiseries=True` (default) splits a series with more than one phase (for example the phases of a cardiac CT) into one logical series for each phase, as `load_image()` and `get_dicom_phases()` do. The rules are on the [Data Loading](data_loading.md#phases-of-a-dicom-series) page.

```python
db = DicomDatabase.from_folders(["cardiac_data/"])
series = db.get_series_df()  # "1.2.3.4.5" becomes "1.2.3.4.5.1", "1.2.3.4.5.2", ... for the phases
```

### Export

```python
db.export_csv("output")  # output_patients.csv, output_studies.csv, output_series.csv, output_instances.csv
db.export_csv("output", include_instance_lists=True)  # with the UIDs and paths of the instances
db.export_json("dataset.json")                         # the hierarchy, with the file paths
db.export_json("dataset.json", include_instance_lists=False)
```

The tables leave out the long `InstanceSOPUIDs` and `InstanceFilePaths` columns by default; `include_instance_lists=True` adds them. This applies to `get_patients_df()`, `get_studies_df()` and `get_series_df()`.

### The Hierarchy

```python
for patient in db.patients:
    print(f"Patient: {patient.patient_id}")
    for study in patient.studies:
        print(f"  Study: {study.study_date}")
        for one in study.series:
            print(f"    Series: {one.modality} ({len(one.instances)} images)")
```

## Image Viewers

`visualize_slices()` shows the slices in a window, and `save_slices()` saves them as files. Both show an image, a mask, or a mask on an image:

| Mode | `image` | `mask` | Display |
|:--|:--|:--|:--|
| Overlay | yes | yes | The mask in colour on the gray image |
| Image | yes | no | The gray image |
| Mask | no | yes | The mask in colour |

In overlay mode, `alpha` sets how much of the mask colour mixes into the gray image: 0 shows the image only, 1 the mask colour only, and 0.25 (default) keeps the image visible.

```python
from pictologics import load_image
from pictologics.utilities import save_slices, visualize_slices

image = load_image("scan.nii.gz")
mask = load_image("segmentation.nii.gz", reference_image=image)

visualize_slices(image=image, mask=mask, alpha=0.4, colormap="tab20")  # scroll through the slices
save_slices("qc/", image=image, mask=mask, slice_selection="10%")      # 10 % of the slices
save_slices("qc/", image=image, slice_selection="every_10")            # every 10th slice
save_slices("qc/", image=image, slice_selection=[0, 50, 100])          # these slices
save_slices("qc/", image=image, mask=mask, format="tiff", dpi=300)     # png (default), jpeg or tiff; dpi: the tag in the file
```

- **Gray scale**: without `window_center` and `window_width`, all slices share one gray scale: the minimum and the maximum of the volume (without NaN values). For CT, give a window: soft tissue 40 / 400, bone 400 / 1800, lung -600 / 1500.
- **Colormaps**: `tab20` (default, 20 colours), `tab10`, `Set1`, `Set2`, `Paired`.
- **Slices**: a single slice index outside the image raises a `ValueError`; in a list, indices outside the image are skipped.
- **Files**: `save_slices` writes the slices in threads (numba's thread count), as RGB files (the overlay is mixed into the colours). Each file has one pixel per voxel. `dpi` (default 300) is only the resolution tag in the file: viewers and printers use it to scale the image. PNG files use compression level 3, 2 times faster than the default level 6, for files 4 % larger.

### Quality Images of Many Cases

`save_slices` already uses threads, so a plain loop is often fast enough:

```python
from pathlib import Path

from pictologics import load_image
from pictologics.utilities import save_slices

cases = [
    ("patient_001/ct/", "patient_001/seg.dcm"),
    ("patient_002/ct/", "patient_002/seg.dcm"),
]
for image_path, mask_path in cases:
    image = load_image(image_path)
    mask = load_image(mask_path, reference_image=image)  # the mask on the grid of the image
    save_slices(Path("qc") / Path(image_path).parent.name, image=image, mask=mask,
                slice_selection="10%", window_center=40, window_width=400)
```

For many large cases, process pools help. Keep the guard: the workers start a new Python process, which imports your script again.

```python
from concurrent.futures import ProcessPoolExecutor


def one_case(case):
    image_path, mask_path, output_dir = case
    image = load_image(image_path)
    mask = load_image(mask_path, reference_image=image)
    return save_slices(output_dir, image=image, mask=mask, slice_selection="10%")


if __name__ == "__main__":
    jobs = [(image, mask, f"qc/case_{k}") for k, (image, mask) in enumerate(cases)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one_case, jobs))
```

Each worker holds one case at a time, so the memory need grows with `max_workers`.

## DICOM Structured Reports (SR)

`SRDocument` reads the measurements of a DICOM structured report (for example TID 1500 measurement reports).

```python
from pictologics.utilities import SRDocument

sr = SRDocument.from_file("measurements.dcm")
print(sr.template_id, len(sr.measurement_groups))

table = sr.get_measurements_df()
print(table[["measurement_name", "value", "unit", "finding_type", "finding_site", "tracking_id"]])
sr.export_csv("measurements.csv")
sr.export_json("measurements.json")
```

- Each measurement group is one TID 1500 container: its `group_id` is the tracking ID of the container, and the group holds its finding type, finding site and derivation.
- The measurement table has one row for each measurement, with the columns `group_id`, `finding_type`, `finding_site`, `derivation` and `tracking_id`.

### Many Reports

```python
batch = SRDocument.from_folders(
    paths=["dicom_data/"],      # recursive by default
    num_workers=4,
    output_dir="sr_exports/",   # writes <SOPInstanceUID>.csv and .json for each report
    export_csv=True,
    export_json=True,
)
print(f"Read {len(batch.documents)} reports")
combined = batch.get_combined_measurements_df()  # with group_finding_type and group_finding_site
batch.export_combined_csv("sr_exports/all_measurements.csv")
batch.export_log("sr_exports/processing_log.csv")
```

- `from_folders()` writes only the file of each report. `export_combined_csv()` and `export_log()` write the combined table and the log.
- Each worker imports Pictologics before it reads a file (about 1 s), so workers pay off only for large batches. `from_folders()` reads SR files of less than 2.5 MB in total (`SR_POOL_BYTES`) in the calling process. To find the SR files, it reads each file only up to its SOP Class UID.

The processing log has one row for each file:

| Column | Description |
|:--|:--|
| `file_path` | The SR file |
| `sop_instance_uid` | The SOP Instance UID |
| `patient_id` | The patient ID of the report |
| `study_instance_uid` | The Study Instance UID |
| `status` | `"success"` or `"error"` |
| `error_message` | The error, if any |
| `num_measurements` | The number of measurements |
| `csv_path`, `json_path` | The exported files |
| `processing_time_ms` | The time to read the file (ms) |
