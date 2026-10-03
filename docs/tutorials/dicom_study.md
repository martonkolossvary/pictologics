# From a DICOM Archive to a Study Table

A study often starts with a folder of DICOM files from a PACS: many patients, many series, and segmentations. This tutorial finds the right series, runs the pipeline on each patient, and makes one table for the statistics, with the record that a paper needs.

## 1. Index the Archive

```python
from pictologics.utilities import DicomDatabase

db = DicomDatabase.from_folders(["archive/"], num_workers=4)
db.export_csv("study/archive")  # study/archive_series.csv and the other tables
series = db.get_series_df(include_instance_lists=True)
print(series[["PatientID", "Modality", "SeriesDescription", "NumInstances", "IsComplete", "HasGaps", "SpacingMM"]])
```

The series table has one row for each series, with its patient, study, modality, the number of files, the slice spacing, and whether slices are missing (`IsComplete`, `HasGaps`). `include_instance_lists=True` adds the paths of the files of each series (`InstanceFilePaths`).

## 2. Choose the Series

Keep the CT series that are complete and thin, and pair each with the SEG of the same patient and the same frame of reference:

```python
ct = series[(series["Modality"] == "CT") & series["IsComplete"] & ~series["HasGaps"] & (series["SpacingMM"] <= 1.5)]
seg = series[series["Modality"] == "SEG"]
pairs = ct.merge(seg, on=["PatientID", "FrameOfReferenceUID"], suffixes=("", "_seg"))
pairs = pairs.sort_values("SpacingMM").drop_duplicates("PatientID")  # the thinnest series of each patient
print(f"{len(pairs)} of {series['PatientID'].nunique()} patients have a CT with a SEG")
```

## 3. Make the Cases

A case gives the folder of the series, and `series_uid` chooses the series in that folder:

```python
import os

cases = [
    {
        "subject_id": row.PatientID,
        "image": os.path.commonpath(row.InstanceFilePaths),  # the folder of the series files
        "image_options": {"series_uid": row.SeriesInstanceUID, "recursive": True},
        "mask": row.InstanceFilePaths_seg[0],  # the SEG file
    }
    for row in pairs.itertuples()
]
```

## 4. The Whole Script

The workers of `run_batch` import the script again, so the script keeps all its work in `main()`:

```python
import json
import os
from pathlib import Path

import pandas as pd

from pictologics import RadiomicsPipeline, save_results
from pictologics.utilities import DicomDatabase


def study_cases(archive):
    db = DicomDatabase.from_folders([archive], num_workers=4)
    db.export_csv("study/archive")
    series = db.get_series_df(include_instance_lists=True)
    ct = series[(series["Modality"] == "CT") & series["IsComplete"] & ~series["HasGaps"] & (series["SpacingMM"] <= 1.5)]
    seg = series[series["Modality"] == "SEG"]
    pairs = ct.merge(seg, on=["PatientID", "FrameOfReferenceUID"], suffixes=("", "_seg"))
    pairs = pairs.sort_values("SpacingMM").drop_duplicates("PatientID")
    return [
        {
            "subject_id": row.PatientID,
            "image": os.path.commonpath(row.InstanceFilePaths),
            "image_options": {"series_uid": row.SeriesInstanceUID, "recursive": True},
            "mask": row.InstanceFilePaths_seg[0],
        }
        for row in pairs.itertuples()
    ]


def main():
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("ct_fbs_25", [
        {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
        {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
        {"step": "discretise", "params": {"method": "FBS", "bin_width": 25}},
        {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
    ])
    table = pipeline.run_batch(study_cases("archive/"), "study/results", workers=4)
    print(table["status"].value_counts())

    # 5. Join the clinical data
    clinical = pd.read_csv("clinical.csv")  # PatientID, age, outcome, ...
    study = clinical.merge(table, left_on="PatientID", right_on="subject_id", how="inner")
    save_results(study, "study/study_table.csv")

    # 6. Keep the record
    pipeline.save_configs("study/configs.yaml")
    pipeline.describe_features().to_csv("study/feature_catalog.csv", index=False)
    first = next(Path("study/results/cases").glob("*.json"))
    record = json.loads(first.read_text())
    print(record["config_hashes"], record["log"][0]["pictologics_version"])


if __name__ == "__main__":
    main()
```

## 5. Check the Table

- **Status**: `table["status"]` is `"completed"` for a case without problems. For an `"incomplete"` or a `"failed"` case, the `error` column tells the cause. Correct the input, delete the file of the case in `study/results/cases/`, and run the script again: it runs only the cases without a file.
- **Warnings**: the `warnings` column holds the warnings of each case, for example a mask offset or a frame of reference that does not match.
- **Quality images**: look at some masks on their images before the statistics (see [Utilities](../user_guide/utilities.md#quality-images-of-many-cases)).

## 6. The Record of the Study

Keep these files with the study table:

| File | Holds |
|:--|:--|
| `study/archive_series.csv` (and the other tables) | The index of the archive: the series that the study saw |
| `study/configs.yaml` | The configurations, to load again with `RadiomicsPipeline.load_configs` |
| `study/feature_catalog.csv` | The data dictionary: each feature with its IBSI code and preprocessing |
| `study/results/cases/*.json` | The features, the warnings and the processing log of each case, with the `config_hashes` and the versions |
| `study/study_table.csv` | The table for the statistics |

Report the Pictologics version, the configuration file and the `config_hash` of each configuration. See [What to Report](../user_guide/results.md#what-to-report).
