# Many ROIs and Batch Studies

This tutorial runs one study: many cases, and many ROIs in each case. It uses `run_rois` for the ROIs of one image and `run_batch` for the cases.

## 1. Many ROIs of One Image

A label map holds all ROIs of an image in one file: 0 for the background, and the label of its ROI in each other voxel (for example an organ segmentation, a DICOM SEG loaded with `combine_segments=True`, or an RTSTRUCT). `run_rois` gives each ROI the results of `run()` with a mask of that label alone, with one image load.

```python
from pictologics import RadiomicsPipeline, format_results, save_results

pipeline = RadiomicsPipeline()
results = pipeline.run_rois(
    "ct.nii.gz",
    "organs.nii.gz",
    labels={"liver": 5, "spleen": 1},  # default: every label of the map
    subject_id="p001",
    config_names=["standard_fbn_32"],
)
rows = [
    format_results(series, meta={"subject_id": "p001", "roi": roi})
    for roi, series in results.items()
]
save_results(rows, "p001_rois.csv")
```

`run_rois` is much faster than one `run()` for each label: it loads the image once, checks it once, and makes each mask only in the box of its label.

## 2. Many Cases with `run_batch`

`run_batch` runs the configurations on a list of cases. It writes one result file for each case, so a stopped batch goes on where it stopped.

```python
from pictologics import RadiomicsPipeline, save_results

if __name__ == "__main__":  # the worker processes import this script again
    pipeline = RadiomicsPipeline.from_template("coronary")
    cases = [
        {"subject_id": "p001", "image": "data/p001/ccta.nii.gz", "mask": "data/p001/plaque.nii.gz"},
        {"subject_id": "p002", "image": "data/p002/ccta.nii.gz", "mask": "data/p002/plaque.nii.gz"},
    ]
    table = pipeline.run_batch(cases, "results", workers=4)
    print(table[["subject_id", "status", "error"]])
    save_results(table, "results/features.csv")
```

- **Cases**: a list of dicts, or a pandas DataFrame with one row for each case. A case holds the `run()` arguments of one image: `subject_id` and `image` (required), `mask`, the mask settings, and `image_options`.
- **Result files**: `results/cases/<subject_id>.json` holds the status, the error, the warnings, the time, the features and the processing log of a case.
- **Resume**: a later call with the same folder skips each case whose file has the same image, image options, mask and configurations. A failed case runs again. To run a case again, delete its file.
- **Workers**: with `workers=4`, four processes run the cases, and each one uses a quarter of the numba threads. Each worker holds one case at a time, so the memory need grows with the number of workers.
- **Status**: the returned table has one row for each case: `"completed"`, `"incomplete"` (a configuration ended with an empty ROI or an error) or `"failed"` (the case did not run, for example because its image did not load).

## 3. Label Maps in a Batch

A case can give a label map (`rois`, and optionally `labels`) in place of `mask`. The case then runs `run_rois`, and each of its ROIs gets a row of the table, with its name in the column `roi`.

```python
cases = [
    {"subject_id": "p001", "image": "data/p001/ct", "rois": "data/p001/aha_segments.nii.gz",
     "labels": {f"segment_{k}": k for k in range(1, 18)}},
]
table = pipeline.run_batch(cases, "results_lv", workers=2)
```

## 4. Options of the Image

`image_options` gives options to `load_image` for an image path: for example a phase of a multi-phase DICOM folder, or a PET series as SUV.

```python
cases = [
    {"subject_id": f"p001_phase{k}", "image": "data/p001/cardiac_ct", "mask": f"data/p001/lv_phase{k}.nii.gz",
     "image_options": {"dataset_index": k}}
    for k in range(10)
]
```

The log entry of each configuration records the options, and the resume compares them too.

## 5. Check the Study

- The file of each case holds its processing log: one entry for each configuration that ran, with the status, the error and the failed step, the steps, the time, the `config_hash` and the `environment` (the versions of Python and the packages, and the thread count). After `run()` and `run_rois()`, `pipeline.get_log()` gives the same entries.
- A `config_hash` is the same for the same configuration on every computer. Report it with your results.
- `pipeline.describe_features()` gives one row for each feature of each configuration, with its preprocessing (the steps and their parameters). Save it with your results as a data dictionary.

## Tips

- Keep `if __name__ == "__main__":` around a `run_batch` call with workers: the workers start with spawn on every platform.
- Workers help because some steps of a case use one thread. On 28 CT cases (512 × 512 × 200) with a 1 mm configuration, 1 process with 14 threads did 3.1 cases per second, and 4 processes with 3 threads each did 6.6. Each worker needs the memory of one case.
- A case with an error does not stop the batch. Look at the `error` column, fix the input, delete the file of the case, and run the batch again.

See also: [Rings Around an ROI](rings.md) (rings of touching ROIs) and [Configuration & Reproducibility](../user_guide/configurations.md).
