# Cardiac CT Phases

A cardiac CT folder often holds many phases of the heart cycle in one DICOM series. This tutorial finds the phases, loads one, and runs the pipeline on each phase.

## 1. Find the Phases

```python
from pictologics.utilities import get_dicom_phases

phases = get_dicom_phases("cardiac_ct/")
for phase in phases:
    print(phase.index, phase.label, phase.num_slices)
# 0 Phase 0% 160
# 1 Phase 10% 160
# ...
```

`get_dicom_phases` splits the series by the first of these tags that changes: the cardiac phase percentage, the temporal position, the trigger time, the acquisition number, the echo number; else by repeated slice positions. A tag splits the series only at repeated slice positions, so a scanner that writes a new acquisition number every few slices gives one phase.

## 2. Load One Phase

```python
from pictologics import load_image

image = load_image("cardiac_ct/", dataset_index=4)  # the 5th phase
image = load_image(phases[4])                        # the same, without reading the folder again
```

A folder with more than one image series (for example two reconstructions) needs `series_uid`; the error lists the series.

## 3. Run Each Phase

For an image path, `image_options` gives options to `load_image`. So the pipeline loads each phase from the folder path:

```python
from pictologics import RadiomicsPipeline, format_results, save_results

pipeline = RadiomicsPipeline.from_template("lv")
rows = []
for phase in phases:
    results = pipeline.run(
        "cardiac_ct/",
        f"masks/lv_phase{phase.index}.nii.gz",
        image_options={"dataset_index": phase.index},
        config_names=["lv_fbn_32"],
    )
    rows.append(format_results(results, meta={"phase": phase.label}))
save_results(rows, "lv_by_phase.csv")
```

The log entry of each configuration records the `image_options`. In `run_batch`, give each phase as its own case with its `image_options` (see [Many ROIs and Batch Studies](batch.md)).

!!! tip "Masks from other programs"
    A NIfTI mask on the grid of the DICOM image loads also when its voxel order differs (axes flipped or swapped, as some converters write). The loader turns the mask to the voxel order of the image. Every image is in the LPS+ frame, as in DICOM.

## 4. The Cardiac Templates

Two templates hold configurations for cardiac CT:

| Template | Structure | Configurations |
|:--|:--|:--|
| `lv` | The left ventricular myocardium, in four compartments: whole, fat, myocardial tissue and calcium | 30 |
| `coronary` | Coronary plaque, in four plaque types: all, non-calcified, low-attenuation and calcified | 30 |

```python
pipeline = RadiomicsPipeline.from_template("coronary")
print(pipeline.list_configs())
```

Look at the steps of a configuration with `pipeline.get_config(name)`, and change them for your study.

See also: [Data Loading](../user_guide/data_loading.md#phases-of-a-dicom-series) and [Rings Around an ROI](rings.md) (the fat around a vessel).
