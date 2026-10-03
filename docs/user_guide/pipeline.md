# The Pipeline

`RadiomicsPipeline` runs configurations on an image and a mask. A configuration is a named list of steps: the first steps prepare the image and the masks, and the `extract_features` step computes the features. Each configuration gives one `pandas.Series` of features.

| Page | Contents |
|:--|:--|
| This page | How to run the pipeline: configurations, masks, image options, source modes, many ROIs and many cases |
| [Pipeline Steps](pipeline_steps.md) | Each step and its parameters |
| [Results and Logs](results.md) | The result tables, the feature catalog and the processing log |
| [Configuration & Reproducibility](configurations.md) | Save, share and load configurations |

## A First Run

```python
from pictologics import RadiomicsPipeline, format_results, save_results

pipeline = RadiomicsPipeline()
results = pipeline.run(
    image="ct.nii.gz",
    mask="lesion.nii.gz",
    subject_id="p001",
    config_names=["standard_fbs_16"],
)
row = format_results(results, meta={"subject_id": "p001"})
save_results([row], "features.csv")
```

- `results` is a dictionary: one feature Series for each configuration.
- `format_results` makes one row, with a column for each feature, for example `standard_fbs_16__mean_intensity_Q4LE`.
- `subject_id` goes into the processing log only. To put it in the table, give it to `format_results` in `meta`.

## Which Configurations Run

- `RadiomicsPipeline()` starts with the six standard configurations. `RadiomicsPipeline(load_standard=False)` starts with none.
- `config_names` sets the configurations to run, in its order. A name that is in the list two times runs one time.
- `"all_standard"` in `config_names` stands for the six standard configurations.
- Without `config_names`, `run()` runs every configuration of the pipeline. When the standard configurations are among them, a warning tells you so.
- An unknown name raises a `ValueError` with the closest name.

### The Standard Configurations

The six standard configurations resample to 0.5 mm cubic voxels (linear interpolation). They compute the intensity, morphology, texture, histogram and IVH features.

| Configuration | Discretisation |
|:--|:--|
| `standard_fbn_8`, `standard_fbn_16`, `standard_fbn_32` | 8, 16 or 32 bins (FBN) |
| `standard_fbs_8`, `standard_fbs_16`, `standard_fbs_32` | Bins of 8, 16 or 32 HU from -1000 HU (FBS) |

The FBS bins start at -1000 HU in every image, so these three configurations are for CT. For MR and PET, use the FBN configurations or your own.

### Templates

A template is a file of configurations in the package. `RadiomicsPipeline.from_template(name)` makes a pipeline with the configurations of the template only.

| Template | Configurations |
|:--|:--|
| `standard` | The six standard configurations |
| `lv` | 30, for CT of the left ventricular myocardium, in four compartments: whole, fat, myocardial tissue and calcium |
| `coronary` | 30, for coronary plaque in CT angiography, in four plaque types: all, non-calcified, low-attenuation and calcified |

```python
pipeline = RadiomicsPipeline.from_template("coronary")
print(pipeline.list_configs())                    # coronary_orig, coronary_fbn_16, ...
print(pipeline.get_config("coronary_cp_fbs_16"))  # the steps of one configuration
pipeline.merge_configs(RadiomicsPipeline.from_template("lv"))  # add a second template
```

Each compartment of a cardiac template has an `orig` configuration (intensity and morphology features) and FBN and FBS configurations (texture, histogram and IVH features). The configurations use `source_mode="auto"` with the sentinel value -3024 HU. See [Cardiac CT Phases](../tutorials/cardiac_phases.md).

## Your Own Configuration

```python
pipeline = RadiomicsPipeline(load_standard=False)
pipeline.add_config("ct_fbs_25", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 25}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
])
results = pipeline.run("ct.nii.gz", "lesion.nii.gz", config_names=["ct_fbs_25"])
```

`add_config` checks the configuration before it keeps it:

- the step names, and the parameter names of each step and of each filter type;
- the parameters that a step or a filter needs, for example `new_spacing`, or the `sigma_mm` of a LoG filter;
- the values: the spacing, the discretise method and its bins, the filter boundary and padding, the wavelet level, the texture distances, the grow distances and the normalise settings;
- the feature family names;
- the step order rules (see [Step Order](#step-order)).

A mistake raises one `ValueError` that lists every problem, with the closest valid name:

```text
ValueError: Configuration 'ct' has 2 problem(s):
  - step 0 (resample): unknown parameter 'spacing' (did you mean 'new_spacing'?)
  - step 0 (resample): missing parameter 'new_spacing'
```

- `add_config(..., validate=False)` checks only the structure: a list of dictionaries, each with a `step` key. A mistake then shows at run time: the log entry gets the error, and the features are `NaN`.
- `add_config` keeps a copy of the steps. A later change to your list does not change the configuration.
- The pipeline does not check a configuration from a file (`load_configs`, `from_yaml`, `from_json`, `from_dict`) by default. With `validate=True`, each problem gives a `UserWarning`, and the configuration loads all the same.

## Step Order

The steps run one after another, in the order of the list. A step can go at any position, and a step can repeat. These rules apply:

1. The texture families need an earlier `discretise` step. Without it, `add_config` raises an error.
2. A `normalise` step must come before the `discretise` step.
3. FBS needs a start that is the same in every image: `min_val`, or a `resegment` step with `range_min` before it. A `filter` or `normalise` step after that `resegment` step cancels its start, because the units change. Without a start, `add_config` raises an error.
4. The histogram features read the discretised image. Without a `discretise` step, they give a warning.
5. A `resegment` step reads the image as it is at that position. Put it before a `filter` step to select by HU, and after a `normalise` step to select by the normalised values.

### Two Masks

The pipeline keeps two masks:

- The **morphological mask** gives the shape features.
- The **intensity mask** gives the voxels of all other features.

At the start, both masks are the ROI. A mask step changes both masks by default. With `apply_to` `"morph"` or `"intensity"`, it changes one of them. For example, this step selects the voxels of the intensity features, and the shape features keep the whole ROI:

```python
{"step": "resegment", "params": {"range_min": -100, "range_max": 200, "apply_to": "intensity"}}
```

### The Image of Each Feature Family

After a `discretise` step, the pipeline keeps two images: the image before the discretisation, and the discretised image. After a `filter` step, the image is the response map.

| Feature family | Reads |
|:--|:--|
| Intensity, spatial intensity, local intensity | The image before the discretisation |
| Morphology | The morphological mask. The integrated intensity (99N0) and the centre of mass shift (KLMA) also read the image before the discretisation |
| Histogram, texture | The discretised image |
| IVH | The discretised image; with `ivh_use_continuous`, the image before the discretisation |

## Masks

- **A mask path**: the pipeline loads the mask with the image as `reference_image`. So a DICOM SEG, an RTSTRUCT, a cropped mask or a mask in another voxel order goes onto the grid of the image (see [Data Loading](data_loading.md#masks)). The settings `mask_subvoxel_tolerance`, `mask_subvoxel_warning_threshold` and `mask_min_overlap_fraction` control the placement, as in `load_image`.
- **An `Image` mask**: it must have the grid of the image: the same shape, spacing, origin and direction. Else `run()` raises a `ValueError`.
- **No mask**: without `mask` (or with `mask=None`), the whole image is the ROI. The morphology features then describe the ROI after the mask steps, for example after `resegment`.
- **Labels**: each voxel that is not 0 is in the ROI. So the labels 1, 2 and 3 make one ROI, and a label is not a weight. To use one label, add a `binarize_mask` step with `mask_values`, or use [`run_rois`](#many-rois-run_rois).
- **Probability masks**: a float mask with values between whole numbers (for example a probability map) gives a warning, because each voxel above 0 counts as ROI. Add a `binarize_mask` step with a `threshold`, for example 0.5.
- **NaN voxels**: an ROI voxel with a NaN or infinite intensity leaves the intensity mask before the features, with a warning. When no ROI voxel has a finite intensity, the configuration ends with an empty ROI.

## Image Options

For an image path, `image_options` gives options to `load_image`: for example a phase of a DICOM folder, a series, or a PET series as SUV. The log entry of each configuration records them. `run_rois()` and the cases of `run_batch()` take them too.

```python
lv = RadiomicsPipeline.from_template("lv")
results = lv.run("cardiac_ct/", "lv_mask.nii.gz", image_options={"dataset_index": 4}, config_names=["lv_orig"])

pipeline = RadiomicsPipeline()
results = pipeline.run("pet_series/", "lesion.nii.gz", image_options={"suv": "bw"}, config_names=["standard_fbn_32"])
```

An `Image` takes no options: `image_options` with an `Image` raises a `ValueError`. An option that `load_image` does not know raises a `TypeError`.

## Source Modes and Sentinel Values

Some images hold a **sentinel value** where they have no data: for example -2048 HU outside the field of view of a CT, or -3024 HU (a stored -2000 with a rescale intercept of -1024). These voxels must stay out of the resampling and the filters, because an interpolation or a filter near them mixes the sentinel into the voxels with data. The `source_mode` of a configuration tells the pipeline which voxels hold image data.

| `source_mode` | The voxels with image data |
|:--|:--|
| `"full_image"` (default) | All voxels |
| `"roi_only"` | The voxels of the ROI. `sentinel_value` is not used |
| `"auto"` | The voxels without the sentinel value: the `sentinel_value` that you give, else a value that the pipeline finds, with a warning |

```python
pipeline.add_config("padded_ct", steps, source_mode="auto", sentinel_value=-2048)
```

- **A known padding value**: use `"auto"` with `sentinel_value`. This is the safest choice.
- **An unknown padding value**: `"auto"` alone looks for -2048, -3024, -1024, -1000, 0 and -32768. A value is a sentinel when it fills at least 5 % of the image, and when it is more than 2 times as frequent outside the ROI as inside. A warning gives the value that the pipeline found, or tells that it found none: the pipeline then uses all voxels. Check the warning, because a real tissue value can pass the test.
- **`"roi_only"`**: for an image with data in the ROI only, for example an export of a lesion. The voxels outside the ROI do not go into the resampling or the filters.
- **The effect**: the resampling and the mean, Gaussian, LoG and Laws filters leave out the voxels without data (normalized interpolation and convolution); the other filters fill them with 0 (see [Source Masks for Sentinel Values](image_filtering.md#source-masks-for-sentinel-values)). After a `resample` step and after a `grow_mask` step, the masks also lose the voxels without data. With region `"image"`, a `normalise` step reads only the voxels with data.

!!! warning "No mask and a padded image"
    Without a mask, the whole image is the ROI. When the `resegment` range holds the padding value, the background stays in the ROI, and the texture features can need much memory. Set a `resegment` range that leaves out the padding value, or use `source_mode="auto"` with the `sentinel_value` of the padding and a `resample` step: the resampling takes the voxels without data out of the masks.

## Many ROIs: `run_rois`

`run_rois()` runs the configurations on each ROI of a label map, with one image load. In a label map, each voxel holds the label of its ROI, and 0 for the background (for example an organ segmentation, a DICOM SEG loaded with `combine_segments=True`, or an RTSTRUCT).

```python
results = pipeline.run_rois(
    "ct.nii.gz",
    "organs.nii.gz",
    labels={"liver": 5, "spleen": 1},  # default: every label of the map
    subject_id="p001",
    config_names=["ct_fbs_25"],
)
rows = [format_results(series, meta={"subject_id": "p001", "roi": roi}) for roi, series in results.items()]
save_results(rows, "p001_rois.csv")
```

- Each ROI gets the results of `run()` with a mask of its label alone, bit for bit.
- `run_rois` loads the image once, checks it for NaN values once, and makes each mask only inside the box of its label. On a CT of 512 × 512 × 200 voxels with 100 labels, one `run()` for each label took 1.81 s with a 1 mm texture configuration, and `run_rois` took 0.92 s. With intensity features alone, the times were 1.21 s and 0.29 s.
- The result is a dictionary from ROI names to the results of `run()`. The name of an ROI is its name in `labels`, else its label as text, for example `"3"`. The log entries of an ROI hold its name in `roi`.
- A [`grow_mask`](pipeline_steps.md#grow_mask) step with `nearest_roi` gives each added voxel to its nearest ROI, so the rings of touching ROIs do not overlap.
- A label map cannot hold overlapping ROIs. For segments that overlap, load each segment as its own mask (`load_seg(..., combine_segments=False)`), and call `run()` for each.

## Many Cases: `run_batch`

`run_batch()` runs the configurations on many cases, in more than one process when you ask for it. It writes the result of each case to its own file when the case ends, so a stopped batch can go on later.

```python
from pictologics import RadiomicsPipeline, save_results

if __name__ == "__main__":  # the worker processes import this script again
    pipeline = RadiomicsPipeline.from_template("coronary")
    cases = [
        {"subject_id": "p001", "image": "p001/ccta.nii.gz", "mask": "p001/plaque.nii.gz"},
        {"subject_id": "p002", "image": "p002/ccta.nii.gz", "mask": "p002/plaque.nii.gz"},
    ]
    table = pipeline.run_batch(cases, "results", workers=4)
    print(table.loc[table["status"] != "completed", ["subject_id", "status", "error"]])
    save_results(table, "results/features.csv")
```

- **Cases**: a list of dicts, or a DataFrame with one row for each case. A case holds the `run()` arguments of one image: `subject_id` and `image` (required), `mask`, the mask settings and `image_options`. A case with a label map gives `rois` (and optionally `labels`) in place of `mask`, and gets one row for each ROI.
- **Result files**: the result of a case goes to `results/cases/<subject_id>.json`, with its status, error, warnings, run time, features and processing log.
- **Resume**: a second call with the same folder skips each case whose file holds the same image, image options, mask (or label map and labels) and configurations (by their `config_hash`). A failed case runs again. To run a case again, delete its file.
- **Workers**: with `workers=4`, four processes run the cases, and each process uses a quarter of the numba threads. Each process holds one case at a time, so the memory need grows with the number of workers. Keep the call inside `if __name__ == "__main__":`, because the workers start with spawn on every platform.
- **The table**: one row for each case (one for each ROI of a label map case), in the order of the cases: `subject_id`, `status`, `error`, `warnings`, `seconds` and the features in the wide format of `format_results()`. The status is `"completed"`; `"incomplete"` when a configuration ended with an empty ROI or an error; or `"failed"` when the case did not run, for example because its image did not load.
- **Errors**: an error of one case does not stop the batch. The warnings of a case go to its `warnings` column. The sentinel and NaN warnings also go to the `logging` module (see [Warnings](#warnings)).
- **Mistakes in the cases**: a case without `subject_id` or `image`, with an unknown key, with both `mask` and `rois`, or with the file name of another case raises a `ValueError` before the batch starts.

The [Many ROIs and Batch Studies](../tutorials/batch.md) tutorial shows a full study.

## Result Guarantees

Each configuration of a run gives a Series with all feature names of the configuration, also when a step or a feature fails:

| Failure | Example | Result |
|:--|:--|:--|
| Empty ROI | A `resegment` range that keeps no voxel | All features are `NaN`. The log status is `"empty_roi"` |
| One feature | No surface mesh, or a PCA of 3 voxels or fewer | That feature is `NaN`. The other features keep their values |
| One feature family | An error in the texture features | The features of that family are `NaN`. A warning names the family, and the log entry lists it in `family_errors` |
| One step | An error in a preprocessing step | All features are `NaN`. The log status is `"error"`, with the error and the failed step |

- The other configurations of the run go on.
- The rows of many cases have the same columns, so `save_results` merges them without gaps.
- A mistake in the arguments of `run()` (an unknown configuration, a mask on another grid, an image that does not load) raises an error, and no configuration runs.

## Shared Work Between Configurations

Configurations often start with the same steps. The pipeline does this shared work one time:

- **Shared steps**: when configurations start with the same steps (with the same parameters, the same source mode and the same sentinel value), the steps run one time, and each configuration goes on from their result.
- **Shared feature families**: with `deduplicate=True` (default), a feature family that gets the same input in two configurations is computed one time and copied. For example, the morphology and intensity features of configurations that differ only in a last `discretise` step. Each family has its own rule: the texture, histogram and IVH families read the discretisation, and each family reads only its own options of `extract_features`.

```python
results = pipeline.run(image, mask, config_names=["all_standard"])
print(pipeline.deduplication_stats)  # reused_families, computed_families, cache_hit_rate
```

- The results are the same with and without the shared work.
- The rules have a version, `"1.1.0"` by default. A new version of Pictologics keeps the old versions, so `RadiomicsPipeline(deduplication_rules="1.1.0")` gives the same reuse in later versions.
- See the [Deduplication API](../api/deduplication.md) for the rules of each family.

## Warnings

The pipeline tells you about a problem with a `UserWarning`. A warning does not stop the run.

| Warning | Cause |
|:--|:--|
| `run() without config_names runs all ... configurations` | `run()` without `config_names` on a pipeline with the standard configurations |
| `The mask holds values that are not whole numbers` | A probability mask without a `binarize_mask` step |
| `Left out ... ROI voxels with a NaN or infinite intensity` | NaN or infinite voxels in the ROI |
| `Auto-detected sentinel value ...` or `No sentinel value auto-detected` | `source_mode="auto"` without `sentinel_value` |
| `Histogram features requested but image is not discretised` | The histogram family without a `discretise` step |
| A warning that names a feature family | An error in that family (see [Result Guarantees](#result-guarantees)) |

The NaN and sentinel warnings also go to the `logging` module, which prints them to the screen (stderr) when your script does not set up logging, also in `run_batch`. To hide them, raise the level of the logger:

```python
import logging
import warnings

logging.getLogger().setLevel(logging.ERROR)  # no WARNING lines of the logging module
with warnings.catch_warnings():
    warnings.simplefilter("ignore")           # no UserWarnings of this run
    results = pipeline.run(image, mask, config_names=["ct_fbs_25"])
```

Put the `setLevel` line at the top of the script, outside `if __name__ == "__main__":`, so that the workers of `run_batch` run it too. The log entries keep the sentinel value and the status of each configuration.

## Troubleshooting

- **All features are `NaN`**: look at the log entry of the configuration, `pipeline.get_log()`. Its `status`, `error` and `failed_step` tell the cause. An `"empty_roi"` status means that a mask step removed every voxel: for example a `resegment` range that does not fit the image units.
- **`ValueError` at `add_config`**: the message lists each problem of the configuration. Correct the names and values, or see [Pipeline Steps](pipeline_steps.md).
- **`MemoryError` in the log of a resample step**: the new grid does not fit in the memory of the computer. The message gives the grid size and the memory need, for example for a PET image at 0.5 mm without a mask (about 4 billion voxels). Use a larger spacing, give a mask, or crop the image.
- **A mask on another grid**: a mask of another spacing or orientation raises a `ValueError` at load time, because Pictologics does not resample masks. See [Data Loading](data_loading.md#sub-voxel-alignment-and-overlap).
- **A slow first run**: numba compiles its code on the first import. See [Installation](installation.md#the-first-import).
