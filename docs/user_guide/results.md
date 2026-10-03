# Results and Logs

`run()` gives a dictionary: one `pandas.Series` of features for each configuration. This page makes tables from the results, saves them, and describes the feature catalog and the processing log.

## The Result of a Run

```python
results = pipeline.run("ct.nii.gz", "lesion.nii.gz", config_names=["ct_fbs_25"])
features = results["ct_fbs_25"]           # a pandas Series
print(features["mean_intensity_Q4LE"])     # the feature name, then its IBSI code
print(len(features))                       # the number of features
```

A feature key is the feature name and its IBSI code, for example `mean_intensity_Q4LE`. The IBSI code identifies the feature in the IBSI reference manual. Each configuration gives all its feature names, also when a step fails (see [Result Guarantees](pipeline.md#result-guarantees)).

## Tables: `format_results`

`format_results()` makes a table row (wide format) or a list of feature rows (long format) from the results of one run.

=== "Wide format"

    One row for each run, with one column for each feature, named `{config}__{feature}`:

    ```python
    from pictologics import format_results

    row = format_results(results, fmt="wide", meta={"subject_id": "p001"})
    # {"subject_id": "p001", "ct_fbs_25__mean_intensity_Q4LE": 41.2, ...}
    ```

=== "Long format"

    One row for each feature, with the columns `config`, `feature_key` and `value`:

    ```python
    table = format_results(results, fmt="long", meta={"subject_id": "p001"}, output_type="pandas")
    # columns: subject_id, config, feature_key, value
    ```

| Parameter | Default | Description |
|:--|:--|:--|
| `fmt` | `"wide"` | `"wide"` or `"long"` |
| `meta` | `None` | Columns to put first, for example the subject and the ROI |
| `output_type` | `"dict"` | `"dict"` (a dict for wide, a list of dicts for long), `"pandas"` (a DataFrame) or `"json"` (a JSON text; `NaN` becomes `null`) |
| `config_col` | `"config"` | The name of the configuration column of the long format |

The long format suits many configurations and statistics programs such as R. The wide format gives one row for each case or ROI.

## Files: `save_results`

`save_results()` writes results to a file. It takes a dict, a list of dicts, a DataFrame, a list of DataFrames, a JSON text or a list of JSON texts.

```python
from pictologics import save_results

rows = []
for case in cases:
    results = pipeline.run(case["image"], case["mask"], config_names=["ct_fbs_25"])
    rows.append(format_results(results, meta={"subject_id": case["subject_id"]}))
    pipeline.clear_log()  # the log grows with each run
save_results(rows, "study/features.csv")
```

- **File type**: the extension sets it: `.csv`, `.tsv` or `.json` (no extension gives CSV). Another extension raises a `ValueError`. `file_format="tsv"` sets the type for any name.
- **Folders**: a missing folder is made.
- **JSON**: `NaN` and infinite values become `null`, so every JSON reader accepts the file.
- **Columns**: the rows merge by column name. When the rows have other columns (for example cases with other configurations), the table has all columns, and the cells without a value are empty.
- **Many cases**: [`run_batch`](pipeline.md#many-cases-run_batch) gives the table of a whole study, and writes each case to its own file while it runs.

## The Feature Catalog: `describe_features`

`describe_features()` gives one row for each feature of each configuration, before you run it. Save it with your results as a data dictionary.

```python
catalog = pipeline.describe_features()
catalog.to_csv("study/feature_catalog.csv", index=False)
texture = catalog[(catalog["family_group"] == "Texture") & (catalog["discretisation_method"] == "FBN")]
```

| Column | Description |
|:--|:--|
| `config`, `feature_key` | The configuration and the feature key of the results |
| `feature_name`, `ibsi_code` | The feature name without the code, and the IBSI code |
| `family`, `family_group` | The family (for example `glcm`) and its group: Intensity, Morphology or Texture |
| `requires_discretisation` | Whether the family needs a discretised image |
| `uses_morph_mask`, `uses_intensity_mask` | The masks that the feature reads |
| `source_mode`, `sentinel_value` | The source mode of the configuration |
| `feature_extraction_step_index`, `feature_extraction_params` | The `extract_features` step of the feature and its parameters |
| `preprocessing_sequence` | The steps before the extraction, for example `1:resample > 2:resegment > 3:discretise` |
| `preprocessing_steps` | The steps before the extraction, as compact JSON |
| `is_resampled`, `resampling_spacing`, `interpolation` | The resampling |
| `is_discretised`, `discretisation_method`, `discretisation_param` | The discretisation |
| `is_filtered`, `filter_type`, `filter_params` | The filter |
| `is_resegmented`, `is_outlier_filtered`, `is_intensity_rounded`, `keeps_largest_component`, `is_mask_grown`, `is_normalised`, `is_mask_binarized` | Whether the steps before the extraction hold this step |
| `..._apply_to`, `..._params` | The mask (`apply_to`) and the parameters of each of these steps |

- A step parameter column holds the parameters of the configuration only. A summary column such as `interpolation` or `discretisation_method` also gives the default when the step leaves out the parameter.
- A step that repeats gets a JSON list in its `..._params` cell, with one entry for each time, with its `step_index` (from 1).

## The Processing Log

The pipeline keeps a log entry for each configuration that `run()` or `run_rois()` runs. `run_batch()` writes the log of each case into its result file.

```python
for entry in pipeline.get_log():           # a copy of the entries
    if entry["status"] != "completed":
        print(entry["config_name"], entry["status"], entry["error"], entry["failed_step"])

pipeline.save_log("study/log.json")        # the entries, with the versions
pipeline.clear_log()                       # empty the log
```

An entry holds these fields:

| Field | Holds |
|:--|:--|
| `timestamp` | The start of the configuration (ISO 8601) |
| `subject_id` | The `subject_id` of the run |
| `config_name`, `config_hash` | The configuration, and the SHA-256 hash of its source mode, sentinel value and steps |
| `roi` | The ROI name (`run_rois()` only) |
| `image_source`, `image_options` | The image path (`"InMemory"` for an `Image`) and its load options |
| `mask_source` | The mask path, `"InMemory"`, or `"GeneratedFullMask"` without a mask |
| `source_mode`, `sentinel_detected`, `sentinel_value`, `sentinel_auto_detected`, `sentinel_proportion` | The source mode and the sentinel that the run used |
| `config_snapshot` | The source mode, the sentinel value and the steps of the configuration |
| `steps_executed` | Each step that ran, with its parameters. FBS steps add `min_val_effective`, `normalise` steps add `center_effective` and `scale_effective`, and filter steps add the requested and effective boundary and parameters |
| `status` | `"completed"`, `"empty_roi"` or `"error"` |
| `error`, `failed_step` | The error and its step, when there is one |
| `family_errors` | The feature families that failed, with their errors |
| `result_feature_count` | The number of features |
| `elapsed_seconds` | The run time of the configuration |
| `environment` | The versions of Python and of the packages that change feature values (NumPy, SciPy, Numba, PyWavelets, nibabel, pydicom, python-gdcm), the platform, and the thread count |
| `schema_version`, `pictologics_version` | The configuration schema and the Pictologics version |
| `deduplication`, `run_parameters`, `mask_repositioning_settings` | The settings of the run |
| `mask_roi_semantics` | `"nonzero_values_are_roi_membership"`: each voxel that is not 0 is ROI |

- `save_log` writes a JSON file with the entries, the log and schema versions, the Pictologics version and the export time. A name without `.json` gets it.
- The log grows with each run. In a long loop, save it and call `clear_log()`.

## What to Report

IBSI asks a study to report how it computed its features. Pictologics gives each part:

1. **The configurations**: save them with `pipeline.save_configs("study/configs.yaml", config_names=[...])`. The file loads again with `RadiomicsPipeline.load_configs`. See [Configuration & Reproducibility](configurations.md).
2. **The `config_hash`** of each configuration: the same configuration gives the same hash on every computer. It is in each log entry.
3. **The versions**: the `pictologics_version` and the `environment` of the log entries.
4. **The data dictionary**: `pipeline.describe_features()`, with the IBSI code of each feature.
5. **The image processing**: the steps and their parameters (in the configurations), for example the spacing and the interpolation, the resegmentation range, and the discretisation method and its bins.
6. **The IBSI compliance** of the software: see [IBSI 1](../ibsi1_compliance.md) and [IBSI 2](../ibsi2_compliance.md). For the Gabor filter, also report the `response` value (see [Image Filtering](image_filtering.md#gabor-filter)).
