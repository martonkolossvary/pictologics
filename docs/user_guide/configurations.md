# Configuration & Reproducibility

A configuration is a named list of steps. This page describes the configurations of the package, the configuration files, and how to share a configuration, so that another site gets the same features.

## The Standard Configurations

`RadiomicsPipeline()` starts with six standard configurations. All six:

- resample to 0.5 × 0.5 × 0.5 mm, with linear interpolation for the image and nearest neighbour for the mask;
- compute the intensity, morphology, texture, histogram and IVH features;
- leave out the spatial and local intensity features, because they take much time on large ROIs.

| Configuration | Discretisation |
|:--|:--|
| `standard_fbn_8` | 8 bins (FBN) |
| `standard_fbn_16` | 16 bins (FBN) |
| `standard_fbn_32` | 32 bins (FBN) |
| `standard_fbs_8` | Bins of 8 HU from -1000 HU (FBS) |
| `standard_fbs_16` | Bins of 16 HU from -1000 HU (FBS) |
| `standard_fbs_32` | Bins of 32 HU from -1000 HU (FBS) |

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run(image, mask, config_names=["standard_fbn_32"])
results = pipeline.run(image, mask, config_names=["standard_fbn_16", "standard_fbs_16"])
results = pipeline.run(image, mask, config_names=["all_standard"])  # all six
```

The steps of `standard_fbs_16`, as `pipeline.get_config("standard_fbs_16")` gives them:

```yaml
- step: resample
  params:
    new_spacing: [0.5, 0.5, 0.5]
    interpolation: linear
- step: discretise
  params:
    method: FBS
    bin_width: 16.0
    min_val: -1000.0
- step: extract_features
  params:
    families: [intensity, morphology, texture, histogram, ivh]
    include_spatial_intensity: false
    include_local_intensity: false
```

The FBN configurations have `method: FBN` and `n_bins` in place of `bin_width` and `min_val`.

## FBN or FBS

| | FBN (a fixed bin number) | FBS (a fixed bin size) |
|:--|:--|:--|
| **The bins** | `n_bins` bins over the intensity range of each ROI | Bins of `bin_width` from a start that is the same in every image |
| **A grey level** | Another intensity range in each image | The same intensity range in every image |
| **CT** | Possible | Recommended: the HU values have a fixed meaning |
| **MR and PET** | Recommended | After a `normalise` step (MR) or in SUV (PET), the values have the same meaning in every image, so FBS suits them too |
| **Small ROIs** | Each bin holds voxels | Many bins can be empty |

- An FBS step needs a fixed start: `min_val`, or a `resegment` step with `range_min` before it. IBSI recommends the lower bound of the resegmentation range. Without both, `add_config` raises an error.
- The standard FBS configurations start at -1000 HU, the HU of air. The cardiac templates start at the lower bound of the HU range of each configuration.

## The Cardiac Templates

Two templates of the package hold configurations for cardiac CT:

```python
pipeline = RadiomicsPipeline.from_template("lv")
results = pipeline.run(image, mask, config_names=["lv_myo_orig", "lv_myo_fbs_16"])

pipeline.merge_configs(RadiomicsPipeline.from_template("coronary"))  # add the second template
```

Each template has four compartments. The `resegment` step of a configuration keeps the HU range of its compartment. All configurations resample to 0.5 mm and use `source_mode="auto"`.

| Template | Compartment | Prefix | HU range |
|:--|:--|:--|:--|
| `lv` | Whole left ventricular myocardium | `lv_` | -179.999 to 2000 |
| | Fat | `lv_fat_` | -179.999 to -30 |
| | Myocardial tissue | `lv_myo_` | -29 to 350 |
| | Calcium | `lv_calc_` | 351 to 2000 |
| `coronary` | All coronary plaque | `coronary_` | -99.999 to 3000 |
| | Non-calcified plaque | `coronary_ncp_` | -99.999 to 350.999 |
| | Low-attenuation plaque | `coronary_lap_` | -99.999 to 29.999 |
| | Calcified plaque | `coronary_cp_` | 351 to 3000 |

- The lower bounds -179.999 and -99.999 keep -180 HU and -100 HU out, also for resampled values. The upper bounds 29.999 and 350.999 keep low-attenuation plaque below 30 HU and non-calcified plaque below the 351 HU of calcified plaque.
- The configurations of each compartment:
    - `<prefix>orig`: the intensity and morphology features.
    - `<prefix>fbn_16`, `<prefix>fbn_32` and `<prefix>fbn_64`: 16, 32 or 64 bins (FBN); the texture, histogram and IVH features.
    - `<prefix>fbs_16`, `<prefix>fbs_32` and `<prefix>fbs_64`: bins of 16, 32 or 64 HU (FBS), from the lower bound of the HU range; the texture, histogram and IVH features.
    - The whole compartments (`lv_` and `coronary_`) also have `fbn_128` and `fbs_128`.
- The `lv` configurations leave out voxels of -3024 HU (`sentinel_value=-3024`): CT padding is often stored as -2000 with a rescale intercept of -1024. The `coronary` configurations find the padding value of each image, with a warning.
- `from_template("standard")` gives the six standard configurations.

## Change a Configuration

`get_config` gives a copy of the steps. Change the copy, and add it with a new name:

```python
pipeline = RadiomicsPipeline()
steps = pipeline.get_config("standard_fbn_32")
for step in steps:
    if step["step"] == "extract_features":
        step["params"]["include_spatial_intensity"] = True  # Moran's I and Geary's C
        step["params"]["include_local_intensity"] = True    # the intensity peaks
pipeline.add_config("fbn_32_with_spatial", steps)
pipeline.remove_config("standard_fbn_8")  # remove a configuration
```

The spatial intensity features compare all pairs of ROI voxels. On large ROIs, they use an FFT, which needs about 32 bytes for each point of a grid two times the ROI box along each axis.

## Configuration Files

`save_configs` writes the configurations to a YAML or a JSON file, by the file extension. A pipeline from `RadiomicsPipeline()` also holds the six standard configurations, so give `config_names`, or start with `load_standard=False`:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline(load_standard=False)
pipeline.add_config("ct_fbs_25", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 25}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
], source_mode="auto", sentinel_value=-2048)

pipeline.save_configs("configs/study.yaml")                             # all configurations
pipeline.save_configs("configs/study.json", config_names=["ct_fbs_25"])  # these only
```

The YAML file:

```yaml
schema_version: '1.0'
pictologics_version: 0.6.0
exported_at: '2026-10-03T19:57:36.489287'
mask_roi_semantics: nonzero_values_are_roi_membership
configs:
  ct_fbs_25:
    steps:
    - step: resample
      params:
        new_spacing:
        - 1.0
        - 1.0
        - 1.0
    - step: resegment
      params:
        range_min: -1000
        range_max: 400
    - step: discretise
      params:
        method: FBS
        bin_width: 25
    - step: extract_features
      params:
        families:
        - intensity
        - morphology
        - texture
    source_mode: auto
    sentinel_value: -2048
deduplication:
  enabled: true
  rules_version: 1.1.0
```

- **The fields**: the schema version of the file, the Pictologics version and the export time, the mask meaning (each voxel that is not 0 is ROI), the configurations with their steps, source mode and sentinel value, and the [deduplication](pipeline.md#shared-work-between-configurations) settings.
- **Ranges**: a `binarize_mask` range `(2, 4)` goes into a file as `{range: [2, 4]}`, because a list selects only the listed labels.
- **NumPy numbers**: NumPy numbers in the steps (for example `np.float64(0.5)`) go into the file as plain numbers.
- **Old files**: a configuration can also be a plain list of steps, without `steps`, `source_mode` and `sentinel_value`.
- **Schema versions**: the current schema version is 1.0. A file without a version is version 1.0. A file of an unknown version loads with a warning.

## Load Configurations

```python
pipeline = RadiomicsPipeline.load_configs("configs/study.yaml")                     # the file only
pipeline = RadiomicsPipeline.load_configs("configs/study.yaml", validate=True)      # with checks
pipeline = RadiomicsPipeline.load_configs("configs/study.yaml", load_standard=True)  # and the standard ones
```

- `load_configs`, `from_yaml`, `from_json` and `from_dict` make a pipeline with the configurations of the file only. `load_standard=True` adds the standard configurations.
- The loaders do not check the steps by default. With `validate=True`, they make the checks of `add_config`: each problem gives a `UserWarning`, and the configuration loads all the same. Use it for a file from another person.
- The deduplication settings of the file apply to the new pipeline.

Text in place of files, for example for a database:

```python
text = pipeline.to_yaml()  # or to_json(), to_dict()
copy = RadiomicsPipeline.from_yaml(text)  # or from_json(), from_dict()
```

## Merge Configurations

```python
pipeline = RadiomicsPipeline.load_configs("team_a.yaml")
pipeline.merge_configs(RadiomicsPipeline.load_configs("team_b.yaml"))                  # keep a name that exists, with a warning
pipeline.merge_configs(RadiomicsPipeline.load_configs("team_b.yaml"), overwrite=True)  # replace it
```

`merge_configs` keeps the source mode and the sentinel value of each configuration.

## The Template Files

The templates are YAML files in the package. The template functions read them:

```python
from pictologics.templates import (
    get_all_templates,
    get_standard_templates,
    get_template_metadata,
    list_template_files,
    load_template_file,
)

list_template_files()        # ['standard_configs.yaml', 'lv_configs.yaml', 'coronary_configs.yaml']
get_standard_templates()     # the steps of the 6 standard configurations
get_all_templates()          # the steps of all 66 configurations of the 3 files
get_template_metadata("lv_configs.yaml")  # the schema version, the description and the config names
load_template_file("lv_configs.yaml")     # the whole file as a dictionary
```

Your own configuration file loads with `load_configs`. Write it with `save_configs`, or by hand in the format above.

## A Study at Two Sites

Site A makes the configurations, tests them and sends the file. Site B loads the file and runs its cases. Both sites then get the features of the same steps.

### 1. Make and Test the Configurations (Site A)

```python
from pictologics import RadiomicsPipeline

base = [
    {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5), "interpolation": "cubic"}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "keep_largest_component", "params": {}},
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 25.0}},  # bins from -1000 HU
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"]}},
]
pipeline = RadiomicsPipeline(load_standard=False)
pipeline.add_config("nodule_fbs_25", base)
wider = [dict(step) for step in base]
wider[3] = {"step": "discretise", "params": {"method": "FBS", "bin_width": 50.0}}
pipeline.add_config("nodule_fbs_50", wider)  # a second bin width, for a sensitivity analysis

results = pipeline.run("test/ct.nii.gz", "test/nodule.nii.gz", config_names=["nodule_fbs_25"])
features = results["nodule_fbs_25"]
print(len(features), features["volume_RNU0"], features["mean_intensity_Q4LE"])
print(pipeline.get_log()[-1]["status"])  # "completed"

pipeline.save_configs("configs/nodule_study_v1.yaml")
```

### 2. Run the Cases (Site B)

```python
from pathlib import Path

from pictologics import RadiomicsPipeline, save_results

if __name__ == "__main__":
    pipeline = RadiomicsPipeline.load_configs("configs/nodule_study_v1.yaml", validate=True)
    cases = [
        {"subject_id": folder.name, "image": str(folder / "ct.nii.gz"), "mask": str(folder / "nodule.nii.gz")}
        for folder in sorted(Path("site_b_data").glob("patient_*"))
    ]
    table = pipeline.run_batch(cases, "site_b_results", workers=4)
    save_results(table, "site_b_results/features.csv")
```

`run_batch` writes each case to `site_b_results/cases/`, with its processing log, and skips the cases that are done when it runs again. See [Many Cases](pipeline.md#many-cases-run_batch).

### 3. Join the Tables

```python
import pandas as pd

site_a = pd.read_csv("site_a_results/features.csv").assign(site="A")
site_b = pd.read_csv("site_b_results/features.csv").assign(site="B")
study = pd.concat([site_a, site_b], ignore_index=True)
study.to_csv("study_features.csv", index=False)
```

Both tables have the same feature columns, because both sites ran the same configurations.

### 4. Keep the Record

Keep these files with the study data:

1. The configuration file, `configs/nodule_study_v1.yaml`.
2. The feature table.
3. The result files of the cases, with their processing logs. Each log entry holds the `config_hash`, the Pictologics version and the package versions.
4. The feature catalog: `pipeline.describe_features().to_csv("feature_catalog.csv", index=False)`.

See [What to Report](results.md#what-to-report).

## Good Practice

- Keep the configuration files in version control (for example git), with the analysis code.
- Write the reason for each choice in the `description` of the file, or in YAML comments (`# ...`).
- Load a file from another person with `validate=True`.
- Report the Pictologics version, the configuration file and the `config_hash` of each configuration.
- Pin the deduplication rules for a long study: `RadiomicsPipeline(deduplication_rules="1.1.0")`.

## Quick Reference

| Task | Code |
|:--|:--|
| Run a configuration | `pipeline.run(image, mask, config_names=["standard_fbn_32"])` |
| Run the standard configurations | `pipeline.run(image, mask, config_names=["all_standard"])` |
| Load a template | `RadiomicsPipeline.from_template("coronary")` |
| List the configurations | `pipeline.list_configs()` |
| Get the steps of a configuration | `pipeline.get_config("name")` |
| Add a configuration | `pipeline.add_config("name", steps)` |
| Remove a configuration | `pipeline.remove_config("name")` |
| Save to a file | `pipeline.save_configs("file.yaml", config_names=[...])` |
| Load from a file | `RadiomicsPipeline.load_configs("file.yaml", validate=True)` |
| Load with the standard configurations | `RadiomicsPipeline.load_configs("file.yaml", load_standard=True)` |
| Merge configurations | `pipeline.merge_configs(other)` |
| Text | `pipeline.to_yaml()`, `RadiomicsPipeline.from_yaml(text)` |
| The feature catalog | `pipeline.describe_features()` |

See also: [The Pipeline](pipeline.md), [Pipeline Steps](pipeline_steps.md), [Image Filtering](image_filtering.md) and the [Cookbook](cookbook.md).
