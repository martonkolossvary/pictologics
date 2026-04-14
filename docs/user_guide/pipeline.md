# Pipeline & Preprocessing

The `RadiomicsPipeline` is the core engine of Pictologics for executing reproducible, standardized radiomic feature extraction workflows. It manages the entire lifecycle from preprocessing to feature extraction and logging.

## Why Use the Pipeline?

1.  **Reproducibility**: Define a configuration once and apply it consistently to every image.
2.  **State Management**: The pipeline tracks the image and masks (morphological and intensity) through every step.
3.  **Standardisation**: Built-in configurations follow IBSI standards.
4.  **Batch Processing**: Run multiple configurations (e.g., different binning strategies) on the same image in a single pass.
5.  **Flexibility**: Steps execute **linearly**, so you can arrange them in any order or repeat steps.

## Getting Started

```python
from pictologics import RadiomicsPipeline, format_results, save_results

# 1. Initialize the pipeline
pipeline = RadiomicsPipeline()

# 2. Run a predefined configuration
results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    subject_id="Subject_001",
    config_names=["standard_fbn_32"],
)

# 3. Format and save results
row = format_results(results, fmt="wide", meta={"subject_id": "Subject_001"})
save_results([row], "results.csv")
```

### Masks Are Optional

`RadiomicsPipeline.run(...)` accepts an optional `mask` argument:

- Pass a mask path or `Image` object → used as the ROI (standard workflow).
- Omit `mask` (or pass `mask=None` / `mask=""`) → Pictologics generates a full (all-ones) ROI mask, treating the **entire image** as the initial ROI.

!!! info "Complete Feature Sets Guaranteed"
    Every configuration always returns a `pandas.Series` with the **full set of expected
    feature names** — even when extraction fails partially or entirely.

    - **Empty ROI** (e.g., too strict `resegment` thresholds): all features are `NaN`.
    - **Partial failures** (e.g., mesh generation error, PCA with ≤3 voxels):
      successfully computed features retain their values; only the missing features are `NaN`.
    - **Unexpected errors**: all features are `NaN`.

    Other configurations in the same `run()` call continue normally.  The processing log
    records errors and the step that caused them.

    This design ensures that multi-configuration batch runs always produce a complete,
    predictable result dictionary — downstream formatting, concatenation, and CSV export
    work without missing columns or unexpected exceptions.

    See [Result Guarantees](#result-guarantees) for details.

    ```python
    results = pipeline.run(image, mask, config_names=["strict", "lenient"])
    # If "strict" empties the ROI:
    #   results["strict"]  -> Series of NaN (all expected feature names present)
    #   results["lenient"] -> Series of computed values
    ```

!!! note "Morphology with Whole-Image ROI"
    With a maskless run, morphology features describe the ROI mask after mask-refining steps
    (e.g., `resegment`, `keep_largest_component`). This is valid computationally, but may not be
    scientifically meaningful for all studies.

## Predefined Configurations

Pictologics includes **6 standard configurations** designed for common radiomics workflows. All share:

- **Resampling**: 0.5mm × 0.5mm × 0.5mm isotropic spacing
- **Feature Families**: intensity, morphology, texture, histogram, and IVH
- **Performance**: Spatial/local intensity disabled by default

| Configuration | Method | Parameters |
| :--- | :--- | :--- |
| `standard_fbn_8` | Fixed Bin Number | `n_bins=8` |
| `standard_fbn_16` | Fixed Bin Number | `n_bins=16` |
| `standard_fbn_32` | Fixed Bin Number | `n_bins=32` |
| `standard_fbs_8` | Fixed Bin Size | `bin_width=8.0` |
| `standard_fbs_16` | Fixed Bin Size | `bin_width=16.0` |
| `standard_fbs_32` | Fixed Bin Size | `bin_width=32.0` |

```python
# Run a single configuration
results = pipeline.run(image, mask, config_names=["standard_fbn_32"])

# Run all 6 standard configurations
all_results = pipeline.run(image, mask, config_names=["all_standard"])
```

!!! tip "Configuration Management"
    For detailed documentation on FBN vs FBS guidance, export/import, YAML/JSON formats, schema versioning, and sharing configurations, see the **[Configuration & Reproducibility](configurations.md)** guide.

## Linear Step Execution

Steps are applied **one after another** in the exact sequence you define. You can **repeat steps**, **arrange steps in any order**, and implement **complex multi-stage preprocessing**:

```python
# Example: Complex workflow with repeated steps
complex_config = [
    {"step": "resample", "params": {"new_spacing": (2.0, 2.0, 2.0)}},
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter_outliers", "params": {"sigma": 3.0}},
    {"step": "round_intensities", "params": {}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["texture", "histogram"]}},
]
```

### Intelligent Image Routing

After discretisation, the pipeline maintains both the **original (raw)** image and the **discretised** image, ensuring each feature type gets the appropriate input automatically:

| Feature Family | Image Used | Why |
|:---------------|:-----------|:----|
| **Intensity** | Raw image | Statistics require original continuous values |
| **Morphology** | Raw image | Volume/surface calculations use original geometry |
| **Histogram** | Discretised | Bin-based statistics require integer bins |
| **Texture** (GLCM, GLRLM, etc.) | Discretised | Co-occurrence matrices require discrete grey levels |
| **IVH** | Configurable | Can use raw (continuous) or discretised values |

## Available Preprocessing Steps

### 1. `resample`

Resamples the image and mask to a new voxel spacing.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `new_spacing` | `tuple` | *(required)* | Target spacing (x, y, z) in mm |
| `interpolation` | `str` | `"linear"` | Image interpolation: `"linear"`, `"cubic"`, `"nearest"` |
| `mask_interpolation` | `str` | `"nearest"` | Mask interpolation: `"nearest"`, `"linear"` |
| `mask_threshold` | `float` | `0.5` | Threshold for non-nearest mask interpolation |
| `round_intensities` | `bool` | `False` | Round intensities to nearest integer after resampling |

### 2. `resegment`

Refines the ROI mask based on intensity thresholds, excluding voxels outside the specified range from feature extraction. This is essential for removing sentinel/NA values (e.g., -1024, -2048) from the ROI.

!!! warning "Memory Usage Alert"
    If your image has a background that resamples to 0, and 0 is within your `resegment` range, **you must use `source_mode="auto"`**.
    Otherwise, `resegment` will include the entire background in the ROI, causing memory exhaustion during texture calculation. `source_mode="auto"` ensures the background remains excluded.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `range_min` | `float` | `None` | Minimum intensity value |
| `range_max` | `float` | `None` | Maximum intensity value |

### 3. `filter_outliers`

Removes outliers from the intensity mask based on standard deviations from the mean.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `sigma` | `float` | `3.0` | Number of standard deviations |

### 4. `keep_largest_component`

Restricts the mask to the largest connected component.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `apply_to` | `str` | `"both"` | `"both"`, `"morph"`, or `"intensity"` |

### 5. `round_intensities`

Rounds image intensities to the nearest integer. Useful before discretisation if values are close to integers.

*No parameters.*

### 6. `binarize_mask`

Creates a binary mask from a multi-label mask.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `threshold` | `float` | `0.5` | Threshold value for binarization |
| `mask_values` | `int`, `list`, or `tuple` | `None` | Specific label(s) to select. Tuple `(min, max)` selects a range |
| `apply_to` | `str` | `"both"` | `"both"`, `"morph"`, or `"intensity"` |

### 7. `discretise`

Discretises image intensities into bins. **Required** before texture feature extraction.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `method` | `str` | *(required)* | `"FBN"` (Fixed Bin Number) or `"FBS"` (Fixed Bin Size) |
| `n_bins` | `int` | `None` | Number of bins (for FBN) |
| `bin_width` | `float` | `None` | Width of each bin (for FBS) |

### 8. `filter`

Applies an IBSI 2 image filter. See the **[Image Filtering](image_filtering.md)** guide for detailed documentation.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `type` | `str` | *(required)* | `"mean"`, `"log"`, `"laws"`, `"gabor"`, `"wavelet"`, `"simoncelli"`, `"riesz"` |
| `boundary` | `str` | `"mirror"` | Boundary condition |

**Filter-specific parameters:**

| Filter | Required Params | Optional Params |
|:-------|:----------------|:----------------|
| `mean` | `support` | `boundary` |
| `log` | `sigma_mm` | `truncate`, `boundary` |
| `laws` | `kernel` | `rotation_invariant`, `pooling`, `compute_energy`, `energy_distance`, `boundary` |
| `gabor` | `sigma_mm`, `lambda_mm`, `gamma` | `rotation_invariant`, `delta_theta`, `pooling`, `boundary` |
| `wavelet` | `wavelet`, `level`, `decomposition` | `rotation_invariant`, `pooling`, `boundary` |
| `simoncelli` | `level` | — |
| `riesz` | `order` | `variant`, `sigma_mm`, `level` |

!!! note "Automatic Spacing Injection"
    For filters requiring physical spacing (`log`, `gabor`), the pipeline uses the image's voxel spacing automatically.

### 9. `extract_features`

Calculates radiomic features from the current state.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `families` | `list[str]` | *(required)* | Feature families to extract (see table below) |
| `include_spatial_intensity` | `bool` | `False` | Include Moran's I / Geary's C |
| `include_local_intensity` | `bool` | `False` | Include local intensity peaks |
| `ivh_params` | `dict` | `None` | Parameters for IVH: `bin_width`, `min_val`, `max_val`, etc. |
| `ivh_discretisation` | `dict` | `None` | Temporary discretisation for IVH only |
| `ivh_use_continuous` | `bool` | `False` | Use raw values for IVH |
| `texture_matrix_params` | `dict` | `None` | E.g., `{"ngldm_alpha": 1}` |

**Available feature families:**

| Family | Description |
|:-------|:------------|
| `"intensity"` | First-order statistics (Mean, Skewness, etc.) |
| `"spatial_intensity"` | Moran's I / Geary's C only |
| `"local_intensity"` | Local/global intensity peak features only |
| `"morphology"` | Shape and size features (Volume, Sphericity, etc.) |
| `"texture"` | GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM |
| `"histogram"` | Intensity histogram features |
| `"ivh"` | Intensity-Volume Histogram features |

## Working with Results

The `format_results()` function converts pipeline output into different formats for analysis or export.

### Format Options

=== "Wide Format"

    One row per subject with all features as columns. Column names use the pattern `{config}__{feature}`.

    ```python
    row = format_results(results, fmt="wide", meta={"subject_id": "case1"})
    # Returns: {"subject_id": "case1", "standard_fbn_32__mean_intensity_Q4LE": 123.4, ...}
    ```

=== "Long Format"

    Tidy data with one row per feature. Config name is automatically included.

    ```python
    df = format_results(results, fmt="long", meta={"subject_id": "case1"}, output_type="pandas")
    # Returns DataFrame: [subject_id, config, feature_name, value]
    ```

### Output Types

| Type | Returns |
|:-----|:--------|
| `"dict"` (default) | Python dictionary (wide) or list of dicts (long) |
| `"pandas"` | `pandas.DataFrame` |
| `"json"` | JSON string |

### Batch Processing Pattern

```python
all_rows = []
for file in image_files:
    res = pipeline.run(image=file, ...)
    all_rows.append(format_results(res, fmt="wide", meta={"filename": file.name}))

# Save everything at once
save_results(all_rows, "full_study_results.csv")
```

### Result Guarantees

Every configuration in a `run()` call **always** returns a `pandas.Series` with a
complete, predictable set of feature names — regardless of whether extraction
succeeded, partially succeeded, or failed entirely.  This guarantee holds at three
levels:

| Failure Level | Example | Behaviour |
|:---|:---|:---|
| **Whole-configuration failure** | Empty ROI after `resegment` | All features set to `NaN` |
| **Partial feature failure** | Mesh generation error in morphology, PCA with ≤3 voxels, empty texture matrix | Successfully computed features retain their values; only the missing features are set to `NaN` |
| **Unexpected runtime error** | Uncaught exception during extraction | All features set to `NaN` |

In every case:

- **No missing columns.** The feature names in the returned Series are identical to
  those a fully successful extraction would have produced.
- **Other configurations continue.** A failure in one configuration does not prevent
  subsequent configurations from running.
- **Errors are logged.** The processing log records the error message and the step that
  caused the failure, accessible via `pipeline.save_log()`.

This design makes batch processing safe: when you collect rows from many subjects with
`format_results()` and merge them with `save_results()`, every row has the same columns.
There are no ragged rows, no missing columns, and no unexpected exceptions.

!!! note "Name-Based Merging"
    `format_results()` and `save_results()` always merge results by **column name**,
    never by position.  Even though all configurations now produce the same set of
    feature names, the merging logic is inherently name-based — columns are identified
    by their `{config}__{feature}` key (wide format) or `feature_name` value (long
    format), so results are always aligned correctly.

#### How It Works Internally

The pipeline uses a static `FEATURE_NAMES` registry (in `pictologics.features`) that
enumerates every feature name produced by each family.  Three mechanisms ensure
completeness:

1. **Empty ROI**: When `EmptyROIMaskError` is raised during preprocessing, the pipeline
   builds a full NaN Series directly from the registry without attempting extraction.
2. **Partial failures**: After each `extract_features` call, a backfill step compares
   the returned feature keys against the registry and inserts `NaN` for any missing
   keys.  This catches edge cases where individual features cannot be computed (e.g.,
   mesh fails → surface-based morphology features are `NaN`, but volume from voxel
   counting is preserved).
3. **Unexpected errors**: If an unhandled exception interrupts extraction, the general
   error handler backfills all expected feature names with `NaN` using the same
   registry.

#### Example: Empty ROI in a Multi-Configuration Run

```python
pipeline.add_config("strict", [
    {"step": "resegment", "params": {"range_min": 100, "range_max": 200}},
    {"step": "extract_features", "params": {"families": ["intensity"]}},
])
pipeline.add_config("lenient", [
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 3000}},
    {"step": "extract_features", "params": {"families": ["intensity"]}},
])

results = pipeline.run(image, mask, config_names=["strict", "lenient"])

# If the strict range empties the ROI:
#   results["strict"]  -> 18 intensity features, all NaN
#   results["lenient"] -> 18 intensity features, computed values
#
# format_results() works normally — the NaN row merges cleanly with other rows:
row = format_results(results, fmt="wide", meta={"subject_id": "case1"})
```

!!! tip "Detecting Failed Configurations"
    After a batch run, inspect the processing log to find which configurations
    failed or produced partial results:

    ```python
    for entry in pipeline._log:
        if "error" in entry:
            print(f"{entry['config_name']}: {entry['error']}")
    ```

## Feature Catalog

The [`describe_features()`][pictologics.pipeline.RadiomicsPipeline.describe_features] method returns a DataFrame cataloguing every feature the pipeline will produce **before** you run it.  Each row is one *(configuration, feature)* pair with columns describing the feature identity, family membership, and preprocessing parameters.

```python
pipeline = RadiomicsPipeline()
catalog = pipeline.describe_features()
catalog.head()
```

| Column | Description |
|---|---|
| `config` | Configuration name |
| `feature_key` | Full feature key as it appears in the output |
| `feature_name` | Human-readable name (IBSI code stripped) |
| `ibsi_code` | 3–4 character IBSI identifier |
| `family` | Granular family (e.g. `glcm`, `ivh`) |
| `family_group` | Broad category: *Intensity*, *Morphology*, or *Texture* |
| `requires_discretisation` | Whether the family needs discretised input |
| `is_discretised` / `discretisation_method` / `discretisation_param` | Discretisation details |
| `is_resampled` / `resampling_spacing` / `interpolation` | Resampling details |
| `is_filtered` / `filter_type` / `filter_params` | Response-map filter details |

### Typical Use Cases

**Export a data dictionary** alongside study results:

```python
catalog.to_csv("feature_catalog.csv", index=False)
```

**Filter features** for downstream analysis:

```python
# Only texture features from FBN configs
texture_fbn = catalog[
    (catalog["family_group"] == "Texture")
    & (catalog["discretisation_method"] == "FBN")
]
```

## Deduplication (Performance Optimization)

When running multiple configurations that share preprocessing steps, the pipeline **automatically avoids redundant computation**.

!!! info "Enabled by Default"
    Deduplication is enabled by default (`deduplicate=True`). Just run multiple configs to benefit.

### How It Works

The system analyzes your configurations and identifies reusable features:

| Feature Family | Depends On | Independent Of |
| :--- | :--- | :--- |
| **Morphology** | Mask geometry (resample, binarize_mask, keep_largest_component) | Intensity values, filters, discretization |
| **Intensity** | Intensity preprocessing (resample, resegment, filter_outliers, filter) | Discretization |
| **Texture / Histogram** | All of the above **plus** discretization | — |

When configs share preprocessing but differ only in discretization:

- **Morphology** and **intensity** are computed **once** and reused
- **Texture** and **histogram** are computed per configuration

### Checking Statistics

```python
stats = pipeline.deduplication_stats
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Reused: {stats['reused_families']} families")
print(f"Computed: {stats['computed_families']} families")
```

!!! note "Results Are Always Complete"
    Deduplication does not affect the completeness guarantee.  Reused features are
    **deep copied** into each configuration's results, and every config returns a
    complete feature set — no missing values.

### Configuration

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `deduplicate` | `bool` | `True` | Enable/disable deduplication |
| `deduplication_rules` | `str` or `DeduplicationRules` | `"1.0.0"` | Rules version for reproducibility |

!!! tip "API Reference"
    For detailed documentation of `ConfigurationAnalyzer`, `DeduplicationPlan`, `PreprocessingSignature`, and `DeduplicationRules`, see the **[Deduplication API](../api/deduplication.md)** reference.

## Logging

The pipeline maintains a detailed log of every step executed, including parameters and errors.

```python
# Save log after running
pipeline.save_log("pipeline_execution_log.json")

# Clear log between runs
pipeline.clear_log()
```

The log file contains:

- Timestamp and subject ID
- Configuration name
- Source mode and sentinel detection status
- List of executed steps with parameters
- Status of each step

## Examples

### Standard Suite (Fast Baseline)

Run all 6 built-in configurations:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    config_names=["all_standard"],
)
```

### Enable Spatial/Local Intensity Extras

```python
cfg = [
    {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5)}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True,  # Moran's I / Geary's C
            "include_local_intensity": True,    # Local intensity peaks
        },
    },
]

pipeline = RadiomicsPipeline().add_config("with_extras", cfg)
results = pipeline.run("image.nii.gz", "mask.nii.gz", config_names=["with_extras"])
```

### IVH with Physical-Unit Mapping

```python
cfg = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 25.0, "min_val": -1000}},
    {
        "step": "extract_features",
        "params": {
            "families": ["ivh"],
            "ivh_params": {"bin_width": 25.0, "min_val": -1000, "target_range_max": 400},
        },
    },
]
```

### Custom CT Pipeline

```python
custom_config = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "resegment", "params": {"range_min": -150, "range_max": 250}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 64}},
    {"step": "extract_features", "params": {
        "families": ["intensity", "morphology", "texture", "histogram", "ivh"]
    }},
]

pipeline = RadiomicsPipeline().add_config("my_custom_ct", custom_config)
results = pipeline.run(image, mask, config_names=["my_custom_ct"])
```

### LoG Filtered Features

```python
log_config = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic"}},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter", "params": {"type": "log", "sigma_mm": 1.5, "truncate": 4.0}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "histogram"]}},
]
```

### Manual Step-by-Step Extraction

If you need granular control, call features directly without the pipeline:

```python
import numpy as np
from pictologics import load_image
from pictologics.preprocessing import (
    resample_image, resegment_mask, filter_outliers,
    discretise_image, apply_mask
)
from pictologics.features.intensity import calculate_intensity_features
from pictologics.features.morphology import calculate_morphology_features
from pictologics.features.texture import calculate_all_texture_features

# Load and preprocess
image = load_image("image.nii.gz")
mask = load_image("mask.nii.gz")
image = resample_image(image, new_spacing=(1.0, 1.0, 1.0))
mask = resample_image(mask, new_spacing=(1.0, 1.0, 1.0), interpolation="nearest")
mask = resegment_mask(image, mask, range_min=-1000, range_max=400)

# Discretise for texture
disc_image = discretise_image(image, method="FBN", n_bins=32, roi_mask=mask)

# Extract features
morph = calculate_morphology_features(mask, image=image, intensity_mask=mask)
intensity = calculate_intensity_features(apply_mask(image, mask))
texture = calculate_all_texture_features(disc_image.array, mask.array, n_bins=32)

all_features = {**morph, **intensity, **texture}
print(f"Extracted {len(all_features)} features")
```

!!! tip "Use the Pipeline Instead"
    The `RadiomicsPipeline` accomplishes the same workflow with automatic image routing, logging,
    deduplication, and configuration export. Manual extraction is mainly useful for debugging
    or understanding the underlying process.

## Performance Tips

- **Spatial/local intensity** can be extremely slow on large ROIs. Keep them disabled unless needed.
- **Texture** requires discretisation. Without a `discretise` step, the pipeline raises an error.
- For large 3D images, consider coarser spacing for exploratory work.
- For CT in Hounsfield Units, FBS (`bin_width`) is often more interpretable; for MRI/PET, FBN (`n_bins`) may be preferable.
