# Radiomics Pipeline

The `RadiomicsPipeline` is the core engine of Pictologics for executing reproducible, standardized radiomic feature extraction workflows. It manages the entire lifecycle of the data, from loading and preprocessing to feature extraction and logging.

## Why use the Pipeline?

1.  **Reproducibility**: By defining a configuration (a sequence of steps), you ensure that the exact same preprocessing is applied to every image.
2.  **State Management**: The pipeline automatically handles the state of the image and masks (morphological and intensity) as they pass through steps like resampling and resegmentation.
3.  **Standardisation**: It comes with built-in configurations that adhere to IBSI standards.
4.  **Batch Processing**: You can run multiple configurations (e.g., different binning strategies) on the same image in a single pass.
5.  **Flexibility**: The pipeline executes steps in a **linear fashion**, allowing you to arrange steps in any order, repeat steps if needed, and implement arbitrarily complex workflows.

---

## Linear Step Execution

The pipeline module executes steps **linearly in order**. This means:

- Steps are applied one after another in the exact sequence you define.
- You can **repeat steps** if needed (e.g., apply `discretise` multiple times with different settings).
- You can **arrange steps in any order** appropriate for your workflow.
- This linear design allows for implementing **complex, multi-stage preprocessing** while maintaining full control.

```python
# Example: Complex workflow with repeated steps
complex_config = [
    {"step": "resample", "params": {"new_spacing": (2.0, 2.0, 2.0)}},
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter_outliers", "params": {"sigma": 3.0}},
    {"step": "round_intensities", "params": {}},  # Round after all preprocessing
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["texture", "histogram"]}},
]
```

---

## Masks are optional

`RadiomicsPipeline.run(...)` accepts an optional `mask` argument.

- If you pass a mask path / mask `Image`, it is used as the ROI (standard radiomics workflow).
- If you omit `mask` (or pass `mask=None` / `mask=""`), Pictologics generates a full (all-ones) ROI mask internally,
  meaning the **entire image** is treated as the initial ROI.

!!! warning
    **Empty ROI is an error**
    If preprocessing removes all ROI voxels (e.g., too strict `resegment` thresholds), the pipeline raises a clear
    error rather than returning empty/partial feature sets.

## Predefined Configurations

Pictologics includes a suite of **Standard Configurations** designed to cover the most common radiomics analysis scenarios. These configurations are compliant with general best practices (e.g., IBSI).

### Common Characteristics
All standard configurations share the following preprocessing steps:

*   **Resampling**: Images are resampled to **0.5mm x 0.5mm x 0.5mm** isotropic spacing using Linear interpolation (Nearest Neighbor for masks).
*   **Feature Families**: All feature families are extracted: `intensity`, `morphology`, `texture`, `histogram`, and `ivh`.

!!! note
    **Performance-friendly default for standard configs**
    The built-in `standard_*` configurations disable the two most time-intensive intensity extras by default:
    `include_spatial_intensity=False` and `include_local_intensity=False`.
    
    This does **not** change the general behavior of the `extract_features` step when you build your own
    custom configuration: if you request `"intensity"` in a custom config and do not specify these flags,
    spatial/local intensity will still be included (backward compatible).

!!! warning
    **Spatial/local intensity features can be very time consuming.**
    Spatial intensity (Moran's I / Geary's C) and local intensity peak features can dominate runtime for larger ROIs.
    For images larger than **50×50×50** voxels, this is generally **not recommended** unless you explicitly need them.
    
    - Standard configs: disabled by default
    - Custom configs: enabled by default (unless you set `include_spatial_intensity=False` / `include_local_intensity=False`)

### Available Configurations

| Configuration Name | Discretisation Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `standard_fbn_8` | Fixed Bin Number (FBN) | `n_bins=8` | Coarse texture analysis. |
| `standard_fbn_16` | Fixed Bin Number (FBN) | `n_bins=16` | Medium texture analysis. |
| `standard_fbn_32` | Fixed Bin Number (FBN) | `n_bins=32` | Fine texture analysis (Common default). |
| `standard_fbs_8` | Fixed Bin Size (FBS) | `bin_width=8.0` | For absolute intensity units (e.g., HU). |
| `standard_fbs_16` | Fixed Bin Size (FBS) | `bin_width=16.0` | For absolute intensity units. |
| `standard_fbs_32` | Fixed Bin Size (FBS) | `bin_width=32.0` | For absolute intensity units. |

### Running Standard Configurations

You can run specific configurations or use the special `"all_standard"` keyword to run all 6 at once.

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()

# Option A: Run specific configurations
results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    config_names=["standard_fbn_32", "standard_fbs_16"]
)

# Option B: Run ALL 6 standard configurations (Recommended for exploration)
all_results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    config_names=["all_standard"]
)

# Accessing results
print(all_results["standard_fbn_32"])
```

### What you get (and what you don't)

The standard configurations are meant to be a fast, reproducible baseline:

*   You get full first-order intensity statistics (`"intensity"`), morphology, textures, histogram, and IVH.
*   You do **not** get spatial/local intensity extras unless you build a custom configuration and enable them.

If you need spatial/local intensity metrics for a specific study, use a custom configuration (examples below).

---

## Custom Configurations

For advanced users, the pipeline allows you to define custom sequences of steps. A configuration is a list of dictionaries, where each dictionary represents a step.

### Structure of a Configuration

```python
config = [
    {
        "step": "step_name",
        "params": { "param1": value1, "param2": value2 }
    },
    # ... more steps
]
```

### Practical tips

*   Keep preprocessing steps explicit (resampling, resegmentation, discretisation) so your results are reproducible.
*   For CT in Hounsfield Units, FBS (`bin_width`) is often more interpretable; for MRI/PET, FBN (`n_bins`) can be a
    reasonable choice depending on your intensity normalization.
*   If you only need a subset of feature families, set `families` to avoid unnecessary work.

### Available Steps

#### 1. `resample`
Resamples the image and mask to a new voxel spacing.

*   `new_spacing`: Tuple of (x, y, z) spacing in mm (e.g., `(1.0, 1.0, 1.0)`).
    *   Alias: `spacing` (older configs/tests).
*   `interpolation`: Interpolation for the image (`"linear"`, `"cubic"`, `"nearest"`). Default: `"linear"`.
*   `mask_interpolation`: Interpolation for the mask (`"nearest"`, `"linear"`). Default: `"nearest"`.
*   `mask_threshold`: When using non-nearest mask interpolation, voxels above this threshold become ROI. Default: `0.5`.
*   `round_intensities`: Whether to round image intensities to nearest integer after resampling. Default: `False`.

#### 2. `resegment`
Refines the mask based on intensity thresholds (e.g., excluding bone from a soft tissue mask).
This is also the **IBSI-recommended approach** for filtering out **sentinel/NA values** (e.g., -1024, -2048 in DICOM)
that represent missing or invalid data.

*   `range_min`: Minimum intensity value.
*   `range_max`: Maximum intensity value.

#### 3. `filter_outliers`
Removes outliers from the intensity mask based on standard deviations from the mean.

*   `sigma`: Number of standard deviations (e.g., `3.0`).

#### 4. `keep_largest_component`
Restricts the mask to the largest connected component. Useful for removing noise or disconnected artifacts.

*   `apply_to`: Which mask(s) to process. Options:
    *   `"both"` (default): Apply to both morphological and intensity masks.
    *   `"morph"`: Apply only to the morphological mask.
    *   `"intensity"`: Apply only to the intensity mask.

#### 5. `round_intensities`
Rounds image intensities to the nearest integer. Useful before discretisation if values are close to integers.

*   *No parameters.*

#### 6. `discretise`
Discretises the image intensities into bins. This is **crucial** for texture analysis.

*   `method`: `"FBN"` (Fixed Bin Number) or `"FBS"` (Fixed Bin Size).
*   `n_bins`: Number of bins (for FBN).
*   `bin_width`: Width of each bin (for FBS).

#### 7. `extract_features`
Calculates the radiomic features based on the current state of the image and mask.

!!! important "Feature Calculation Inputs"
     The pipeline automatically selects the appropriate image state for each feature family:
    
    *   **Intensity, Morphology, Spatial Intensity, Local Intensity**: Calculated on the **Raw Image** (non-discretised, floating-point values).
    *   **Texture, Histogram**: Calculated on the **Discretised Image** (integer bins).
    *   **IVH**: Configurable. Defaults to **Discretised Image**, but can use **Raw Image** (`ivh_use_continuous=True`) or a **Temporary Discretisation** (`ivh_discretisation={...}`).

*   `families`: List of feature families to extract. Options:
    *   `"intensity"`: First-order statistics (Mean, Skewness, etc.).
        *   By default, this also includes **spatial intensity** and **local intensity**.
        *   Disable these expensive computations via `include_spatial_intensity=False` and/or
            `include_local_intensity=False` in the step `params`.
    *   `"spatial_intensity"`: Compute only spatial intensity (Moran's I / Geary's C).
    *   `"local_intensity"`: Compute only local/global intensity peak features.
    *   `"morphology"`: Shape and size features (Volume, Sphericity, etc.).
    *   `"texture"`: GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM.
    *   `"histogram"`: Intensity histogram features.
    *   `"ivh"`: Intensity-Volume Histogram features.

Additional optional parameters (advanced usage):

*   `include_spatial_intensity` / `include_local_intensity`: Booleans controlling whether the expensive
    spatial/local intensity extras are included when `"intensity"` is requested.
*   `ivh_params`: Dict forwarded to `calculate_ivh_features(...)`. Supported keys include:
    `bin_width`, `min_val`, `max_val`, `target_range_min`, `target_range_max`.
    (There are also backward-compatible aliases: `ivh_bin_width`, `ivh_min_val`, `ivh_max_val`,
    `ivh_target_range_min`, `ivh_target_range_max`.)
*   `ivh_discretisation`: Dict specifying a **temporary discretisation** for IVH only. This allows
    using different binning for IVH vs texture features. Example: `{"method": "FBS", "bin_width": 2.5, "min_val": -1000}`.
*   `ivh_use_continuous`: Boolean. If `True`, uses raw (non-discretised) intensity values for IVH calculation.
    Useful for "continuous IVH" as specified in some IBSI configurations.
*   `texture_matrix_params`: Dict forwarded to `calculate_all_texture_matrices(...)`.
    Currently useful key: `ngldm_alpha` (IBSI default is `0`).

---

## Examples

### Example 1: Standard suite (fast baseline)

Runs all 6 built-in configurations. Spatial/local intensity extras are disabled by default.

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    config_names=["all_standard"],
)

# Access one configuration
print(results["standard_fbn_32"].head())

```

### Example 1b: Maskless run (whole-image ROI)

If you do not have a segmentation mask, you can omit the `mask` argument entirely.

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run(
    image="path/to/image.nii.gz",
    # mask omitted -> whole-image ROI
    config_names=["standard_fbn_32"],
)

print(results["standard_fbn_32"].head())
```

!!! note
    **Morphology meaning**
    With a maskless run, morphology features describe the ROI mask after any mask-refining steps
    (e.g., `resegment`, `keep_largest_component`). Starting from a whole-image ROI can be valid,
    but may not be scientifically meaningful for many radiomics studies.

### Example 2: Enable spatial/local intensity extras (custom config)

Use this when you explicitly need Moran’s I / Geary’s C and local intensity peak features.

```python
from pictologics import RadiomicsPipeline

cfg = [
    {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5)}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True,
            "include_local_intensity": True,
        },
    },
]

pipeline = RadiomicsPipeline().add_config("with_intensity_extras", cfg)
out = pipeline.run("path/to/image.nii.gz", "path/to/mask.nii.gz", config_names=["with_intensity_extras"])
print(out["with_intensity_extras"].filter(like="_"))
```

### Example 3: Only compute the expensive parts (explicit families)

If you only want spatial/local intensity and not the entire first-order intensity set:

```python
from pictologics import RadiomicsPipeline

cfg = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {
        "step": "extract_features",
        "params": {"families": ["spatial_intensity", "local_intensity"]},
    },
]

pipeline = RadiomicsPipeline().add_config("intensity_extras_only", cfg)
out = pipeline.run("path/to/image.nii.gz", "path/to/mask.nii.gz", config_names=["intensity_extras_only"])
print(out["intensity_extras_only"].head())
```

### Example 4: IVH with physical-unit mapping (advanced)

When you discretise with FBS, you can map IVH to physical units by passing `bin_width` and `min_val`.

```python
from pictologics import RadiomicsPipeline

cfg = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 25.0, "min_val": -1000}},
    {
        "step": "extract_features",
        "params": {
            "families": ["ivh"],
            "ivh_params": {
                "bin_width": 25.0,
                "min_val": -1000,
                "target_range_max": 400,
            },
        },
    },
]

pipeline = RadiomicsPipeline().add_config("ivh_hu", cfg)
out = pipeline.run("path/to/image.nii.gz", "path/to/mask.nii.gz", config_names=["ivh_hu"])
print(out["ivh_hu"].head())
```

### Example 5: Texture with NGLDM tolerance (`ngldm_alpha`)

IBSI default is `ngldm_alpha=0` (exact match). If you want tolerance of ±1 grey level, set `ngldm_alpha=1`.

```python
from pictologics import RadiomicsPipeline

cfg = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {
        "step": "extract_features",
        "params": {
            "families": ["texture"],
            "texture_matrix_params": {"ngldm_alpha": 1},
        },
    },
]

pipeline = RadiomicsPipeline().add_config("texture_ngldm_tolerant", cfg)
out = pipeline.run("path/to/image.nii.gz", "path/to/mask.nii.gz", config_names=["texture_ngldm_tolerant"])
print(out["texture_ngldm_tolerant"].head())
```

### Example: Custom CT Pipeline

```python
custom_config = [
    # 1. Resample to 1mm isotropic
    {
        "step": "resample",
        "params": {"new_spacing": (1.0, 1.0, 1.0)}
    },
    # 2. Restrict to soft tissue window (-150 to 250 HU)
    {
        "step": "resegment",
        "params": {"range_min": -150, "range_max": 250}
    },
    # 3. Discretise with Fixed Bin Number = 64
    {
        "step": "discretise",
        "params": {"method": "FBN", "n_bins": 64}
    },
    # 4. Extract everything
    {
        "step": "extract_features",
        "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"]}
    }
]

pipeline = RadiomicsPipeline()
pipeline.add_config("my_custom_ct", custom_config)
results = pipeline.run(image, mask, config_names=["my_custom_ct"])
```

---

## Logging

The pipeline maintains a detailed log of every step executed, including parameters and any errors encountered. This is vital for auditing and debugging.

```python
# After running the pipeline
pipeline.save_log("pipeline_execution_log.json")
```

The log file contains:

*   Timestamp
*   Subject ID
*   Configuration Name
*   List of executed steps with their parameters
*   Status of each step

