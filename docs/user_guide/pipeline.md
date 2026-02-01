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

Pictologics includes **6 standard configurations** designed for common radiomics workflows. All standard configurations share:

- **Resampling**: 0.5mm × 0.5mm × 0.5mm isotropic spacing
- **Feature Families**: intensity, morphology, texture, histogram, and IVH
- **Performance-optimized**: Spatial/local intensity disabled by default

| Configuration | Method | Parameters |
| :--- | :--- | :--- |
| `standard_fbn_8` | Fixed Bin Number | `n_bins=8` |
| `standard_fbn_16` | Fixed Bin Number | `n_bins=16` |
| `standard_fbn_32` | Fixed Bin Number | `n_bins=32` |
| `standard_fbs_8` | Fixed Bin Size | `bin_width=8.0` |
| `standard_fbs_16` | Fixed Bin Size | `bin_width=16.0` |
| `standard_fbs_32` | Fixed Bin Size | `bin_width=32.0` |

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()

# Run a single configuration
results = pipeline.run("standard_fbn_32", image, mask)

# Run all 6 standard configurations
all_results = pipeline.run_all_standard_configs(image, mask)
```

!!! tip "Learn More"
    For detailed configuration specifications, FBN vs FBS guidance, export/import capabilities, and best practices, see the **[Predefined Configurations](predefined_configurations.md)** guide.

---

## Deduplication (Performance Optimization)

When running multiple configurations that share preprocessing steps but differ only in discretization (e.g., running all 6 standard configurations), the pipeline can **automatically avoid redundant computation** by enabling deduplication.

### How It Works

!!! info "Enabled by Default"
    Deduplication is **enabled by default** (`deduplicate=True`). You don't need to explicitly set it—just create a pipeline and run multiple configurations to benefit from automatic optimization.

The deduplication system analyzes your configurations and identifies which feature families can be computed once and reused:

1. **Preprocessing Signature**: Each configuration's preprocessing steps (resample, resegment, filter_outliers, etc.) are hashed into a unique signature.
2. **Feature Family Dependencies**: The system knows which preprocessing steps affect which feature families:
    - **Morphology**: Depends only on mask geometry operations (resample, binarize_mask, keep_largest_component). **Not affected by intensity operations or filters.**
    - **Intensity** (including spatial/local): Depends on intensity preprocessing (resample, resegment, filter_outliers, filter). Different filters produce different intensity features.
    - **Texture/Histogram/IVH**: Depends on all of the above **plus** discretization
3. **Execution Plan**: When configs share preprocessing but differ only in discretization, families like morphology and intensity are computed once and reused.

### Behavior: deduplicate=True vs False

| Setting | Behavior | Results |
| :--- | :--- | :--- |
| `deduplicate=True` (default) | Features computed once per unique preprocessing signature, then **copied** to matching configs | All configs receive complete, identical feature values for reused families |
| `deduplicate=False` | Features computed independently for every config | Same results, but slower execution |

!!! note "Results Are Always Complete"
    When deduplication copies features, they are **copied into the results dictionary**—never empty or missing. Every configuration returns a complete feature set, whether features were freshly computed or reused from cache.

!!! tip "Copy Behavior"
    Reused features are **deep copied** to each configuration's result dictionary. Modifying results from one configuration will NOT affect results from another.

### Usage

```python
from pictologics import RadiomicsPipeline

# Deduplication is enabled by default - no need to set it!
pipeline = RadiomicsPipeline()  # deduplicate=True by default

# Run all 6 standard configurations
results = pipeline.run_all_standard_configs(image, mask)

# Check deduplication statistics
print(pipeline.deduplication_stats)
# Example output:
# {
#     'reused_families': 24,  # Families reused from cache
#     'computed_families': 12, # Families freshly computed
#     'cache_hit_rate': 0.67   # 67% of families were reused
# }

# Verify: all configs have complete results (no missing features)
for config_name, features in results.items():
    print(f"{config_name}: {len(features)} features")
```

### Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `deduplicate` | `bool` | `True` | Enable/disable deduplication optimization |
| `deduplication_rules` | `str` or `DeduplicationRules` | `"1.0.0"` | Rules version for reproducibility |

### When to Use Deduplication

!!! success "Recommended For"
    - Running **multiple discretization strategies** (FBN 8/16/32 + FBS 8/16/32) on the same preprocessed image
    - **Batch processing** where many configs share preprocessing steps
    - **Sensitivity studies** varying only discretization parameters

!!! warning "Not Recommended For"
    - Single configuration runs (no benefit)
    - Configurations with different preprocessing steps (nothing to deduplicate)
    - Memory-constrained environments (cached results consume memory)

### Example: 6 Configs with Shared Preprocessing

```python
from pictologics import RadiomicsPipeline

# Define shared preprocessing
base_steps = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},
]

extract_all = {
    "step": "extract_features",
    "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"]},
}

# Create pipeline (deduplication enabled by default)
pipeline = RadiomicsPipeline()  # No need to set deduplicate=True

# Add 6 configurations (only discretization differs)
for n_bins in (8, 16, 32):
    pipeline.add_config(
        f"fbn_{n_bins}",
        base_steps + [
            {"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}},
            extract_all,
        ],
    )

for bin_width in (8.0, 16.0, 32.0):
    pipeline.add_config(
        f"fbs_{int(bin_width)}",
        base_steps + [
            {"step": "discretise", "params": {"method": "FBS", "bin_width": bin_width}},
            extract_all,
        ],
    )

# Run all configs - morphology/intensity computed once, texture 6 times
results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    config_names=["fbn_8", "fbn_16", "fbn_32", "fbs_8", "fbs_16", "fbs_32"],
)

# Inspect cache performance
stats = pipeline.deduplication_stats
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")

# Verify all configs have complete, identical morphology values
ref_vol = results["fbn_8"]["volume_mesh_ml_HTUR"]
for config in ["fbn_16", "fbn_32", "fbs_8", "fbs_16", "fbs_32"]:
    assert results[config]["volume_mesh_ml_HTUR"] == ref_vol
print("✓ Morphology features identical across all configurations")
```

In this example, `morphology` and `intensity` features are computed only once (for the first config) and reused for all 6 configurations, while `texture` and `histogram` are computed 6 times (once per discretization). This can provide significant speedups for large datasets.

!!! tip "API Reference"
    For detailed documentation of the deduplication classes (`ConfigurationAnalyzer`, `DeduplicationPlan`, `PreprocessingSignature`, `DeduplicationRules`), see the **[Deduplication API](../api/deduplication.md)** reference.

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

#### 7. `filter`
Applies an image filter (convolutional filter) to the image. Supports IBSI 2 standard filters.

*   `type`: Filter type (required). Options:
    *   `"mean"`: Mean filter
    *   `"log"`: Laplacian of Gaussian
    *   `"laws"`: Laws' texture energy kernels
    *   `"gabor"`: Gabor filter
    *   `"wavelet"`: Separable wavelets (Haar, Daubechies, Coiflet)
    *   `"simoncelli"`: Non-separable Simoncelli wavelet
    *   `"riesz"`: Riesz transform
    
    !!! tip "See Filter Details"
        For detailed explanations and visual examples of each filter, see the [Image Filtering](image_filtering.md) guide.

*   `boundary`: Boundary condition. Options: `"mirror"` (default), `"nearest"`, `"zero"`, `"periodic"`.
*   Additional filter-specific parameters (see table below).

**Filter Parameter Reference:**

| Filter | Required Params | Optional Params |
|:-------|:----------------|:----------------|
| `mean` | `support` | `boundary` |
| `log` | `sigma_mm` | `truncate`, `boundary`, `spacing_mm` |
| `laws` | `kernel` | `rotation_invariant`, `pooling`, `compute_energy`, `energy_distance`, `boundary` |
| `gabor` | `sigma_mm`, `lambda_mm`, `gamma` | `rotation_invariant`, `delta_theta`, `pooling`, `average_over_planes`, `spacing_mm`, `boundary` |
| `wavelet` | `wavelet`, `level`, `decomposition` | `rotation_invariant`, `pooling`, `boundary` |
| `simoncelli` | `level` | *(no boundary)* |
| `riesz` | `order` | `variant` (`"base"`, `"log"`, `"simoncelli"`), `sigma_mm`, `level` |

!!! note "Automatic spacing injection"
    For filters that require physical spacing (`log`, `gabor`), the pipeline automatically uses the image's voxel spacing if `spacing_mm` is not explicitly provided.

!!! tip "IBSI 2 Compliance"
    For IBSI 2 Phase 2 compliance, use `boundary="mirror"` and apply filters after resampling and intensity rounding.

#### 8. `extract_features`
Calculates the radiomic features based on the current state of the image and mask.

!!! note "Feature Calculation Inputs"
     The pipeline automatically selects the appropriate image state for each feature family:
    
    *   **Intensity, Morphology, Spatial Intensity, Local Intensity**: Calculated on the **Raw Image** (non-discretised, floating-point values).
    *   **Texture, Histogram**: Calculated on the **Discretised Image** (integer bins).
    *   **IVH**: Configurable. Defaults to **Discretised Image**, but can use **Raw Image** (`ivh_use_continuous=True`) or a **Temporary Discretisation** (`ivh_discretisation={...}`).

*   `families`: List of feature families to extract. Options:
    *   `"intensity"`: First-order statistics (Mean, Skewness, etc.).
        *   By default, spatial/local intensity features are **not** included.
        *   Enable via `include_spatial_intensity=True` and/or `include_local_intensity=True`
            in the step `params`.
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

### Example 7: Laplacian of Gaussian (LoG) Filter

Apply LoG filter for edge/blob detection before feature extraction:

```python
from pictologics import RadiomicsPipeline

log_config = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic"}},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter", "params": {
        "type": "log",
        "sigma_mm": 1.5,
        "truncate": 4.0,
    }},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "histogram"]}},
]

pipeline = RadiomicsPipeline().add_config("log_filtered", log_config)
results = pipeline.run("image.nii.gz", "mask.nii.gz", config_names=["log_filtered"])
```

### Example 8: Laws Texture Energy (IBSI 2)

Extract texture energy using Laws kernel with rotation invariance:

```python
laws_config = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic"}},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter", "params": {
        "type": "laws",
        "kernel": "L5E5E5",
        "rotation_invariant": True,
        "pooling": "max",
        "compute_energy": True,
        "energy_distance": 7,
    }},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "texture", "histogram"]}},
]
```

### Example 9: Wavelet Decomposition

Apply Daubechies 3 wavelet with rotation-invariant averaging:

```python
wavelet_config = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic"}},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter", "params": {
        "type": "wavelet",
        "wavelet": "db3",
        "level": 1,
        "decomposition": "LLH",
        "rotation_invariant": True,
        "pooling": "average",
    }},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
]
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

---

## Configuration Export & Import

Pictologics supports exporting and importing pipeline configurations in **YAML** and **JSON** formats for reproducible research.

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
pipeline.add_config("my_study_config", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
])

# Export to YAML or JSON
pipeline.save_configs("my_configs.yaml")

# Import from file
pipeline = RadiomicsPipeline.load_configs("my_configs.yaml")
```

!!! tip "Full Configuration Guide"
    For complete documentation on configuration file formats, schema versioning, merging configurations, 
    validation, and the template system API, see the **[Predefined Configurations](predefined_configurations.md)** guide.
