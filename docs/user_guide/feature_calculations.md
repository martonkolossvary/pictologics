# Feature Calculations

This guide walks you through extracting radiomic features from medical images using **Pictologics**.

## Prerequisites

You will need:

1.  A medical image (e.g., `image.nii.gz`, DICOM folder, or a single DICOM file).
2.  Optionally a corresponding mask/segmentation (e.g., `mask.nii.gz`).

!!! note
    **Mask is optional**
    If you omit `mask` (or pass `mask=None` / `mask=""`), Pictologics treats the entire image as the ROI by
    generating a full (all-ones) mask internally. This can be useful for certain workflows (see the
    [Case examples](case_examples.md) page), but it may not be scientifically appropriate for all studies.

Detailed information on how to load images and masks can be found in the [Data Loading](data_loading.md) guide.

## Method 1: The Radiomics Pipeline (Recommended)

For reproducible research and standardisation, use the `RadiomicsPipeline`. This ensures that all preprocessing steps (resampling, resegmentation, discretisation) are applied consistently.

Pictologics includes a set of **Standard Configurations** commonly used in radiomic analyses. You can run all of them with a single command.

!!! note
    **Default performance behavior**
    The built-in `standard_*` configurations disable the two most time-intensive intensity extras by default:
    spatial intensity (Moran's I / Geary's C) and local intensity peak features.
    
    If you need these metrics, see the customization examples below.

```python
from pictologics import RadiomicsPipeline, format_results, save_results

# 1. Initialize the pipeline
pipeline = RadiomicsPipeline()

# 2. Run the "all_standard" configurations
results = pipeline.run(
    image="path/to/image.nii.gz",
    mask="path/to/mask.nii.gz",
    subject_id="Subject_001",
    config_names=["all_standard"]
)

# 3. Format and Save Results
row = format_results(
    results, 
    fmt="wide", 
    meta={"subject_id": "Subject_001"}
)
save_results([row], "results.csv")

print(f"Saved {len(results)} configurations to results.csv")
```

### Intelligent image routing

The pipeline automatically uses the correct image for each feature family. After discretisation, the pipeline maintains both the **original (raw)** image and the **discretised** image, ensuring each feature type gets the appropriate input.

| Feature Family | Image Used | Why |
|----------------|------------|-----|
| **Intensity** | Raw image | Statistics require original continuous values |
| **Morphology** | Raw image | Volume/surface calculations use original geometry |
| **Histogram** | Discretised | Bin-based statistics require integer bins |
| **Texture** (GLCM, GLRLM, etc.) | Discretised | Co-occurrence matrices require discrete grey levels |
| **IVH** | Configurable | Can use raw (continuous) or discretised values |

**Example workflow:**

```python
pipeline.add_config("my_config", [
    # Step 0: Binarize mask before resampling
    {"step": "binarize_mask", "params": {"threshold": 0.5}},
    # Step 1: Resample to 0.5mm isotropic
    {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5)}},
    
    # Step 2: Discretise with 32 bins (creates discretised copy)
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    
    # Step 3: Extract features (routing happens automatically)
    {"step": "extract_features", "params": {
        "families": ["intensity", "morphology", "texture", "histogram"]
    }}
])

# When extract_features runs:
# - intensity features → uses resampled raw image
# - morphology features → uses resampled raw image  
# - histogram features → uses discretised image
# - texture features → uses discretised image
```

This means you don't need to worry about which image to pass — the pipeline handles it correctly.

### Working with results

The `format_results()` function converts pipeline output into different formats for analysis or export.

**Format Options (`fmt`):**

1.  **Wide Format** (`fmt="wide"`): One row per subject with all features as columns.
    Column names use the pattern `{config}__{feature}` (e.g., `standard_fbn_32__mean_intensity_Q4LE`).
    ```python
    row = format_results(results, fmt="wide", meta={"subject_id": "case1"})
    # Returns: {"subject_id": "case1", "standard_fbn_32__mean_intensity_Q4LE": 123.4, ...}
    ```

2.  **Long Format** (`fmt="long"`): Tidy data with one row per feature. The configuration name is automatically included in a `config` column — you don't need to specify it in `meta`.
    ```python
    df = format_results(results, fmt="long", meta={"subject_id": "case1"}, output_type="pandas")
    # Returns DataFrame with columns: [subject_id, config, feature_name, value]
    # Example rows:
    # | subject_id | config          | feature_name        | value  |
    # |------------|-----------------|---------------------|--------|
    # | case1      | standard_fbn_32 | mean_intensity_Q4LE | 123.4  |
    # | case1      | standard_fbn_32 | volume_RNU0         | 5420.0 |
    # | case1      | standard_fbs_8  | mean_intensity_Q4LE | 123.4  |
    ```

**Output Types (`output_type`):**

- `"dict"` (default): Returns a Python dictionary (wide) or list of dicts (long)
- `"pandas"`: Returns a `pandas.DataFrame`
- `"json"`: Returns a JSON string

### Batch Processing Pattern

```python
all_rows = []
for file in image_files:
    res = pipeline.run(image=file, ...)
    # Format and collect
    all_rows.append(format_results(res, fmt="wide", meta={"filename": file.name}))

# Save everything at once (automatically merges columns)
save_results(all_rows, "full_study_results.csv")
```

### Customizing the pipeline

You can define your own steps to enable advanced features or change parameters.

```python
from pictologics import RadiomicsPipeline

# Define a config ensuring 0.5mm isotropic pixels and enabling robust texture extraction
cfg = [
    {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5), "round_intensities": True}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True,  # Enable Moran's I / Geary's C
            "include_local_intensity": True,    # Enable local intensity peaks
        },
    },
]

pipeline = RadiomicsPipeline().add_config("my_custom_config", cfg)
# ... run as normal
```

---

## Performance notes (practical)

*   **Spatial/local intensity** can be extremely slow on large ROIs. If you do not need them, keep `include_spatial_intensity=False` and `include_local_intensity=False`.
*   **Texture** requires discretisation. If you request `"texture"` without including a `discretise` step first, the pipeline will raise an error.
*   If you are working with large 3D images, consider resampling to a coarser spacing for exploratory work.

## Method 2: Step-by-Step Manual Extraction

If you want to understand the underlying process or need granular control over a specific function, you can call the feature extraction functions directly.

Create a new Python file (e.g., `extract.py`) and add the following code:

```python
import numpy as np
from pictologics import load_image
from pictologics.preprocessing import discretise_image, resample_image
from pictologics.features.intensity import calculate_intensity_features
from pictologics.features.morphology import calculate_morphology_features
from pictologics.features.texture import calculate_all_texture_features

# 1. Load Data
# ---------------------------------------------------------
print("Loading data...")
image = load_image("path/to/image.nii.gz")
mask = load_image("path/to/mask.nii.gz")

# 2. Preprocessing (Optional but Recommended)
# ---------------------------------------------------------
# Resample to isotropic 1x1x1 mm spacing for standardisation
print("Resampling...")
image = resample_image(image, new_spacing=(1.0, 1.0, 1.0))
mask = resample_image(mask, new_spacing=(1.0, 1.0, 1.0), interpolation="nearest")

# 3. Extract Morphology Features
# ---------------------------------------------------------
print("Calculating morphology...")
morph_features = calculate_morphology_features(mask)

# 4. Extract Intensity Features
# ---------------------------------------------------------
print("Calculating intensity...")
# Get voxels inside the mask
masked_voxels = image.array[mask.array == 1]
intensity_features = calculate_intensity_features(masked_voxels)

# 5. Extract Texture Features
# ---------------------------------------------------------
print("Calculating texture...")
# Texture requires discretisation (binning)
# Fixed Bin Number (FBN) with 32 bins is a common choice
disc_image = discretise_image(image, method="FBN", n_bins=32, roi_mask=mask)

texture_features = calculate_all_texture_features(
    disc_array=disc_image.array,
    mask_array=mask.array,
    n_bins=32
)

# 6. Combine and Print Results
# ---------------------------------------------------------
all_features = {**morph_features, **intensity_features, **texture_features}

print("\n--- Extraction Complete ---")
print(f"Total Features: {len(all_features)}")
print(f"Volume (RNU0): {all_features.get('volume_RNU0', 'N/A')} mm^3")
print(f"Mean Intensity (Q4LE): {all_features.get('mean_intensity_Q4LE', 'N/A')}")
print(f"Contrast (ACUI): {all_features.get('contrast_ACUI', 'N/A')}")
```


---

## Image Filtering (IBSI 2)

Pictologics includes IBSI 2-compliant image filters for creating response maps. Filters can be applied via the pipeline or called directly.

### Filter Types

| Filter | Description | Key Parameters |
|:-------|:------------|:---------------|
| `mean` | Averaging filter | `support` (kernel size) |
| `log` | Laplacian of Gaussian | `sigma_mm`, `truncate` |
| `laws` | Laws' texture kernels | `kernel`, `rotation_invariant`, `compute_energy` |
| `gabor` | Gabor filter bank | `sigma_mm`, `lambda_mm`, `gamma` |
| `wavelet` | Separable wavelets | `wavelet` (db3, haar, coif1), `level`, `decomposition` |
| `simoncelli` | Non-separable wavelet | `level` |
| `riesz` | Riesz transform | `order`, `variant` |

### Using Filters in the Pipeline

```python
from pictologics import RadiomicsPipeline

# Create pipeline with LoG filter
pipeline = RadiomicsPipeline()

cfg = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic"}},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter", "params": {
        "type": "log",
        "sigma_mm": 1.5,
        "truncate": 4.0,
    }},
    {"step": "extract_features", "params": {"families": ["intensity"]}},
]

pipeline.add_config("log_filtered", cfg)
results = pipeline.run("image.nii.gz", "mask.nii.gz", config_names=["log_filtered"])
```

### More Filter Examples

```python
# Laws filter with rotation invariance
{"step": "filter", "params": {
    "type": "laws",
    "kernel": "L5E5E5",
    "rotation_invariant": True,
    "pooling": "max",
    "compute_energy": True,
}}

# Gabor filter
{"step": "filter", "params": {
    "type": "gabor",
    "sigma_mm": 5.0,
    "lambda_mm": 2.0,
    "gamma": 1.5,
    "rotation_invariant": True,
}}

# Daubechies wavelet
{"step": "filter", "params": {
    "type": "wavelet",
    "wavelet": "db3",
    "level": 1,
    "decomposition": "LLH",
    "rotation_invariant": True,
}}
```

### Direct Filter Usage

For granular control, call filter functions directly:

```python
from pictologics import load_image
from pictologics.preprocessing import resample_image, apply_mask
from pictologics.filters import laplacian_of_gaussian, BoundaryCondition
from pictologics.features.intensity import calculate_intensity_features

# Load and preprocess
image = load_image("image.nii.gz")
mask = load_image("mask.nii.gz")
image = resample_image(image, (1.0, 1.0, 1.0), interpolation="cubic")

# Apply LoG filter
response = laplacian_of_gaussian(
    image.array,
    sigma_mm=1.5,
    spacing_mm=image.spacing,
    truncate=4.0,
    boundary=BoundaryCondition.MIRROR
)

# Extract features from filtered image
roi_values = apply_mask(response, mask)
features = calculate_intensity_features(roi_values)
```

!!! tip "IBSI 2 Compliance"
    For IBSI 2 Phase 2 compliance, use `boundary="mirror"` (default) and apply filters after resampling to isotropic spacing.

See the [Pipeline guide](pipeline.md) for the full filter parameter reference table.

---

## Next Steps

*   Read the [Pipeline guide](pipeline.md) for configuration patterns and advanced parameter pass-through.
*   See the [Benchmarks](../benchmarks.md) to understand performance and compliance.
