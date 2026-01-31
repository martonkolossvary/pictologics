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

!!! tip "Reproducible Research"
    Save your configuration settings for reproducibility using `pipeline.save_configs("config.yaml")`.
    See the **[Predefined Configurations](predefined_configurations.md)** guide for details on exporting,
    sharing, and version-controlling your pipeline configurations.

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

If you want to understand the underlying process or need granular control over specific functions, you can call the feature extraction functions directly. This example mirrors a **complete** radiomics workflow, including all preprocessing steps, filtering, and feature families available in the pipeline.

### The "Hard" Way (Manual Extraction)

This script performs 12 distinct steps to extract all feature families.

```python
import numpy as np
from pictologics import load_image
from pictologics.preprocessing import (
    resample_image,
    resegment_mask,
    filter_outliers,
    discretise_image,
    apply_mask
)
from pictologics.features.intensity import (
    calculate_intensity_features,
    calculate_intensity_histogram_features,
    calculate_ivh_features,
    calculate_spatial_intensity_features,
    calculate_local_intensity_features
)
from pictologics.features.morphology import calculate_morphology_features
from pictologics.features.texture import calculate_all_texture_features

# 1. Load Data
# ---------------------------------------------------------
print("1. Loading data...")
image = load_image("image.nii.gz")
mask = load_image("mask.nii.gz")

# 2. Preprocessing & Standardization
# ---------------------------------------------------------
print("2. Resampling to 1x1x1 mm...")
image = resample_image(image, new_spacing=(1.0, 1.0, 1.0), interpolation="linear")
mask = resample_image(mask, new_spacing=(1.0, 1.0, 1.0), interpolation="nearest")

print("3. Rounding intensities (for consistent binning)...")
image.array = np.round(image.array)

print("4. Binarizing mask (thresholding)...")
mask.array = (mask.array > 0.5).astype(np.uint8)

print("5. Resegmenting (range filtering)...")
# Exclude values outside typical range (e.g. -1000 to 400 HU for lung)
mask = resegment_mask(image, mask, range_min=-1000, range_max=400)

print("6. Outlier filtering...")
# Remove statistical outliers (mean +/- 3 sigma) from the mask
mask = filter_outliers(image, mask, sigma=3.0)

# 3. Discretisation
# ---------------------------------------------------------
print("7. Discretising image (FBN 32 bins)...")
# Essential for Texture and Histogram features
disc_image = discretise_image(
    image, 
    method="FBN", 
    n_bins=32, 
    roi_mask=mask
)

# 4. Feature Extraction (All Families)
# ---------------------------------------------------------
print("8. Calculating Morphology...")
morph_feat = calculate_morphology_features(mask, image=image, intensity_mask=mask)

print("9. Calculating Intensity (First Order)...")
roi_values = apply_mask(image, mask)
int_feat = calculate_intensity_features(roi_values)

print("10. Calculating Histogram...")
# Uses discretised values
hist_feat = calculate_intensity_histogram_features(apply_mask(disc_image, mask))

print("11. Calculating IVH (Intensity Volume Histogram)...")
# Can use raw or discretised values (usually raw)
ivh_feat = calculate_ivh_features(roi_values, bin_width=10.0)

print("12. Calculating Texture (GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM)...")
tex_feat = calculate_all_texture_features(
    disc_array=disc_image.array,
    mask_array=mask.array,
    n_bins=32
)

print("13. Calculating Advanced Intensity...")
# Very computationally intensive
sp_feat = calculate_spatial_intensity_features(image, mask)
loc_feat = calculate_local_intensity_features(image, mask)

# 5. Combine Results
# ---------------------------------------------------------
all_features = {
    **morph_feat, **int_feat, **hist_feat, 
    **ivh_feat, **tex_feat, **sp_feat, **loc_feat
}

print(f"\n--- Extraction Complete: {len(all_features)} features calculated ---")
```

### The "Easy" Way (Pipeline Implementation)

The `RadiomicsPipeline` accomplishes the exact same logical workflow in a fraction of the code, with automatic image routing (handling raw vs. discretised copies transparently).

```python
from pictologics import RadiomicsPipeline

# Define the exact same workflow configuration
config = [
    # Preprocessing
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "round_intensities": True}},
    {"step": "binarize_mask", "params": {"threshold": 0.5}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
    {"step": "filter_outliers", "params": {"sigma": 3.0}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    
    # Feature Extraction (All families)
    {"step": "extract_features", "params": {
        "families": ["intensity", "morphology", "histogram", "ivh", "texture"],
        # Enable advanced metrics
        "include_spatial_intensity": True,
        "include_local_intensity": True,
        # Customize specific family parameters
        "ivh_params": {"bin_width": 10.0}
    }}
]

# Run it
pipeline = RadiomicsPipeline().add_config("comprehensive", config)
results = pipeline.run("image.nii.gz", "mask.nii.gz", config_names=["comprehensive"])

print(f"--- Extraction Complete: {len(results['comprehensive'])} features calculated ---")
```


---

## Image Filtering (IBSI 2)

Pictologics includes **IBSI 2-compliant image filters** (LoG, Wavelets, Gabor, etc.) for creating response maps.

!!! info "See Image Filtering Guide"
    Detailed documentation on available filters, parameters, and examples can be found in the [Image Filtering](image_filtering.md) guide.

Filters are typically applied **after** resampling and **before** feature extraction. The pipeline handles this order automatically.

