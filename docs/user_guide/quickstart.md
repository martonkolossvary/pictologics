# Quick Start Guide

This guide will walk you through the process of extracting radiomic features from a medical image using **Pictologics**.

## Prerequisites

Ensure you have installed the package:
```bash
pip install pictologics
```

You will need:
1.  A medical image (e.g., `image.nii.gz`, DICOM folder, or a single DICOM file).
2.  Optionally a corresponding mask/segmentation (e.g., `mask.nii.gz`).

!!! note
    **Mask is optional**
    If you omit `mask` (or pass `mask=None` / `mask=""`), Pictologics treats the entire image as the ROI by
    generating a full (all-ones) mask internally. This can be useful for certain workflows (see the
    [Case examples](case_examples.md) page), but it may not be scientifically appropriate for all studies.

## Method 1: The Radiomics Pipeline (Recommended)

For reproducible research and standardisation, we recommend using the `RadiomicsPipeline`. This ensures that all preprocessing steps (resampling, resegmentation, discretisation) are applied consistently.

Pictologics includes a set of **Standard Configurations** commonly used in radiomic analyses. You can run all of them with a single command.

!!! note
    **Default performance behavior**
    The built-in `standard_*` configurations disable the two most time-intensive intensity extras by default:
    spatial intensity (Moran’s I / Geary’s C) and local intensity peak features.
    
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

# 3. Format and Export Results
print(f"Successfully ran {len(results)} configurations.")

#  Option A: Get a flat dictionary (Wide Format)
#  Inject subject ID or other metadata directly into the row
row = format_results(
    results, 
    fmt="wide", 
    meta={"subject_id": "Subject_001", "group": "control"}
)

#  Option B: Save to CSV immediately
#  save_results handles list of dicts/dataframes automatically
save_results([row], "results.csv")
```

### Working with results

The `pictologics.results` module makes it easy to handle outputs, especially for batch processing.

1.  **Wide Format**: One row per case, huge column count (e.g. `standard_fbn_32__original_glcm_Energy`).
    ```python
    row = format_results(results, fmt="wide", meta={"id": "case1"})
    ```

2.  **Long Format**: Tidy data, one row per feature. Great for analysis in R/Seaborn.
    ```python
    df = format_results(results, fmt="long", meta={"id": "case1"}, output_type="pandas")
    # Columns: [id, config, feature_name, value]
    ```

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
*   **Texture** requires discretisation. If you request `"texture"` without discretising first, the pipeline can temporarily discretise **only** if you provide `texture_settings` in `extract_features` (e.g., `{"method": "FBN", "n_bins": 32}`). *However, doing this repeatedly for every texture family is slower than a single `discretise` step.*
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

## Next Steps

*   Read the [Pipeline guide](pipeline.md) for configuration patterns and advanced parameter pass-through.
*   See the [Benchmarks](../benchmarks.md) to understand performance and compliance.

