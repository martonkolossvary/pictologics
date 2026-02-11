# Quick Start

Get from zero to extracting radiomic features in under 5 minutes. This page walks you through the core concepts and a minimal working example so you can start analyzing medical images right away.


## What Is Radiomics?

Radiomics is the process of extracting quantitative features — such as texture, shape, and intensity statistics — from medical images ([Kolossváry, Márton et al. J Thorac Imaging. 2018 Jan;33(1):26-34](https://journals.lww.com/thoracicimaging/fulltext/2018/01000/cardiac_computed_tomography_radiomics__a.5.aspx)). These features can then be used in statistical models, machine learning, or clinical research to characterize tissue properties that may not be visible to the human eye.

**Pictologics** automates this process: you provide an image and (optionally) a segmentation mask, and it returns a set of reproducible, IBSI-compliant features.


## Minimal Example

The fastest way to get started is with a **pipeline**. A pipeline is a sequence of processing steps — such as resampling, intensity binning, and feature extraction — that are applied to your image in order. Think of it as a recipe: you define the steps once, and then apply the same recipe to any number of images.

```python
from pictologics import load_image, RadiomicsPipeline, format_results
from pictologics.results import save_results

# Step 1: Load your data
# Pictologics accepts NIfTI (.nii, .nii.gz) and DICOM files.
# The "image" is the medical scan (e.g., a CT or MRI volume).
# The "mask" is a binary segmentation that defines the region of interest (ROI)
# — the specific area in the image you want to analyze (e.g., a tumor).
image = load_image("path/to/ct_scan.nii.gz")
mask = load_image("path/to/segmentation.nii.gz")

# Step 2: Create a pipeline and add a configuration
# A configuration is a named set of processing steps.
# Here we define three steps:
#   1. Resample the image to 1mm isotropic voxel spacing (standardizes resolution)
#   2. Discretise intensities into 32 bins (required for texture analysis)
#   3. Extract intensity, morphology (shape), and texture features
pipeline = RadiomicsPipeline()
pipeline.add_config("my_analysis", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {
        "families": ["intensity", "morphology", "texture"],
    }},
])

# Step 3: Run the pipeline
# The pipeline applies each step in order: resample → discretise → extract.
# "subject_id" is an optional label used to identify this case in the output.
results = pipeline.run(image=image, mask=mask, subject_id="patient_001")

# Step 4: Save the extracted features to a CSV file
# Each feature becomes a column; each subject becomes a row.
formatted = format_results(results, fmt="wide", output_type="dict")
save_results(formatted, "features.csv")
```

!!! tip "What does the output look like?"
    The result is a dictionary mapping configuration names to feature sets. Each feature set is a
    `pandas.Series` with descriptive names like:

    | Feature Name | Family | Description |
    |:-------------|:-------|:------------|
    | `mean_intensity_Q4LE` | Intensity | Mean value within the ROI |
    | `volume_RNU0` | Morphology | Volume of the ROI computed from a mesh surface |
    | `sphericity_QCFX` | Morphology | How sphere-like the ROI shape is (1.0 = perfect sphere) |
    | `joint_entropy_TU9B` | Texture (GLCM) | Entropy of the grey-level co-occurrence matrix |

    **The trailing 4-character code** (e.g., `Q4LE`, `RNU0`) is the official
    [IBSI feature identifier](https://theibsi.github.io/). Identifiers more than 4 letters long are specific to Pictologics and are not part of the IBSI standard (for example due to typos or missing abbreviations in the documentation). The
    [Image Biomarker Standardisation Initiative (IBSI)](https://ibsi.readthedocs.io/en/latest/) assigns a
    unique alphanumeric code to every standardised radiomic feature. Pictologics appends this
    code to every feature name so you can always trace a result back to its exact IBSI definition
    — for example, `mean_intensity_Q4LE` corresponds to IBSI feature **Q4LE** ("Mean intensity").

    Saving to CSV produces a table where each row is one subject and each column is one feature.

## Two Input Modes

Pictologics supports two approaches for defining which voxels to analyze. The right choice depends on how your data is structured.

| Mode | Input Files | When to Use |
|:-----|:------------|:------------|
| **Image + Mask** | 2 files (image + segmentation) | Standard radiomics with explicit ROI masks |
| **Sentinel Values** | 1 file (embedded background marker) | Pre-cropped lesion exports, coronary artery CTs |

### Mode 1: Image + Mask (Traditional)

This is the most common setup. You have **two separate files**:

- An **image** containing raw intensity values (e.g., CT Hounsfield Units or MRI signal intensities).
- A **mask** (also called a segmentation) that defines the region of interest (ROI). In the simplest case this is a binary volume (1 = inside the ROI, 0 = outside), but it can also contain **multiple integer labels** encoding different structures (e.g., 1 = left ventricle, 2 = right ventricle). Pictologics can select, combine, or iterate over these labels using the `binarize_mask` pipeline step — see [Data Loading → Combining Segments](data_loading.md#combining-specific-segments-into-a-binary-mask) and the [Cookbook](cookbook.md) for detailed examples.

```python
# Load image and mask from separate files
image = load_image("ct_scan.nii.gz")       # The medical scan
mask = load_image("segmentation.nii.gz")   # Binary ROI (1 = analyze, 0 = ignore)

# Run the pipeline — features are computed only within the mask
results = pipeline.run(image=image, mask=mask)
```

### Mode 2: Sentinel Values (Single-File)

Some workflows produce a **single file** where the ROI is embedded in the data itself. Valid voxels contain real intensity values, while background or excluded regions are filled with a **sentinel value** — a fixed, artificial number (such as -2048) that signals "this voxel is not real data."

This is common in pre-cropped lesion exports and research datasets where masking is embedded rather than stored separately.

If left untreated, sentinel values corrupt resampling (interpolation bleeds the artificial value into neighbors) and filtering (convolution kernels mix sentinel values into the response). Handling sentinel values properly requires **two complementary mechanisms**:

1. **`source_mode="auto"`** creates a *source mask* that protects **resampling and filtering only**. It applies masked interpolation and normalized convolution so sentinel voxels don't corrupt valid neighbors. However, it does **not** change the ROI — sentinel voxels still remain in the region used for feature extraction.
2. **`resegment`** restricts the **ROI** to a valid intensity range, excluding sentinel voxels from **feature extraction**. Without this step, sentinel values would bias intensity statistics, texture matrices, and all other computed features.

You typically need both:

```python
# Configure the pipeline with automatic sentinel detection.
# source_mode="auto" tells the pipeline to scan for common sentinel values
# (e.g., -2048, -1024, 0) and apply protected resampling/filtering if one is found.
pipeline.add_config("sentinel_aware", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    # resegment is still needed to exclude sentinels from the ROI for feature extraction
    {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
], source_mode="auto")

# Run without a mask — the ROI is derived from non-sentinel voxels via resegment
results = pipeline.run(image=image)
```

| Modality | Common Sentinel | Meaning |
|:---------|:----------------|:--------|
| CT | -2048 HU | Outside value range |
| CT | -1024 HU | Missing data |
| MRI | 0 | Background |

!!! tip "Source Mode Options"
    - `"full_image"` (default) — All voxels are assumed valid. Use this for standard image + mask workflows.
    - `"auto"` — Automatically detect and handle sentinel values. Protects resampling and filtering. Combine with `resegment` to also exclude sentinels from the ROI.
    - `"roi_only"` — Explicitly enable sentinel handling. Pair with `sentinel_value=-2048` (or your known value) for deterministic behavior.

## Pipeline Steps at a Glance

Steps execute **in the order you define them**, so you have full control over the processing sequence. Here are the most commonly used steps:

| Step | Purpose | When to Use |
|:-----|:--------|:------------|
| `resample` | Standardize voxel spacing across patients | Almost always — ensures features are comparable |
| `resegment` | Restrict ROI to an intensity range (e.g., [-100, 400] HU) | When you need to exclude outliers or sentinel values |
| `keep_largest_component` | Remove small disconnected mask fragments | When your mask has noise or satellite lesions |
| `binarize_mask` | Select specific labels from a multi-label mask | When your mask encodes multiple structures |
| `discretise` | Bin intensities for texture features (FBN or FBS) | **Required** before `extract_features` with texture |
| `filter` | Apply IBSI 2 image filters (LoG, Gabor, Wavelets, etc.) | For filtered radiomics / response map analysis |
| `extract_features` | Calculate radiomic features from the processed ROI | The final step — produces the output features |

!!! note "Full Step Reference"
    For a complete list of all steps, their parameters, and usage examples, see the
    **[Pipeline & Preprocessing](pipeline.md)** guide.

## Using Predefined Configurations

If you don't want to define steps manually, Pictologics includes **6 standard configurations** that are ready to use. They cover common discretisation strategies with sensible defaults:

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()

# Run a single predefined configuration
# "standard_fbn_32" uses 0.5mm isotropic resampling and 32 fixed bins
results = pipeline.run(
    image="ct_scan.nii.gz",
    mask="segmentation.nii.gz",
    config_names=["standard_fbn_32"],
)

# Or run all 6 standard configurations at once
all_results = pipeline.run(
    image="ct_scan.nii.gz",
    mask="segmentation.nii.gz",
    config_names=["all_standard"],
)
```

See **[Configuration & Reproducibility](configurations.md)** for the full list and how to export/share configurations.

## Next Steps

- **[Data Loading](data_loading.md)** — Loading DICOM series, NIfTI files, SEG objects, and merging masks
- **[Pipeline & Preprocessing](pipeline.md)** — Full reference for all preprocessing steps and feature extraction
- **[Image Filtering](image_filtering.md)** — IBSI 2 compliant filter suite
- **[Configuration & Reproducibility](configurations.md)** — Predefined configs, YAML/JSON export, sharing
- **[Cookbook](cookbook.md)** — End-to-end batch processing scripts and real-world recipes
