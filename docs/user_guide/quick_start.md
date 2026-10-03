# Quick Start

This page runs a first feature extraction, and explains the result.

## What Is Radiomics?

Radiomics computes numbers (features) from medical images: for example the mean intensity, the shape and the texture of a region ([Kolossváry et al., J Thorac Imaging 2018;33(1):26-34](https://journals.lww.com/thoracicimaging/fulltext/2018/01000/cardiac_computed_tomography_radiomics__a.5.aspx)). Statistical models and machine learning then use the features, for example to tell tissue types apart.

You give Pictologics an image and a mask of the region of interest (ROI). Pictologics gives the features, as the Image Biomarker Standardisation Initiative ([IBSI](https://theibsi.github.io/)) defines them.

## A First Run

```python
from pictologics import RadiomicsPipeline, format_results, save_results

pipeline = RadiomicsPipeline(load_standard=False)
pipeline.add_config("my_analysis", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture"]}},
])

results = pipeline.run(
    image="ct_scan.nii.gz",
    mask="segmentation.nii.gz",
    subject_id="patient_001",
    config_names=["my_analysis"],
)
row = format_results(results, meta={"subject_id": "patient_001"})
save_results([row], "features.csv")
```

1. `RadiomicsPipeline(load_standard=False)` makes a pipeline without the six standard configurations.
2. `add_config` adds a configuration: a named list of steps. The steps run in their order:
    1. `resample` makes all voxels 1 × 1 × 1 mm, so that images from other scanners compare.
    2. `discretise` puts the intensities of the ROI into 32 bins. The texture features need this step.
    3. `extract_features` computes the intensity, shape (morphology) and texture features.
3. `run` loads the image and the mask, and runs the configurations of `config_names`. The mask goes onto the grid of the image. `subject_id` goes into the processing log.
4. `format_results` makes one table row, and `save_results` writes it to a CSV file.

## The Result

`results` is a dictionary with one `pandas.Series` of features for each configuration:

```python
features = results["my_analysis"]
print(features["mean_intensity_Q4LE"])
```

| Feature key | Family | Description |
|:--|:--|:--|
| `mean_intensity_Q4LE` | Intensity | The mean intensity of the ROI |
| `volume_RNU0` | Morphology | The volume of the ROI, from a mesh of its surface |
| `sphericity_QCFX` | Morphology | How round the ROI is (1 for a sphere) |
| `joint_entropy_TU9B` | Texture (GLCM) | The entropy of the grey level co-occurrence matrix |

- **IBSI codes**: the last part of a feature key is its IBSI code, for example `Q4LE` for the mean intensity. The code identifies the feature in the [IBSI reference manual](https://ibsi.readthedocs.io/en/latest/). A number after the code tells the variant, for example `_10` in `volume_at_intensity_fraction_0.10_BC2M_10`.
- **Complete rows**: each configuration gives all its feature names. A feature that cannot be computed (for example because a step removed all ROI voxels) is `NaN`. So the rows of many cases have the same columns.
- **Columns**: in the CSV file, each column is `{configuration}__{feature}`, for example `my_analysis__mean_intensity_Q4LE`.

## Images and Masks

- **The image** holds the intensities, for example the CT values in Hounsfield units (HU).
- **The mask** (a segmentation) marks the ROI: 1 in the ROI, 0 outside. A mask can hold more labels, for example 1 for the liver and 2 for the spleen. Each voxel that is not 0 is in the ROI. To use one label, add a `binarize_mask` step, or run each label with `run_rois` (see [The Pipeline](pipeline.md#many-rois-run_rois)).
- **No mask**: without a mask, the whole image is the ROI.
- **File types**: NIfTI, NRRD, MetaImage, DICOM folders and files, DICOM SEG and RTSTRUCT masks (see [Data Loading](data_loading.md)).

## Images with a Sentinel Value

Some images hold a fixed value, a **sentinel value** such as -2048, where they have no data: for example outside the field of view, or around a cropped lesion. A resampling or a filter near these voxels mixes the sentinel into the voxels with data. The `source_mode` of a configuration keeps them out:

```python
pipeline.add_config(
    "padded_ct",
    [
        {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
        {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},
        {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
    ],
    source_mode="auto",
    sentinel_value=-2048,  # the padding value of these images
)
results = pipeline.run("lesion_export.nii.gz", config_names=["padded_ct"])  # no mask
```

- `source_mode="auto"` with `sentinel_value` keeps the voxels with the sentinel out of the resampling and the filters. After the resampling, these voxels are also out of the ROI.
- Without `sentinel_value`, `"auto"` looks for common sentinel values, and a warning gives the value that it found.
- The `resegment` step keeps the ROI voxels in an intensity range.

See [Source Modes and Sentinel Values](pipeline.md#source-modes-and-sentinel-values) for all modes.

## The Steps

| Step | Does | Use it |
|:--|:--|:--|
| `resample` | Gives all images the same voxel spacing | Almost always, so that the features compare |
| `resegment` | Keeps the ROI voxels in an intensity range, for example -100 to 400 HU | To remove other tissues or sentinel values |
| `filter_outliers` | Removes the ROI voxels far from the ROI mean | To remove noise or artefacts |
| `keep_largest_component` | Keeps the largest connected part of the ROI | When the mask has small loose parts |
| `grow_mask` | Grows or shrinks the mask, or keeps a ring at its edge (in mm) | For the tissue around a lesion or a vessel |
| `binarize_mask` | Selects labels of a mask | When the mask holds more structures |
| `normalise` | Normalises the intensities (z-score or percentiles) | For MR images and other images without fixed units |
| `discretise` | Puts the intensities into bins (FBN or FBS) | Before the texture features |
| `filter` | Applies an IBSI 2 filter (LoG, Gabor, wavelets and more) | For the features of filtered images |
| `extract_features` | Computes the features | As the last step |

See [Pipeline Steps](pipeline_steps.md) for the parameters of each step.

## Ready-Made Configurations

`RadiomicsPipeline()` starts with six standard configurations: they resample to 0.5 mm and compute all feature families, with FBN (8, 16 or 32 bins) or FBS (bins of 8, 16 or 32 HU from -1000 HU, for CT).

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
results = pipeline.run("ct_scan.nii.gz", "segmentation.nii.gz", config_names=["standard_fbs_16"])
results = pipeline.run("ct_scan.nii.gz", "segmentation.nii.gz", config_names=["all_standard"])  # all six

coronary = RadiomicsPipeline.from_template("coronary")  # 30 configurations for coronary plaque
print(coronary.list_configs())
```

Always give `config_names`: without it, `run()` runs every configuration of the pipeline, also the six standard ones, with a warning. See [The Pipeline](pipeline.md#which-configurations-run).

## Next Steps

- [Data Loading](data_loading.md): DICOM series, NIfTI files, SEG and RTSTRUCT masks, PET in SUV.
- [The Pipeline](pipeline.md): configurations, masks, source modes, many ROIs and many cases.
- [Pipeline Steps](pipeline_steps.md): each step and its parameters.
- [Results and Logs](results.md): tables, the feature catalog and the processing log.
- [Image Filtering](image_filtering.md): the IBSI 2 filters.
- [Configuration & Reproducibility](configurations.md): save and share configurations.
- [Cookbook](cookbook.md) and the tutorials: complete scripts for common tasks.
