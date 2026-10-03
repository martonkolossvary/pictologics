# Cookbook

Complete scripts for common tasks. Change the paths, and each script runs as it is. The [tutorials](../tutorials/batch.md) explain single topics in more detail.

## Find a Recipe

| Task | Recipe |
|:--|:--|
| Images without mask files, for example lesion exports with -2048 outside the lesion | [1. Images without masks](#1-images-without-masks) |
| DICOM cases with a DICOM SEG or RTSTRUCT | [2. DICOM cases with a segmentation](#2-dicom-cases-with-a-segmentation) |
| Many mask files for one image | [3. Many masks for one image](#3-many-masks-for-one-image) |
| Each segment of a SEG, an RTSTRUCT or a label map as its own ROI | [4. Each segment as its own ROI](#4-each-segment-as-its-own-roi) |
| Features of filtered images (IBSI 2) | [5. Filtered radiomics](#5-filtered-radiomics) |
| Many discretisations of one image | [6. Many discretisations](#6-many-discretisations) |
| A data dictionary of the features | [7. A data dictionary](#7-a-data-dictionary) |
| The fat or tissue around an ROI | [Rings Around an ROI](../tutorials/rings.md) |
| The phases of a cardiac CT | [Cardiac CT Phases](../tutorials/cardiac_phases.md) |
| PET images in SUV | [PET in SUV](../tutorials/pet_suv.md) |
| MR images | [MR Radiomics with Normalisation](../tutorials/mr_normalisation.md) |
| Quality images of the masks | [Utilities](utilities.md#quality-images-of-many-cases) |
| A study from a DICOM archive: find the series, run the cases, join the clinical data | [From a DICOM Archive to a Study Table](../tutorials/dicom_study.md) |
| Masks from nnU-Net, TotalSegmentator, 3D Slicer, ITK-SNAP or a treatment planning system | [Masks from Other Tools](../tutorials/masks.md) |

## Common Mistakes

1. **No `config_names`**: `RadiomicsPipeline()` holds the six standard configurations. `run()` without `config_names` runs them all, and `save_configs()` without `config_names` saves them all. Give `config_names`, or start with `RadiomicsPipeline(load_standard=False)`.
2. **FBS without a start**: an FBS `discretise` step needs `min_val`, or a `resegment` step with `range_min` before it. A filter or a `normalise` step cancels that start: after them, give `min_val`.
3. **`roi_only` with a sentinel value**: `source_mode="roi_only"` does not use `sentinel_value`. For a known padding value, use `source_mode="auto", sentinel_value=-2048`.
4. **A mask `Image` on another grid**: `run()` places a mask path on the grid of the image, but an `Image` mask must already have that grid. Load a mask with `load_image(path, reference_image=image)`, or give the path to `run()`.
5. **Label masks**: each voxel that is not 0 is in the ROI. For one label, add a `binarize_mask` step. For each label, use `run_rois` (recipe 4).
6. **Morphology after a filter**: the integrated intensity (99N0) and the centre of mass shift (KLMA) read the image, so after a filter they read the response map. Compute the morphology in a configuration without the filter (recipe 5).
7. **A name that exists**: `add_config` with the name of a configuration in the pipeline replaces that configuration. Do not use the names of the standard configurations, such as `standard_fbn_32`, for your own.
8. **Workers without a guard**: keep `run_batch(..., workers=4)` in `if __name__ == "__main__":`. The workers import your script again.

## 1. Images Without Masks

**The task**: a folder of NIfTI lesion exports. Each file holds a CT of one lesion, with -2048 outside the lesion, and there is no mask file. The study needs the intensity and shape features, and the texture features of four discretisations.

```python
from pathlib import Path

from pictologics import RadiomicsPipeline, save_results

STEPS = [
    {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5), "round_intensities": True}},
    {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},  # HU
    {"step": "keep_largest_component", "params": {}},
]
DISCRETISATIONS = {
    "fbn_8": {"method": "FBN", "n_bins": 8},
    "fbn_16": {"method": "FBN", "n_bins": 16},
    "fbs_8": {"method": "FBS", "bin_width": 8},  # bins from -100 HU, the resegment start
    "fbs_16": {"method": "FBS", "bin_width": 16},
}


def build_pipeline():
    pipeline = RadiomicsPipeline(load_standard=False)
    padding = {"source_mode": "auto", "sentinel_value": -2048}  # the value outside the lesion
    pipeline.add_config("orig", STEPS + [
        {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
    ], **padding)
    for name, discretise in DISCRETISATIONS.items():
        pipeline.add_config(name, STEPS + [
            {"step": "discretise", "params": discretise},
            {"step": "extract_features", "params": {"families": ["texture", "histogram", "ivh"]}},
        ], **padding)
    return pipeline


if __name__ == "__main__":
    pipeline = build_pipeline()
    cases = [
        {"subject_id": path.name.removesuffix(".gz").removesuffix(".nii"), "image": str(path)}
        for path in sorted(Path("lesion_exports").glob("*.nii*"))
    ]
    table = pipeline.run_batch(cases, "results", workers=2)
    print(table.loc[table["status"] != "completed", ["subject_id", "status", "error"]])
    save_results(table, "results/features.csv")
```

- **No mask**: a case without `mask` uses the whole image as the ROI. The `resample` step takes the voxels of -2048 out of the ROI (`source_mode="auto"` with `sentinel_value`), the `resegment` step keeps -100 to 3000 HU, and `keep_largest_component` keeps the lesion only.
- **A compact table**: the `orig` configuration gives the intensity and morphology features. The other four give only the features that read the discretisation. So the table has each feature one time, for example `orig__mean_intensity_Q4LE` and `fbn_8__joint_entropy_TU9B`.
- **Shared work**: the five configurations start with the same three steps, so the steps run one time for each image.
- **An unknown padding value**: leave out `sentinel_value`. The pipeline then finds the value of each image, with a warning.
- **The result files**: `results/cases/<subject_id>.json` holds the features and the log of each case. A second run skips the cases that are done.

## 2. DICOM Cases with a Segmentation

**The task**: one folder for each case, with a DICOM image series and its DICOM SEG (or RTSTRUCT) file, each at any depth:

```text
cases/
  p001/
    Image/          the DICOM series
    Segmentation/   one SEG or RTSTRUCT file
  p002/
    ...
```

```python
from pathlib import Path

from pictologics import RadiomicsPipeline, save_results

EXTRACT = {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"]}}


def segmentation_file(folder):
    """The one DICOM file of a segmentation folder, at any depth."""
    files = sorted(Path(folder).rglob("*.dcm"))
    if len(files) != 1:
        raise ValueError(f"{folder} holds {len(files)} DICOM files, not 1")
    return str(files[0])


if __name__ == "__main__":
    pipeline = RadiomicsPipeline(load_standard=False)
    for name, discretise in {
        "fbs_25": {"method": "FBS", "bin_width": 25, "min_val": -1000},
        "fbn_64": {"method": "FBN", "n_bins": 64},
    }.items():
        pipeline.add_config(name, [
            {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
            {"step": "discretise", "params": discretise},
            EXTRACT,
        ])

    cases = [
        {
            "subject_id": case.name,
            "image": str(case / "Image"),
            "image_options": {"recursive": True},  # the series at any depth
            "mask": segmentation_file(case / "Segmentation"),
        }
        for case in sorted(Path("cases").iterdir())
        if case.is_dir()
    ]
    table = pipeline.run_batch(cases, "results", workers=4)
    save_results(table, "results/features.csv")

    # The long format: one row for each feature
    features = [column for column in table.columns if "__" in column]
    long = table.melt(id_vars=["subject_id"], value_vars=features, var_name="column", value_name="value")
    long[["config", "feature_key"]] = long["column"].str.split("__", n=1, expand=True)
    save_results(long.drop(columns="column"), "results/features_long.csv")
```

- **The mask**: the pipeline loads the SEG or the RTSTRUCT onto the grid of the image. All segments (or ROIs) of the file together make the ROI, because each voxel that is not 0 is in the ROI. For each segment on its own, see recipe 4.
- **Phases**: for a series with more than one phase (for example a cardiac CT), give the phase in `image_options`, for example `{"recursive": True, "dataset_index": 4}`. `get_dicom_phases(folder, recursive=True)` lists the phases (see [Cardiac CT Phases](../tutorials/cardiac_phases.md)).
- **Offsets**: a mask origin a small part of a voxel off the image grid gives a warning, and the mask snaps to the nearest voxel. Give `mask_subvoxel_warning_threshold` in a case to change the limit (see [Sub-Voxel Alignment](data_loading.md#sub-voxel-alignment-and-overlap)).

**More than one SEG file in a case**: merge the files into one mask file first, and give that file as the `mask` of the case.

```python
from pictologics import load_and_merge_images, load_image, save_image

for case in sorted(Path("cases").iterdir()):
    image = load_image(str(case / "Image"), recursive=True)
    seg_files = sorted((case / "Segmentation").rglob("*.dcm"))
    mask = load_and_merge_images(seg_files, reference_image=image, reposition_to_reference=True, binarize=True)
    save_image(mask, Path("merged_masks") / f"{case.name}.nii.gz")
```

`load_and_merge_images` takes SEG files and other image files, but not RTSTRUCT files. For an RTSTRUCT, use `load_rtstruct` (see [Data Loading](data_loading.md#dicom-rtstruct)).

## 3. Many Masks for One Image

**The task**: one folder holds the images and their masks as NIfTI files: `CASE001_IMG.nii.gz`, and `CASE001_MASK1.nii.gz`, `CASE001_MASK2.nii.gz` and more for each image. The masks of an image together make one ROI. The study needs the features of six discretisations, without other steps, as long-format JSON.

```python
from pathlib import Path

from pictologics import RadiomicsPipeline, format_results, load_and_merge_images, load_image, save_results

EXTRACT = {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"]}}

pipeline = RadiomicsPipeline(load_standard=False)
for n_bins in (8, 16, 32):
    pipeline.add_config(f"fbn_{n_bins}", [{"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}}, EXTRACT])
for bin_width in (8, 16, 32):
    pipeline.add_config(f"fbs_{bin_width}", [
        {"step": "discretise", "params": {"method": "FBS", "bin_width": bin_width, "min_val": -1000}},  # from air
        EXTRACT,
    ])

folder = Path("nifti_folder")
tables = []
for image_path in sorted(folder.glob("*_IMG.nii*")):
    subject_id = image_path.name.split("_IMG")[0]
    mask_paths = sorted(folder.glob(f"{subject_id}_MASK*.nii*"))
    if not mask_paths:
        raise ValueError(f"No masks for {subject_id}")
    image = load_image(str(image_path))
    mask = load_and_merge_images(mask_paths, reference_image=image, reposition_to_reference=True, binarize=True)
    results = pipeline.run(image, mask, subject_id=subject_id, config_names=pipeline.list_configs())
    tables.append(format_results(results, fmt="long", meta={"subject_id": subject_id}, output_type="pandas"))
    pipeline.clear_log()

save_results(tables, "results/features_long.json")
```

- `binarize=True` makes each mask 0 and 1 before the merge, so the merged mask is one ROI. With `relabel_masks=True` in place of `binarize`, each file gets its own label (1, 2, 3, ...), for `run_rois`.
- The six configurations differ only in the discretisation, so the intensity and morphology features are computed one time and copied (see recipe 6).
- The JSON file holds a list of rows: `subject_id`, `config`, `feature_key` and `value`, with `null` for `NaN`.

## 4. Each Segment as Its Own ROI

**The task**: each case holds a DICOM series and a SEG file with segments, for example the liver, the spleen and the kidneys. The study needs a row for each segment.

```python
from pathlib import Path

from pictologics import RadiomicsPipeline, save_results
from pictologics.loaders import get_segment_info

if __name__ == "__main__":
    pipeline = RadiomicsPipeline(load_standard=False)
    pipeline.add_config("fbn_32", [
        {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
        {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram"]}},
    ])

    cases = []
    for case in sorted(Path("cases").iterdir()):
        seg_file = str(case / "segmentation.dcm")
        labels = {s["segment_label"]: s["segment_number"] for s in get_segment_info(seg_file)}
        cases.append({
            "subject_id": case.name,
            "image": str(case / "Image"),
            "image_options": {"recursive": True},
            "rois": seg_file,  # a label map: each segment has its number
            "labels": labels,  # the ROI names in the table
        })
    table = pipeline.run_batch(cases, "results", workers=4)
    save_results(table, "results/features_by_segment.csv")  # one row for each segment
```

- **The label map**: the pipeline loads the SEG as one label image, in which each voxel holds the number of its segment. `run_rois` (here through `run_batch`) gives each segment the features of `run()` with a mask of that segment alone, and it is much faster than one run for each segment.
- **The table**: one row for each segment, with its name in the column `roi`. The names in `labels` must differ.
- **An RTSTRUCT**: give its path as `rois`. `get_segment_info` gives its ROI names and numbers in the same form.
- **A label map file**: give a NIfTI label map as `rois`, with `labels={"liver": 1, "spleen": 2}` (or no `labels` for all labels).

**Overlapping segments** (for example a tumour inside an organ) do not fit in one label image. Load each segment as its own mask, and run each:

```python
from pictologics import format_results, load_image, load_seg

image = load_image("cases/p001/Image", recursive=True)
masks = load_seg("cases/p001/segmentation.dcm", reference_image=image, combine_segments=False)  # {number: mask}
names = {s["segment_number"]: s["segment_label"] for s in get_segment_info("cases/p001/segmentation.dcm")}
rows = [
    format_results(pipeline.run(image, mask, config_names=["fbn_32"]), meta={"subject_id": "p001", "roi": names[number]})
    for number, mask in masks.items()
]
save_results(rows, "results/p001_segments.csv")
```

## 5. Filtered Radiomics

**The task**: the features of IBSI 2 filter response maps of a CT: LoG at two scales, a Laws energy map, a Gabor filter and a wavelet, with the preprocessing of IBSI 2 Phase 2 (configuration B).

```python
import math

from pictologics import RadiomicsPipeline, format_results, save_results

PREPROCESS = [  # IBSI 2 Phase 2, configuration B
    {"step": "resample", "params": {
        "new_spacing": (1.0, 1.0, 1.0), "interpolation": "cubic", "mask_interpolation": "linear", "mask_threshold": 0.5,
    }},
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},  # HU, before the filters
]
FILTERS = {
    "log_1.5": {"type": "log", "sigma_mm": 1.5},
    "log_3": {"type": "log", "sigma_mm": 3.0},
    "laws_e5": {"type": "laws", "kernel": "L5E5E5", "rotation_invariant": True, "pooling": "max",
                "compute_energy": True, "energy_distance": 7},
    "gabor": {"type": "gabor", "sigma_mm": 5.0, "lambda_mm": 2.0, "gamma": 1.5, "rotation_invariant": True,
              "delta_theta": math.pi / 4, "pooling": "average"},
    "wavelet_hhh": {"type": "wavelet", "wavelet": "db3", "level": 1, "decomposition": "HHH",
                    "rotation_invariant": True, "pooling": "average"},
}

pipeline = RadiomicsPipeline(load_standard=False)
pipeline.add_config("orig", PREPROCESS + [
    {"step": "extract_features", "params": {"families": ["intensity", "morphology"]}},
])
for name, params in FILTERS.items():
    pipeline.add_config(name, PREPROCESS + [
        {"step": "filter", "params": params},
        {"step": "extract_features", "params": {"families": ["intensity"]}},
    ])
pipeline.add_config("log_1.5_fbn_32", PREPROCESS + [  # texture of a response map
    {"step": "filter", "params": FILTERS["log_1.5"]},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {"step": "extract_features", "params": {"families": ["texture", "histogram"]}},
])

results = pipeline.run("ct.nii.gz", "gtv.nii.gz", subject_id="p001", config_names=pipeline.list_configs())
save_results([format_results(results, meta={"subject_id": "p001"})], "filtered_features.csv")
print(results["log_1.5"]["mean_intensity_Q4LE"])
```

- **The resegment step** comes before the filters, so it selects the voxels by HU.
- **Morphology** comes from the `orig` configuration, without a filter (see [Common Mistakes](#common-mistakes), item 6).
- **Texture of a response map**: use FBN, or FBS with `min_val`, because the filter cancels the FBS start of the `resegment` step.
- **Shared work**: the configurations start with the same three steps, so the steps run one time. The filters compute only the region around the ROI.
- **The log** of each filter step records the requested and the effective parameters and boundary. For the Gabor filter, report the `response` (default `"modulus"`). See [Image Filtering](image_filtering.md).

## 6. Many Discretisations

**The task**: the features of six discretisations of one image, with the same preprocessing. The pipeline computes the features that do not read the discretisation one time, and copies them.

```python
from pictologics import RadiomicsPipeline

PREPROCESS = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},
    {"step": "filter_outliers", "params": {"sigma": 3.0}},
    {"step": "round_intensities", "params": {}},
    {"step": "keep_largest_component", "params": {}},
]
EXTRACT = {"step": "extract_features", "params": {"families": ["intensity", "morphology", "texture", "histogram", "ivh"]}}

pipeline = RadiomicsPipeline(load_standard=False)  # shared work is on by default
for n_bins in (8, 16, 32):
    pipeline.add_config(f"fbn_{n_bins}", PREPROCESS + [{"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}}, EXTRACT])
for bin_width in (8, 16, 32):
    pipeline.add_config(f"fbs_{bin_width}", PREPROCESS + [{"step": "discretise", "params": {"method": "FBS", "bin_width": bin_width}}, EXTRACT])

results = pipeline.run("ct.nii.gz", "lesion.nii.gz", config_names=pipeline.list_configs())
print(pipeline.deduplication_stats)
# {'reused_families': 10, 'computed_families': 20, 'cache_hit_rate': 0.333...}
```

- **The counts**: the morphology and the intensity families are computed one time and copied to the other five configurations (10 copies). The texture, histogram and IVH families read the discretisation, so they are computed for each configuration (18), with the morphology and intensity families of the first (2): 20 in all.
- **The results**: each configuration has all its features. The copied values are the same as computed values.
- **The preprocessing**: the five steps before the `discretise` step run one time.
- **A compact table**: to have each intensity and morphology feature one time in the table, put them in their own configuration, as in recipe 1.
- **Settings**: `RadiomicsPipeline(deduplicate=False)` computes each family in each configuration, for example to measure the time of each. `RadiomicsPipeline(deduplication_rules="1.1.0")` pins the rules. `save_configs` writes these settings, and `load_configs` reads them.

## 7. A Data Dictionary

**The task**: a table of all features of the configurations, before the run, for a study protocol.

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline()
catalog = pipeline.describe_features()
catalog.to_csv("feature_catalog.csv", index=False)

print(catalog.groupby("config").size())  # the number of features of each configuration
texture_fbn = catalog[(catalog["family_group"] == "Texture") & (catalog["discretisation_method"] == "FBN")]
print(texture_fbn[["config", "feature_name", "ibsi_code"]].head())
print(catalog[["family", "requires_discretisation"]].drop_duplicates().sort_values("family"))
```

See [The Feature Catalog](results.md#the-feature-catalog-describe_features) for all columns.
