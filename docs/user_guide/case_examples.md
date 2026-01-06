# Case examples

This section provides practical, end-to-end examples for common Pictologics workflows. The goal is to show how to combine loading, preprocessing, feature extraction, and result export patterns into scripts you can reuse. Many of these common processing techniques are in development to make the batch processing of cases even more easy. Therefore check back often to see any of these processing steps being implemented into the core package.

---

## Case 1: Batch radiomics from a folder of NIfTI files (no masks)

### Scenario

You have a folder of NIfTI volumes where each file is a separate exported segmentation-like volume. There are no mask files.

You want to:

- Process every `*.nii` / `*.nii.gz` file in a folder.
- Use the whole image as the initial ROI (because no mask is provided).
- Resample to **0.5×0.5×0.5 mm**.
- Restrict the ROI to intensities in **[-100, 3000]** (CT HU example).
- Remove disjoint parts by keeping the **largest connected component**.
- Compute **all radiomic feature families** for four discretisation settings:
  - FBN with `n_bins=8`
  - FBN with `n_bins=16`
  - FBS with `bin_width=8`
  - FBS with `bin_width=16`
- Export a **single wide CSV** where:
  - Each row is one input NIfTI file.
  - Columns include metadata (e.g., filename) + all computed features.
- Save pipeline logs to a separate folder.

### Important notes

- **Maskless pipeline runs**: `RadiomicsPipeline.run(...)` accepts `mask=None`, `mask=""`, or an omitted mask argument.
  In that case, Pictologics generates a full (all-ones) ROI mask internally (whole-image ROI).
- **Empty ROI is an error**: If your preprocessing removes all voxels (e.g., a too-tight HU range), the pipeline raises
  a clear error instead of silently returning empty outputs.
- **Morphology on whole-image ROI**: Shape features will describe the shape of the ROI mask.
  With a maskless run, that ROI starts as the full image volume, then becomes whatever remains after resegmentation
  and connected-component filtering. This is valid computationally, but make sure it matches your scientific intent.
- **Sentinel value filtering**: The `resegment` step with `range_min=-100` also filters out common DICOM sentinel
  values (e.g., -1024, -2048) that represent missing/invalid data. This is the IBSI-recommended approach for
  handling NA values in medical imaging data.

### Full example script

```python
from pathlib import Path

from pictologics import RadiomicsPipeline
from pictologics.results import format_results, save_results

def strip_nii_suffix(name):
    """Remove .nii or .nii.gz suffix from filename."""
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name

def main():
    # Configure paths
    input_dir = Path("path/to/nifti_folder")
    output_csv = Path("results.csv")
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define common preprocessing steps
    base_steps = [
        # Resample to 0.5mm isotropic. Round intensities to integers (useful for HU).
        {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5), "round_intensities": True}},
        # Exclude voxels outside standard HU range
        {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},
        # Keep only the largest connected component of the ROI mask
        {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    ]

    # Shared feature extraction configuration
    extract_all = {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": False,
            "include_local_intensity": False,
        },
    }

    # Initialize the pipeline
    pipeline = RadiomicsPipeline()

    # Add 4 configurations (2 FBN, 2 FBS)
    for n_bins in (8, 16):
        pipeline.add_config(
            f"case1_fbn_{n_bins}",
            base_steps + [
                {"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}},
                extract_all,
            ],
        )

    for bin_width in (8, 16):
        pipeline.add_config(
            f"case1_fbs_{bin_width}",
            base_steps + [
                {"step": "discretise", "params": {"method": "FBS", "bin_width": bin_width}},
                extract_all,
            ],
        )

    # Prepare for batch processing
    rows = []
    nifti_paths = sorted(input_dir.glob("*.nii*"))
    if not nifti_paths:
        raise ValueError(f"No NIfTI files found in: {input_dir}")

    # Process each NIfTI file
    for path in nifti_paths:
        subject_id = strip_nii_suffix(path.name)
        pipeline.clear_log()

        # Run pipeline (mask omitted -> whole-image ROI)
        results = pipeline.run(
            image=str(path),
            subject_id=subject_id,
            config_names=["case1_fbn_8", "case1_fbn_16", "case1_fbs_8", "case1_fbs_16"],
        )

        # Format results as flat dictionary and add metadata
        row = format_results(
            results,
            fmt="wide",
            meta={"subject_id": subject_id, "file": str(path)}
        )
        rows.append(row)

        # Save per-case logs
        pipeline.save_log(str(log_dir / f"{subject_id}.json"))

    # Consolidated export of all results
    save_results(rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    main()
```

### Output format

- One row per file.
- Columns include:
  - `subject_id`
  - `file`
  - Feature columns prefixed by configuration name (e.g., `case1_fbn_8__mean_intensity_Q4LE`).

---

## Case 2: Batch radiomics from DICOM case folders (Image + Segmentation)

### Scenario

You have a folder of *cases*. Each case is a separate folder and contains two subfolders:

- `Image/`: the DICOM image series (CT/MR/etc.), stored at an arbitrary depth.
- `Segmentation/`: the DICOM segmentation, also stored at an arbitrary depth.

You want to:

- For each case, recursively discover the DICOM series folders.
- Load the image series and segmentation series.
- Convert the segmentation to a binary ROI mask by keeping all voxels where the segmentation value is `> 0` (handled automatically during loading).
- Resample to **1×1×1 mm**.
- Do **not** apply intensity filtering/resegmentation, and do **not** keep the largest connected component.
- Compute all radiomic feature families for two discretisations:
  - FBS with `bin_width=256`
  - FBN with `n_bins=64`
- Export a **long-format CSV** (tidy data, one row per feature).
- Save pipeline logs into a separate folder (one JSON per case).
- Show a progress bar during batch processing.

### Notes

- **Progress bar dependency**: This example uses `tqdm`.
    - If you are running this outside the library repo, install it with `pip install tqdm`.
    - If you are adding it to your Poetry-managed project, use `poetry add tqdm`.
- **Segmentation DICOM at arbitrary depth**: The helper `_find_dicom_series_root(...)` looks for the subfolder with
    the most `.dcm` files and uses that as the series root.
- **Multiple masks in one SEG**: If your segmentation DICOM encodes multiple labels (e.g., values 1..N), the
    `_binarize_segmentation_mask(...)` step turns it into a single ROI by keeping all voxels where the value is `> 0`.
- **Multi-phase DICOM series**: If your image series contains multiple phases (e.g., cardiac CT with 10%, 20%... phases),
    use `get_dicom_phases()` to discover available phases and `dataset_index` to select one:
    ```python
    from pictologics.utilities import get_dicom_phases
    
    phases = get_dicom_phases(str(image_root))
    print(f"Found {len(phases)} phases")
    # Load a specific phase (default is 0)
    image = load_image(str(image_root), recursive=True, dataset_index=0)
    ```

### Full example script

```python
from pathlib import Path
import numpy as np
from pictologics import Image, RadiomicsPipeline, load_image, load_and_merge_images
from pictologics.results import format_results, save_results


def main():
    # Configure paths
    cases_dir = Path("path/to/cases_root")
    output_csv = Path("dicom_results_long.csv")
    log_dir = Path("dicom_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Find all case directories
    case_dirs = sorted([p for p in cases_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No case folders found in: {cases_dir}")

    # Shared feature extraction settings
    extract_all = {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": False,
            "include_local_intensity": False,
        },
    }

    # Initialize the pipeline
    pipeline = RadiomicsPipeline()

    # Configuration A: Fixed Bin Size
    pipeline.add_config(
        "case2_fbs_256",
        [
            {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
            {"step": "discretise", "params": {"method": "FBS", "bin_width": 256.0}},
            extract_all,
        ],
    )

    # Configuration B: Fixed Bin Number
    pipeline.add_config(
        "case2_fbn_64",
        [
            {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 64}},
            extract_all,
        ],
    )

    # Use tqdm for a progress bar
    from tqdm import tqdm

    rows = []
    for case_dir in tqdm(case_dirs, desc="Radiomics (DICOM cases)", unit="case"):
        subject_id = case_dir.name
        image_root = case_dir / "Image"
        seg_root = case_dir / "Segmentation"

        # load_image with recursive=True finds the best series folder automatically
        image = load_image(str(image_root), recursive=True)
        
        # Load the segmentation using load_and_merge_images with binarize=True
        # recursive=True ensures we find the DICOM series inside the Segmentation folder
        mask = load_and_merge_images(
            [str(seg_root)], 
            binarize=True, 
            recursive=True
        )

        pipeline.clear_log()

        # Execute extraction for both configurations
        results = pipeline.run(
            image=image,
            mask=mask,
            subject_id=subject_id,
            config_names=["case2_fbs_256", "case2_fbn_64"],
        )

        # Format results and store
        row = format_results(
            results,
            fmt="long",  # Tidy format: [subject_id, config, feature_name, value]
            meta={
                "subject_id": subject_id,
                "image_root": str(image_root),
                "seg_root": str(seg_root),
            },
        )
        rows.append(row)

        # Save per-case logs
        pipeline.save_log(str(log_dir / f"{subject_id}.json"))

    # Final export
    save_results(rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    main()
```

---

## Case 3: Batch radiomics from a flat NIfTI folder (multiple masks per image)

### Scenario

You have a single large folder containing both images and masks as NIfTI files.

- Image files end with `_IMG` (e.g., `CASE001_IMG.nii.gz`).
- Mask files end with `MASKn` where `n` is a number (e.g., `CASE001_MASK1.nii.gz`, `CASE001_MASK2.nii.gz`).
- There can be multiple segmentation masks per image.

You want to:

- For each image, automatically find all its corresponding masks.
- Merge all masks into a single ROI using `load_and_merge_images(...)`.
- Do **not** apply any preprocessing (no resampling, no thresholding, no connected components).
- Compute radiomics for the **six standard discretisations** (FBN 8/16/32 and FBS 8/16/32).
- Export a **long-format JSON** file for easy ingestion into NoSQL databases or web apps.
- Save logs for each case into a separate folder.
- Show a progress bar during batch processing.

!!! tip
    **Flexible Image Loading**
    The `load_and_merge_images` function uses `load_image` for each path in its input list. This means you can merge:
    - **Multiple NIfTI files** (as shown here).
    - **DICOM series folders**.
    - **Single DICOM slice files**.
    - Or a **mix** of these formats, provided they share the same spatial geometry.

    You can also pass **`recursive=True`** (to search subfolders) or **`dataset_index=N`** (for 4D files) to control how each mask is loaded.

### Full example script

```python
from pathlib import Path

import numpy as np

from pictologics import Image, RadiomicsPipeline, load_and_merge_images, load_image
from pictologics.results import format_results, save_results





def strip_suffixes(name):
    """Strip .nii/.nii.gz and trailing _IMG to get a stable case id."""
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]
    if name.endswith("_IMG"):
        name = name[: -len("_IMG")]
    return name


def main():
    # Configure paths
    input_dir = Path("path/to/nifti_folder")
    output_file = Path("case3_results_long.json")
    log_dir = Path("case3_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use tqdm for a progress bar
    from tqdm import tqdm

    # Identify all images (files ending in _IMG)
    image_paths = sorted(input_dir.glob("*_IMG.nii*"))
    if not image_paths:
        raise ValueError(f"No *_IMG NIfTI files found in: {input_dir}")

    # Initialize the pipeline
    pipeline = RadiomicsPipeline()

    # Shared feature extraction settings
    extract_all = {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": False,
            "include_local_intensity": False,
        },
    }

    # Add 6 target configurations (no preprocessing requested)
    for n_bins in (8, 16, 32):
        pipeline.add_config(
            f"case3_fbn_{n_bins}",
            [
                {"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}},
                extract_all,
            ],
        )

    for bin_width in (8.0, 16.0, 32.0):
        pipeline.add_config(
            f"case3_fbs_{int(bin_width)}",
            [
                {"step": "discretise", "params": {"method": "FBS", "bin_width": bin_width}},
                extract_all,
            ],
        )

    target_configs = [
        "case3_fbn_8", "case3_fbn_16", "case3_fbn_32",
        "case3_fbs_8", "case3_fbs_16", "case3_fbs_32",
    ]

    # Process each image and its associated masks
    rows = []
    for img_path in tqdm(image_paths, desc="Radiomics (NIfTI images)", unit="image"):
        subject_id = strip_suffixes(img_path.name)

        # Find all corresponding masks (e.g., CASE001_MASK1.nii.gz, etc.)
        mask_paths = sorted(input_dir.glob(f"{subject_id}_MASK*.nii*"))
        if not mask_paths:
            raise ValueError(f"No masks found for {subject_id}")

        image = load_image(str(img_path))

        # Merge multiple masks into one and ensure binary semantics
        mask = load_and_merge_images(
            [str(p) for p in mask_paths],
            reference_image=image,
            binarize=True
        )

        pipeline.clear_log()

        # Run extraction for all target configurations
        results = pipeline.run(
            image=image,
            mask=mask,
            subject_id=subject_id,
            config_names=target_configs,
        )

        # Format results and store metadata
        row = format_results(
            results,
            fmt="long",
            meta={
                "subject_id": subject_id,
                "image": str(img_path),
                "masks": ";".join(str(p) for p in mask_paths),
            },
        )
        rows.append(row)

        # Save per-case logs
        pipeline.save_log(str(log_dir / f"{subject_id}.json"))

    # Consolidated export (save_results handles list of DataFrames for long format automatically)
    save_results(rows, output_file)
    print(f"Wrote {len(rows)} cases to {output_file}")


if __name__ == "__main__":
    main()
```

---


## Case 4: Parallel batch radiomics from DICOM cases (merge multiple segmentation folders)

### Scenario

You have a folder of *cases*. Each case is a separate folder and contains:

- `Image/`: the DICOM image series (CT/MR/etc.), stored at an arbitrary depth.
- `Segmentation/`: **multiple** subfolders, each containing a segmentation series at an arbitrary depth.

You want to:

- For each case, recursively discover the DICOM image series folder.
- Discover **all** segmentation series folders under `Segmentation/` and load them.
- Convert each segmentation to a binary mask (values `> 0` become 1).
- Merge all binary masks into a single ROI mask per case.
- Apply **all preprocessing steps supported by the pipeline** (resample, resegment, outlier filtering,
  intensity rounding, and largest connected component).
- Compute radiomics for **six discretisations** (FBN 8/16/32 and FBS 8/16/32).
- Run cases **in parallel** on multiple CPU cores, with a user-controlled `n_jobs`.
- Export a **single wide JSON** file (one object per case) and save per-case logs.

### Notes

- **Progress bar dependency**: This example uses `tqdm`.
    - If you are running this outside the library repo, install it with `pip install tqdm`.
    - If you are adding it to your Poetry-managed project, use `poetry add tqdm`.
- **Multiprocessing requirement**: On Windows/macOS, keep the parallel execution inside
  `if __name__ == "__main__":` (as shown) to avoid process-spawn issues.
- **JIT warmup in parallel workers**: Pictologics performs a Numba JIT warmup at package import.
    With `ProcessPoolExecutor`, each worker is a separate Python process, so warmup happens **once per worker process**
    (on its first import of `pictologics`) and then stays warm for all cases that worker processes.
    It is **not** re-run for every case unless you explicitly call `warmup_jit()` inside your per-case function.
    You can disable auto-warmup via `PICTOLOGICS_DISABLE_WARMUP=1` if you prefer to skip the upfront cost.
- **Preprocessing parameters are dataset-dependent**: The `resegment` range here uses the CT HU example
  `[-100, 3000]`. Adjust or remove it for non-CT data.

### Full example script

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from pictologics import Image, RadiomicsPipeline, load_image, load_and_merge_images
from pictologics.results import format_results, save_results


def collect_segmentation_series_roots(seg_root):
    """Collect all segmentation series subfolders."""
    if not seg_root.exists():
        raise ValueError(f"Folder does not exist: {seg_root}")

    subdirs = sorted([p for p in seg_root.iterdir() if p.is_dir()])
    if not subdirs:
        return [seg_root]

    return subdirs

def build_case4_pipeline():
    """Define the pipeline with preprocessing and all feature families."""
    pipeline = RadiomicsPipeline()

    # Define standard CT preprocessing
    preprocess_steps = [
        {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
        {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},
        {"step": "filter_outliers", "params": {"sigma": 3.0}},
        {"step": "round_intensities"},
        {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    ]

    extract_all = {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": False,
            "include_local_intensity": False,
        },
    }

    # Add discretisation variants
    for n_bins in (8, 16, 32):
        pipeline.add_config(
            f"case4_fbn_{n_bins}",
            preprocess_steps + [{"step": "discretise", "params": {"method": "FBN", "n_bins": n_bins}}, extract_all],
        )

    for bin_width in (8.0, 16.0, 32.0):
        pipeline.add_config(
            f"case4_fbs_{int(bin_width)}",
            preprocess_steps + [{"step": "discretise", "params": {"method": "FBS", "bin_width": bin_width}}, extract_all],
        )

    config_names = [f"case4_fbn_{b}" for b in (8, 16, 32)] + [f"case4_fbs_{b}" for b in (8, 16, 32)]
    return pipeline, config_names

def process_case(case_dir, log_dir):
    """Worker function for single case processing."""
    case_path = Path(case_dir)
    subject_id = case_path.name
    image_root = case_path / "Image"
    seg_root = case_path / "Segmentation"

    # Load image recursively
    image = load_image(str(image_root), recursive=True)

    # Load and merge all found segmentations
    seg_roots = collect_segmentation_series_roots(seg_root)
    
    # load_and_merge_images handles multiple paths, geometry checking, and binarization
    try:
        mask = load_and_merge_images(
            [str(p) for p in seg_roots], 
            reference_image=image, 
            binarize=True, 
            recursive=True
        )
    except ValueError as e:
         raise ValueError(f"Failed to load/merge masks for {subject_id}: {e}")

    # Setup and run pipeline
    pipeline, config_names = build_case4_pipeline()
    pipeline.clear_log()
    results = pipeline.run(image=image, mask=mask, subject_id=subject_id, config_names=config_names)

    # Save results and log
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    pipeline.save_log(str(Path(log_dir) / f"{subject_id}.json"))

    return format_results(
        results,
        fmt="wide",
        meta={
            "subject_id": subject_id,
            "image_root": str(image_root),
            "seg_roots": ";".join(str(p) for p in seg_roots),
        },
    )

def main():
    # Configure paths
    cases_dir = Path("path/to/cases_root")
    output_file = Path("case4_parallel_results.json")
    log_dir = Path("case4_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Start parallel processing
    n_jobs = 4
    case_dirs = sorted([p for p in cases_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No case folders found in: {cases_dir}")

    from tqdm import tqdm
    rows = []
    errors = []

    # Map each case to the worker function
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(process_case, str(case_dir), str(log_dir)): case_dir
            for case_dir in case_dirs
        }

        with tqdm(total=len(futures), desc="Radiomics (parallel cases)", unit="case") as pbar:
            for fut in as_completed(futures):
                case_dir = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:
                    errors.append((str(case_dir), repr(e)))
                finally:
                    pbar.update(1)

    if errors:
        msg = "\n".join(f"- {case}: {err}" for case, err in errors)
        raise RuntimeError(f"One or more cases failed:\n{msg}")

    # Final data export
    save_results(rows, output_file)
    print(f"Wrote {len(rows)} cases to {output_file}")

if __name__ == "__main__":
    main()
```

---

## Case 5: Batch radiomics from DICOM SEG files with multiple segments

### Scenario

You have a folder of *cases*. Each case contains:

- `Image/`: the DICOM image series (CT/MR/etc.)
- `segmentation.dcm`: a DICOM SEG file with multiple labeled segments (e.g., liver, spleen, kidneys)

You want to:

- Load the image and inspect the available segments in the SEG file.
- Process **each segment separately** to get per-organ radiomics.
- Apply standard preprocessing and compute features for each segment.
- Export results with segment labels in the output.

### Notes

- **`get_segment_info()`** returns metadata about each segment (number, label, algorithm).
- **`load_seg()` with `combine_segments=False`** returns a dict mapping segment numbers to `Image` objects.
- **Alignment**: Use `reference_image` to ensure the SEG mask matches the CT geometry.

### Full example script

```python
from pathlib import Path
from pictologics import load_image, load_seg, RadiomicsPipeline
from pictologics.loaders import get_segment_info
from pictologics.results import format_results, save_results


def main():
    # Configure paths
    cases_dir = Path("path/to/cases_root")
    output_csv = Path("case5_per_segment_results.csv")
    log_dir = Path("case5_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Find all case directories
    case_dirs = sorted([p for p in cases_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise ValueError(f"No case folders found in: {cases_dir}")

    # Initialize pipeline with standard config
    pipeline = RadiomicsPipeline()
    pipeline.add_config(
        "case5_fbn_32",
        [
            {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
            {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
            {
                "step": "extract_features",
                "params": {
                    "families": ["intensity", "morphology", "texture", "histogram"],
                    "include_spatial_intensity": False,
                    "include_local_intensity": False,
                },
            },
        ],
    )

    from tqdm import tqdm
    rows = []

    for case_dir in tqdm(case_dirs, desc="Radiomics (per-segment)", unit="case"):
        subject_id = case_dir.name
        image_root = case_dir / "Image"
        seg_file = case_dir / "segmentation.dcm"

        # Load the reference image
        image = load_image(str(image_root), recursive=True)

        # Inspect available segments
        segments = get_segment_info(str(seg_file))
        print(f"\n{subject_id}: Found {len(segments)} segments")
        for seg in segments:
            print(f"  Segment {seg['segment_number']}: {seg['segment_label']}")

        # Load each segment separately, aligned to image geometry
        segment_masks = load_seg(
            str(seg_file),
            combine_segments=False,  # Returns dict {seg_num: Image}
            reference_image=image
        )

        # Process each segment
        for seg_num, mask in segment_masks.items():
            # Find segment label from metadata
            seg_info = next(s for s in segments if s["segment_number"] == seg_num)
            seg_label = seg_info["segment_label"]

            pipeline.clear_log()
            results = pipeline.run(
                image=image,
                mask=mask,
                subject_id=f"{subject_id}_{seg_label}",
                config_names=["case5_fbn_32"],
            )

            row = format_results(
                results,
                fmt="wide",
                meta={
                    "subject_id": subject_id,
                    "segment_number": seg_num,
                    "segment_label": seg_label,
                },
            )
            rows.append(row)

            # Save per-segment log
            pipeline.save_log(str(log_dir / f"{subject_id}_{seg_label}.json"))

    # Export all results
    save_results(rows, output_csv)
    print(f"\nWrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
```

### Output format

- One row per segment per case.
- Columns include:
  - `subject_id` - Case identifier
  - `segment_number` - Numeric segment ID from SEG file
  - `segment_label` - Human-readable segment name (e.g., "Liver", "Spleen")
  - Feature columns prefixed by configuration name

---

## Output Options

For details on result formatting (`wide` vs `long`, `output_type` options) and export functions, see the [Feature Calculations - Working with Results](feature_calculations.md#working-with-results).

