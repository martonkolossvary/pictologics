# Data Loading

This page loads images and masks into Pictologics: NIfTI, NRRD, MetaImage and DICOM images, multi-phase series, PET in SUV, and masks from NIfTI, DICOM SEG, RTSTRUCT and 3D Slicer files.

## Which Function for Which File

| Input | Function |
|:--|:--|
| A NIfTI file (`.nii`, `.nii.gz`) | `load_image(path)` |
| A NRRD file (`.nrrd`, `.nhdr`) or a 3D Slicer segmentation (`.seg.nrrd`) | `load_image(path)` |
| A MetaImage file (`.mha`, `.mhd`) | `load_image(path)` |
| A DICOM folder of one series | `load_image(folder)` (with `series_uid` when the folder holds more than one series) |
| The DICOM files of one series, or one phase of `get_dicom_phases()` | `load_image([files])`, `load_image(phase)` |
| One DICOM file (also enhanced multiframe) | `load_image(path)` |
| A DICOM PET series in SUV | `load_image(folder, suv="bw")` |
| A DICOM SEG | `load_image(path, reference_image=image)`, or `load_seg()` for each segment |
| A DICOM RTSTRUCT | `load_image(path, reference_image=image)`, or `load_rtstruct()` for each ROI |
| Many mask files | `load_and_merge_images()` |
| The segments or ROIs of a file | `get_segment_info(path)` |
| A NIfTI copy of an image or a mask | `save_image(image, path)` |

The functions are in `pictologics` (`load_image`, `load_seg`, `load_rtstruct`, `load_and_merge_images`, `save_image`, `create_full_mask`), `pictologics.loaders` (`get_segment_info`) and `pictologics.utilities` (`get_dicom_phases`).

## The Image Class

Every loader gives an `Image`:

```python
from pictologics import Image

# Image fields:
# - array: numpy.ndarray, 3D, in (X, Y, Z) order
# - spacing: (x, y, z) voxel size in mm
# - origin: (x, y, z) world position of the first voxel, in mm
# - direction: 3 x 3 matrix of direction cosines (the columns are the axes), or None
# - modality: for example "CT", "MR", "PT", "Nifti", "Nrrd", "MetaImage", "SEG", "RTSTRUCT", "MergedImage"
# - source_mask: None, or a bool array of the voxels with image data
# - frame_of_reference_uid: the DICOM FrameOfReferenceUID, or None
```

- **Axis order**: (X, Y, Z), as ITK and SimpleITK use. A DICOM slice is (rows, columns) = (Y, X); the loaders turn it.
- **World frame**: `origin` and `direction` are in the LPS+ frame (left, posterior, superior) for every format, as in DICOM and ITK. A NIfTI affine is in RAS+, so the loader changes the sign of its X and Y rows. Give an `Image` that you make yourself its geometry in LPS+ too.
- **Values**: DICOM, NIfTI, NRRD and MetaImage images load as float64 (DICOM with `apply_rescale=True`, the default).
- **Dimensions**: the NIfTI and DICOM loaders give 3D arrays also for a 2D image (one slice). 2D NRRD and MetaImage files raise an error.
- `image.with_source_mask(valid)` gives a copy with a source mask (see [Sentinel Values](#sentinel-values)). With `copy=False`, the new image shares the voxel array and does not copy it.

## Images

### NIfTI

```python
from pictologics import load_image

image = load_image("scan.nii.gz")
print(image.array.shape, image.spacing, image.origin)
```

### NRRD and MetaImage

`load_image()` reads NRRD files (`.nrrd`, and `.nhdr` headers with a detached data file) and MetaImage files (`.mha`, and `.mhd` headers with a `.raw` or `.zraw` data file), as 3D Slicer and ITK write them. Pictologics reads them itself: you need no other package.

```python
image = load_image("scan.nrrd")
```

- The loader converts the RAS and LAS spaces of NRRD to LPS+.
- A file with one more axis (a 4D image, or the layers of a `.seg.nrrd` file) gives the volume of `dataset_index`.
- NRRD data: raw, gzip, bzip2 or text. MetaImage data: raw, zlib or text.
- An axis of channels (for example colours) raises an error, in both formats: the readers take one value per voxel. Data in more than one file is not supported.

### DICOM Folders

```python
image = load_image("dicom_folder/")
```

- The loader reads the files at the top of the folder. `recursive=True` takes the subfolder with the most DICOM files.
- A folder with more than one image series (for example two reconstructions) raises an error that lists the series; `series_uid` chooses one.
- Files without image pixels (RTSTRUCT, RTPLAN, SR) and SEG or RT dose objects are skipped. Images of another orientation or size than most images (for example a scout) are left out, with a warning.
- The slices are sorted by their position along the slice normal. The slice spacing comes from the positions: when the spacing tag differs from them by more than 1 %, a warning tells it.
- Uneven slice positions (a missing slice) and positions that move sideways (a gantry tilt) give a warning.
- A multi-phase series (for example a cardiac CT) gives its first phase; see [Multi-Volume Data](#multi-volume-data).

### Single and Multiframe DICOM Files

```python
image = load_image("image.dcm")
```

An enhanced multiframe file gives its geometry from its functional groups. When its frames hold more than one volume (repeated positions), `dataset_index` chooses the volume.

### Compressed DICOM

Compressed pixel data loads as uncompressed data does: RLE, JPEG Lossless, JPEG-LS (lossless and near-lossless), JPEG 2000 (lossless and lossy) and baseline JPEG. pydicom decodes it with python-gdcm and Pillow, which install with Pictologics.

- 12-bit lossy JPEG (JPEG Extended) needs `pylibjpeg` and `pylibjpeg-libjpeg` (GPL-3.0); install them yourself to read these files.
- An error while decoding names the file. Colour (RGB) DICOM raises an error: radiomics needs one value per voxel.

### Intensity Rescaling

`load_image()` applies the `RescaleSlope` and `RescaleIntercept` of each DICOM slice, so the values are real units (for example HU). The image is float64, also without rescale tags.

```python
ct = load_image("ct_scan/")                         # HU, float64
raw = load_image("ct_scan/", apply_rescale=False)    # the stored values, in their stored type
```

| Format | Rescaling |
|:--|:--|
| NIfTI | Always the `scl_slope` and `scl_inter` of the header |
| DICOM | The slope and intercept of each slice, when `apply_rescale=True` (default) |

## Multi-Volume Data

`dataset_index` chooses one volume of data with more than one: a 4D NIfTI, NRRD or MetaImage file, the layers of a `.seg.nrrd` file, the phases of a DICOM series, or the volumes of a multiframe DICOM file.

```python
volume = load_image("fmri.nii.gz", dataset_index=4)  # the 5th volume
```

### Phases of a DICOM Series

```python
from pictologics.utilities import get_dicom_phases

phases = get_dicom_phases("cardiac_ct/")  # also recursive= and series_uid=
for phase in phases:
    print(phase.index, phase.label, phase.num_slices, phase.split_tag)

image = load_image("cardiac_ct/", dataset_index=4)  # the 5th phase
image = load_image(phases[4])                        # the same, from the phase's files
```

A `DicomPhaseInfo` holds `index`, `label`, `num_slices`, `file_paths`, `split_tag` and `split_value`. The label names the tag and its value: for example `"Phase 10%"`, `"Temporal 2"`, `"Trigger 100ms"`, `"Acquisition 2"`, `"Echo 2"`, `"Volume 2"` (split by repeated positions) or `"Dataset 0"` (one phase).

The phases come from the first of these tags that changes:

1. `NominalPercentageOfCardiacPhase` (cardiac phases)
2. `TemporalPositionIdentifier`
3. `TriggerTime`
4. `AcquisitionNumber`
5. `EchoNumbers` (multi-echo MR)

A tag splits a series only where slice positions repeat, and each phase holds each position once. So a scanner that writes a new `AcquisitionNumber` every few slices gives one volume. Without such a tag, phase k gets the k-th file at each position (by `InstanceNumber`).

In a pipeline, give the phase as `image_options={"dataset_index": 4}` (see the [Cardiac CT Phases](../tutorials/cardiac_phases.md) tutorial).

## PET in SUV

`suv` converts a DICOM PET image (Modality PT) from activity concentration (Bq/ml) to its standardized uptake value:

```python
pet = load_image("pet_series/", suv="bw")
```

| `suv` | Normalised by | Unit |
|:--|:--|:--|
| `"bw"` | The body weight (PatientWeight) | g/ml |
| `"lbm"` | The lean body mass by the Janmahasatian formula, from the weight, the height (PatientSize) and the sex (PatientSex) | g/ml |
| `"lbm_james"` | The lean body mass by the James formula (PERCIST 1.0, and many older programs) | g/ml |
| `"bsa"` | The body surface area by the Du Bois formula, from the weight and the height | cm²/ml |

The factor follows the [QIBA vendor-neutral pseudo-code](https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV)), and a missing, empty or zero attribute raises an error that names it: the loader never guesses a value. On the [QIBA FDG-PET/CT digital reference object](https://depts.washington.edu/petctdro/DROsuv_main.html), `"bw"` gives its SUV values to within 6.4e-5. See the [PET in SUV](../tutorials/pet_suv.md) tutorial for the rules, the lean body mass formulas and a pipeline.

## Masks

### Masks on the Image Grid

Load a mask with `reference_image`: the loader places it on the grid of the image.

```python
ct = load_image("ct_folder/")
mask = load_image("segmentation.nii.gz", reference_image=ct)
# mask.array has the shape and the voxel order of ct.array
```

- **Voxel order**: a mask can hold the grid of the image in another voxel order (axes flipped or swapped, as some converters write NIfTI files). The loader turns it to the voxel order of the image. So a DICOM image and a NIfTI mask on the same grid load together.
- **Cropped masks**: a mask of a part of the grid (for example the box of the ROI) goes to its place in the image, by its origin. A mask of the full shape with another origin goes to its place in the same way.
- **Other grids**: a mask of another spacing or orientation raises an error. Pictologics does not resample masks onto the image grid.
- `transpose_axes` turns the axes of a file whose geometry does not show its voxel order, for example `(1, 0, 2)` swaps X and Y.

In a pipeline, a mask path gets the image as its `reference_image` by itself: `pipeline.run("ct_folder/", "segmentation.nii.gz", ...)`.

### Sub-Voxel Alignment and Overlap

Mask origins from other programs can be a small part of a voxel off the image grid. Three settings control the placement:

| Parameter | Default | Effect |
|:--|:--|:--|
| `subvoxel_warning_threshold` | `0.01` | An offset above this part of a voxel gives a warning; the mask snaps to the nearest voxel |
| `subvoxel_tolerance` | `0.5` | An offset above this raises an error. 0.5 is the largest offset that rounding can give, so the default never raises; lower it to find imprecise files |
| `min_overlap_fraction` | `0.5` | At least this part of the mask must lie in the image, else an error (a mask of another patient); 0 turns the check off |

```python
ct = load_image("ct_folder/")
mask = load_image("mask.nii.gz", reference_image=ct, subvoxel_warning_threshold=0.1)  # warn above 10 %
strict = load_image("mask.nii.gz", reference_image=ct, subvoxel_tolerance=0.05)      # error above 5 %
```

In `run()`, the same settings are `mask_subvoxel_tolerance`, `mask_subvoxel_warning_threshold` and `mask_min_overlap_fraction`.

| Problem | Result |
|:--|:--|
| Another spacing | `ValueError` (no mask resampling) |
| Another orientation (not the image axes in another order or with other signs) | `ValueError` |
| An offset above `subvoxel_warning_threshold` | `UserWarning`, and the mask snaps to the nearest voxel |
| An offset above `subvoxel_tolerance` | `ValueError` |
| An overlap below `min_overlap_fraction` | `ValueError` |
| No overlap (with `min_overlap_fraction=0`) | `UserWarning`, and an empty mask |
| A partial overlap | The overlapping part is placed; the rest is cut off |
| A DICOM mask (SEG, RTSTRUCT) of another `FrameOfReferenceUID` | `UserWarning`: the mask can belong to another scan |

### DICOM SEG

`load_image()` finds a DICOM SEG by itself. `load_seg()` gives more control:

```python
from pictologics import load_image, load_seg
from pictologics.loaders import get_segment_info

ct = load_image("ct_folder/")
for segment in get_segment_info("segmentation.dcm"):
    print(segment["segment_number"], segment["segment_label"])

labels = load_seg("segmentation.dcm", reference_image=ct)                          # one label image
two = load_seg("segmentation.dcm", reference_image=ct, segment_numbers=[1, 2])     # some segments
masks = load_seg("segmentation.dcm", reference_image=ct, combine_segments=False)   # {number: mask}
```

- Give `reference_image`: a SEG often holds frames only on the slices of its ROI, so without it the mask does not have the grid of the image.
- In one label image, the label of a segment is its segment number (uint8, or uint16 above 255 segments). Where segments overlap, one of them keeps the voxel: load overlapping segments with `combine_segments=False`.
- A FRACTIONAL SEG gives the voxels at or above `fractional_threshold` (default 0.5). A Label Map SEG loads as the others do; its segment 0 (the background) is left out.
- `load_seg()` takes the alignment settings of `load_image()`: `transpose_axes`, `subvoxel_tolerance`, `subvoxel_warning_threshold`, `min_overlap_fraction`.

For each segment of a label image, use `run_rois()` (see [Many ROIs and Batch Studies](../tutorials/batch.md)). For overlapping segments, run each mask:

```python
for number, mask in masks.items():
    results = pipeline.run(ct, mask, config_names=["standard_fbn_32"])
```

### DICOM RTSTRUCT

An RT Structure Set holds the contours of each ROI as polygons, not voxels. Pictologics fills them onto the grid of the image of the contours, so it needs the image:

```python
from pictologics import RadiomicsPipeline, load_image, load_rtstruct
from pictologics.loaders import get_segment_info

ct = load_image("ct_folder/")
for roi in get_segment_info("rtstruct.dcm"):
    print(roi["segment_number"], roi["segment_label"], roi["contour_count"])

labels = load_image("rtstruct.dcm", reference_image=ct)  # one label image: the label is the ROI Number
masks = load_rtstruct("rtstruct.dcm", ct, roi_names=["GTV", "PTV"], combine_rois=False)  # by name
results = RadiomicsPipeline().run(ct, masks["GTV"], config_names=["standard_fbn_32"])
```

- A voxel is in an ROI when its center lies inside an odd number of the contours of the ROI on its slice (the even-odd rule): a contour inside another contour cuts a hole.
- A center on an edge is inside on one side of the edge only, so two ROIs that share an edge share no voxel.
- The plane of each contour must be a slice plane of the image. A contour up to `subvoxel_tolerance` voxels (default 0.5) from a slice is filled on the nearest slice, with a warning above `subvoxel_warning_threshold`.
- In one label image, a later ROI wins where ROIs overlap. Use `combine_rois=False` for overlapping ROIs (such as a GTV in a PTV).
- `run(image, "rtstruct.dcm")` and `run_rois(image, "rtstruct.dcm")` fill the RTSTRUCT onto the grid of the image.

### 3D Slicer Segmentations (.seg.nrrd)

A `.seg.nrrd` file holds each segment as a label value in a layer; overlapping segments need more than one layer:

```python
from pictologics.loaders import get_segment_info

for segment in get_segment_info("segmentation.seg.nrrd"):
    print(segment["segment_label"], segment["label_value"], segment["layer"])

layer = load_image("segmentation.seg.nrrd", reference_image=ct, dataset_index=1)  # layer 1
```

### Merging Masks

`load_and_merge_images()` combines many mask files into one image:

```python
from pictologics import load_and_merge_images

paths = ["liver.nii.gz", "kidney.nii.gz", "spleen.nii.gz"]
ct = load_image("ct_folder/")
merged = load_and_merge_images(paths, reference_image=ct, reposition_to_reference=True, relabel_masks=True)
# liver = 1, kidney = 2, spleen = 3 (by the order of the paths)
```

| Option | Effect |
|:--|:--|
| `relabel_masks=True` | Each file gets its own label: 1, 2, 3, ... in the order of the paths |
| `binarize=True` | Every voxel that is not 0 becomes 1 before the merge |
| `conflict_resolution` | Where files overlap: `"max"` (default), `"min"`, `"first"` or `"last"` |
| `reposition_to_reference=True` | Place each file on the grid of `reference_image`, with the alignment settings above. Without it, every file must have the exact grid of the first, and nothing is turned or placed |

- The result has the common type of the loaded arrays: float64 for NIfTI, NRRD and MetaImage files and rescaled DICOM, the stored type for SEG files and DICOM with `apply_rescale=False`. `binarize` gives uint8, and `relabel_masks` gives the smallest unsigned type that holds the labels. The modality is `"MergedImage"`.
- The files load with `load_image()` without a reference image, so RTSTRUCT paths do not work here (use `load_rtstruct()`), and `series_uid` and `suv` do not apply.

## Sentinel Values

Some images hold a **sentinel value** where they have no data, for example -2048 HU outside the field of view, or 0 around a cropped image. In a pipeline, the `source_mode` of a configuration keeps these voxels out of resampling, filtering and the ROI: see [Source Modes and Sentinel Values](pipeline.md#source-modes-and-sentinel-values).

To find and mark them yourself:

```python
from pictologics.preprocessing import create_source_mask_from_sentinel, detect_sentinel_value

image = load_image("lesion_export.nii.gz")
sentinel = detect_sentinel_value(image)  # for example -2048.0, or None
if sentinel is not None:
    valid = create_source_mask_from_sentinel(image, sentinel)  # True: a voxel with data
    image = image.with_source_mask(valid.array)
```

## Saving Images

`save_image()` writes an image, a mask or a response map as a NIfTI file (`.nii` or `.nii.gz`). The geometry goes back to the RAS+ affine of NIfTI, so `load_image()`, 3D Slicer and ITK read the same grid:

```python
from pictologics import load_image, load_rtstruct, save_image

ct = load_image("ct_folder/")
masks = load_rtstruct("rtstruct.dcm", ct, roi_names=["GTV"], combine_rois=False)
save_image(masks["GTV"], "gtv.nii.gz")  # an RTSTRUCT ROI as a NIfTI mask
```

- A bool mask is saved as uint8, and a 64-bit integer array as int32 when its values fit. Other arrays keep their type.
- NIfTI keeps the geometry in float32 (about 1e-5 mm).
- A NIfTI file has no `FrameOfReferenceUID` and no DICOM modality: a saved mask no longer gets the frame-of-reference check.

## Creating a Full Mask

To analyse the whole image, make a mask of ones on its grid:

```python
from pictologics import create_full_mask

full_mask = create_full_mask(image)
```

`RadiomicsPipeline.run()` makes it by itself when `mask` is `None`.

## Next Steps

- [The Pipeline](pipeline.md): run the radiomics pipeline.
- [Masks from Other Tools](../tutorials/masks.md): label maps, probability maps, SEG, RTSTRUCT and 3D Slicer masks in a pipeline.
- [Tutorials](../tutorials/batch.md): many ROIs and cases, cardiac phases, PET and MR.
