# Pipeline Steps

A configuration is a list of steps. Each step is a dictionary with the step name and its parameters:

```python
{"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}}
```

The steps run in the order of the list. See [Step Order](pipeline.md#step-order) for the order rules, and [Two Masks](pipeline.md#two-masks) for `apply_to`.

| Step | Does |
|:--|:--|
| [`resample`](#resample) | Resamples the image and the masks to a new spacing |
| [`resegment`](#resegment) | Keeps the ROI voxels in an intensity range |
| [`filter_outliers`](#filter_outliers) | Removes the ROI voxels far from the ROI mean |
| [`keep_largest_component`](#keep_largest_component) | Keeps the largest connected part of the ROI |
| [`grow_mask`](#grow_mask) | Grows or shrinks the ROI by a distance in mm, or keeps a ring |
| [`round_intensities`](#round_intensities) | Rounds the intensities to whole numbers |
| [`binarize_mask`](#binarize_mask) | Selects labels of a label mask, or thresholds a probability mask |
| [`normalise`](#normalise) | Normalises the intensities, for example of an MR image |
| [`discretise`](#discretise) | Puts the intensities into bins |
| [`filter`](#filter) | Applies an IBSI 2 filter |
| [`extract_features`](#extract_features) | Computes the features |

## `resample`

Resamples the image and the masks to a new voxel spacing, with the IBSI "align grid centers" method.

| Parameter | Default | Description |
|:--|:--|:--|
| `new_spacing` | *(required)* | The new spacing (x, y, z) in mm: three positive numbers |
| `interpolation` | `"linear"` | The image interpolation: `"linear"`, `"cubic"` or `"nearest"` |
| `mask_interpolation` | `"nearest"` | The mask interpolation: `"nearest"`, `"linear"` or `"cubic"` |
| `mask_threshold` | `0.5` | With a mask interpolation other than `"nearest"`: a voxel with an interpolated value of this or more is in the ROI |
| `round_intensities` | `False` | Round the intensities to whole numbers after the resampling (IBSI does this for CT) |

- **Masks**: a mask interpolation other than `"nearest"` gives a mask of 0 and 1, so a label map loses its labels. Select the label with a `binarize_mask` step before the `resample` step.
- **Memory**: before it resamples, the pipeline checks that the new grid fits in the memory of the computer: at least 24 bytes for each voxel. When the grid does not fit, the configuration stops with a `MemoryError` in its log entry, and its features are `NaN`.
- **Speed**: when only steps near the ROI follow, the pipeline resamples only the region around the ROI, and the memory check uses that region. A later filter or resample, a `binarize_mask` step that keeps the value 0, or a whole-image `normalise` step needs the whole grid.

## `resegment`

Keeps the ROI voxels with an intensity in a range (IBSI re-segmentation). The voxels outside the range leave the ROI.

| Parameter | Default | Description |
|:--|:--|:--|
| `range_min` | `None` | The lowest intensity to keep |
| `range_max` | `None` | The highest intensity to keep |
| `apply_to` | `"both"` | `"both"`, `"morph"` or `"intensity"` |

- With `"both"`, the shape features describe the voxels in the range, for example the calcium of a plaque. With `"intensity"`, the shape features keep the whole ROI.
- The step reads the image as it is at its position: before a filter, the HU values; after a filter, the response values.
- The `range_min` of a `resegment` step that changes the intensity mask is the default start of the FBS bins (see [`discretise`](#discretise)).

## `filter_outliers`

Removes the ROI voxels with an intensity more than `sigma` standard deviations from the ROI mean (the intensity outlier filtering of IBSI re-segmentation).

| Parameter | Default | Description |
|:--|:--|:--|
| `sigma` | `3.0` | The number of standard deviations |
| `apply_to` | `"both"` | `"both"`, `"morph"` or `"intensity"` |

## `keep_largest_component`

Keeps the largest connected part of the ROI. Voxels that touch at a face, an edge or a corner are connected (26-connectivity, as in IBSI).

| Parameter | Default | Description |
|:--|:--|:--|
| `apply_to` | `"both"` | `"both"`, `"morph"` or `"intensity"` |

## `grow_mask`

Grows or shrinks the ROI by a distance in mm, or keeps a ring at its edge: for example the fat around a vessel, or the tissue around a lesion. The distances use the spacing of each axis, so the mask changes by the same amount in every direction, straight out from its surface. Put the step after `resample` and before `resegment`: a `resegment` step then keeps one tissue of the ring, for example fat from -190 to -30 HU.

| Parameter | Default | Description |
|:--|:--|:--|
| `to_mm` | *(required)* | Grow the mask by this distance (mm). A negative distance shrinks it |
| `from_mm` | `None` | Leave out the mask grown (or shrunk, when negative) by this distance, which leaves a ring. Must be below `to_mm` |
| `nearest_roi` | `False` | In `run_rois`: give each added voxel to its nearest ROI of the label map |
| `apply_to` | `"both"` | `"both"`, `"morph"` or `"intensity"` |

```python
fat_ring = [
    {"step": "resample", "params": {"new_spacing": (0.5, 0.5, 0.5)}},
    {"step": "grow_mask", "params": {"from_mm": 0, "to_mm": 3}},  # the 3 mm around the mask
    {"step": "resegment", "params": {"range_min": -190, "range_max": -30}},  # fat
    {"step": "extract_features", "params": {"families": ["intensity"]}},
]
```

| `to_mm` | `from_mm` | Result |
|:--|:--|:--|
| `3` | | The mask grown by 3 mm |
| `-1` | | The mask without its outer 1 mm |
| `3` | `0` | The ring of 3 mm around the mask |
| `5` | `2` | The ring from 2 to 5 mm around the mask |
| `0` | `-1` | The outer 1 mm of the mask |

- **Distance**: outside the mask, the distance of a voxel is the distance from its center to the nearest center of a mask voxel. Inside the mask, it is the distance to the nearest center of a voxel outside the mask. The voxels past the image edge count as outside, so a shrink also removes the mask voxels at the image edge.
- **Whole voxels**: the mask changes by whole voxels. A grow by 1 mm on 0.4 mm voxels adds 2 voxels along an axis.
- **Image data**: a grown voxel without image data (see [Source Modes](pipeline.md#source-modes-and-sentinel-values)) leaves the mask.
- **Touching ROIs**: with `nearest_roi`, `run_rois` gives each added voxel to the ROI of the label map with the nearest voxel. The rings of touching ROIs (for example the AHA segments of the myocardium, or the segments of a vessel) then share the space between them, and no ring covers another ROI. Where two vessel segments meet, the border between their rings is square to the vessel. At the free ends of the vessel, the ring goes around the end. `run()` has one ROI, so `nearest_roi` changes nothing there.
- **Function**: `pictologics.preprocessing.grow_mask` makes the same change outside the pipeline. See the [Rings Around an ROI](../tutorials/rings.md) tutorial.

## `round_intensities`

Rounds the intensities to the nearest whole number. It has no parameters.

## `binarize_mask`

Makes a mask of 0 and 1 from a label mask or a probability mask. Without this step, each voxel that is not 0 is in the ROI.

| Parameter | Default | Description |
|:--|:--|:--|
| `mask_values` | `None` | The labels to keep: a number, a list of labels, or a tuple `(min, max)` for a range of labels (both ends in the range). In a YAML or JSON file, write a range as `{range: [min, max]}`, because a list keeps only the listed labels |
| `threshold` | `0.5` | Without `mask_values`: a voxel of this value or more is in the ROI |
| `apply_to` | `"both"` | `"both"`, `"morph"` or `"intensity"` |

```python
{"step": "binarize_mask", "params": {"mask_values": 5}}        # label 5
{"step": "binarize_mask", "params": {"mask_values": [1, 3]}}   # labels 1 and 3
{"step": "binarize_mask", "params": {"mask_values": (2, 4)}}   # labels 2, 3 and 4
{"step": "binarize_mask", "params": {"threshold": 0.5}}        # a probability of 0.5 or more
```

For many labels of one label map, [`run_rois`](pipeline.md#many-rois-run_rois) is faster than one configuration for each label.

## `normalise`

Normalises the intensities with one linear map, `(x - center) / scale`: for MR images and other images without fixed units. The center and the scale come from the statistics of a region. IBSI asks you to report the method; it sets no rule.

| Parameter | Default | Description |
|:--|:--|:--|
| `method` | *(required)* | `"zscore"`: the mean and the standard deviation (the region gets mean 0 and standard deviation 1). `"percentile"`: the lower percentile and the distance to the upper one (that range becomes 0 to 1) |
| `region` | *(required)* | `"roi"`: the intensity mask. `"image"`: every voxel with image data (see [Source Modes](pipeline.md#source-modes-and-sentinel-values)) |
| `percentiles` | `[1, 99]` | The two percentiles of the `"percentile"` method. `[0, 100]` takes the minimum and the maximum |
| `range_min` | `None` | Leave voxels below this value out of the statistics, for example the MR background |
| `range_max` | `None` | Leave voxels above this value out of the statistics |

```python
mr_config = [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "normalise", "params": {"method": "zscore", "region": "image", "range_min": 10}},
    {"step": "resegment", "params": {"range_min": -3, "range_max": 3}},  # outliers out
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 0.25}},
    {"step": "extract_features", "params": {"families": ["intensity", "texture"]}},
]
```

- **The map changes every voxel** of the image. With region `"roi"`, the ROI gets mean 0 and standard deviation 1, so its first-order features lose their information. A whole-image or reference region keeps it.
- **The log**: the log entry of the step records the map (`center_effective`, `scale_effective`).
- **Order**: the step must come before `discretise`. It cancels the FBS start of earlier `resegment` steps, as a filter does, because the units change. A later `resegment` step or `min_val` sets the start again. After a normalisation, FBS suits MR too, because the bins then have the same meaning in every image.
- **Whole grid**: region `"image"` needs the whole grid, so the steps before it do not cut the image to the ROI region.
- **Other regions**: to normalise by another mask (for example a reference tissue), use the function `pictologics.preprocessing.normalise_image` before the pipeline. See the [MR Radiomics with Normalisation](../tutorials/mr_normalisation.md) tutorial.

## `discretise`

Puts the intensities of the image into bins (grey levels 1, 2, 3, ...). The texture families need it, and the histogram and IVH families read its bins.

| Parameter | Default | Description |
|:--|:--|:--|
| `method` | `"FBN"` | `"FBN"` (a fixed bin number), `"FBS"` (a fixed bin size) or `"FIXED_CUTOFFS"` |
| `n_bins` | | FBN: the number of bins (required) |
| `bin_width` | | FBS: the width of a bin (required) |
| `min_val` | | The start of the first bin. FBS: see below. FBN: the ROI minimum |
| `max_val` | | FBN: the end of the last bin. Default: the ROI maximum |
| `cutoffs` | | FIXED_CUTOFFS: the bin edges, from low to high (required). A value below the first edge gets bin 1, and a value at or above the last edge gets the last bin |

- **FBN** spreads the ROI intensities of each image over `n_bins` bins. It suits images without fixed units, such as MR and PET.
- **FBS** bins start at the same value in every image, so a grey level has the same intensity range in every image (IBSI). An FBS step without `min_val` starts at the largest `range_min` of the earlier `resegment` steps that change the intensity mask (`apply_to` `"both"` or `"intensity"`). A `filter` or `normalise` step after them cancels this start, because the values have other units then. Without `min_val` and without such a `resegment` step, `add_config` raises an error. The log entry of the step records the start that it used (`min_val_effective`).
- **FBS for CT**: give `min_val` -1000 (air), or a `resegment` step, as the standard FBS configurations do.

## `filter`

Applies an IBSI 2 filter. The image is then the response map, for all later steps and features. See [Image Filtering](image_filtering.md) for each filter.

| Parameter | Default | Description |
|:--|:--|:--|
| `type` | *(required)* | `"mean"`, `"gaussian"`, `"log"`, `"laws"`, `"gabor"`, `"wavelet"`, `"simoncelli"` or `"riesz"` |
| `boundary` | `"mirror"` (`"periodic"` for Simoncelli and Riesz) | `"mirror"`, `"nearest"`, `"periodic"` (or `"wrap"`), `"zero"` (or `"constant"`) |
| `padding_value` | `0` | The constant of the `"constant"` (or `"zero"`) boundary (IBSI 2 Z3VE). A value other than 0 needs that boundary |

The parameters of each filter type:

| `type` | Required | Other parameters (default) |
|:--|:--|:--|
| `mean` | | `support` (15) |
| `gaussian` | `sigma_mm` | `truncate` (4.0) |
| `log` | `sigma_mm` | `truncate` (4.0) |
| `laws` | | `kernel` (`"L5E5E5"`), `rotation_invariant` (False), `pooling` (`"max"`), `compute_energy` (False), `energy_distance` (7) |
| `gabor` | `sigma_mm`, `lambda_mm` | `gamma` (1.0), `theta` (0.0), `rotation_invariant` (False), `delta_theta`, `pooling` (`"average"`), `average_over_planes` (False), `response` (`"modulus"`) |
| `wavelet` | | `wavelet` (`"db2"`), `level` (1), `decomposition` (`"LHL"`), `rotation_invariant` (False), `pooling` (`"average"`) |
| `simoncelli` | | `level` (1) |
| `riesz` | `order` (variant `"base"`), `sigma_mm` (variant `"log"`) | `variant` (`"base"`, `"log"` or `"simoncelli"`), `order` ((1, 0, 0) for `"log"` and `"simoncelli"`), `truncate` (4.0, `"log"`), `level` (1, `"simoncelli"`) |

- **Spacing**: the Gaussian, LoG, Gabor and Riesz-LoG filters work in mm. The pipeline gives them the spacing of the image.
- **Checks**: `add_config` raises an error for an unknown or a missing parameter, an unknown boundary, a `padding_value` without the constant boundary, a bad wavelet level or decomposition, a bad Riesz order, and an unknown Gabor `response`.
- **The log**: the log entry of each filter step records `boundary_requested`, `boundary_effective`, `params_requested` and `params_effective`.

## `extract_features`

Computes the features of the image and the masks as they are at this step.

| Parameter | Default | Description |
|:--|:--|:--|
| `families` | `["intensity", "morphology", "texture", "histogram", "ivh"]` | The feature families (see the table below) |
| `include_spatial_intensity` | `False` | With the intensity family: also Moran's I and Geary's C |
| `include_local_intensity` | `False` | With the intensity family: also the local and global intensity peaks |
| `ivh_params` | `None` | The IVH units: `bin_width` and `min_val` (the bins, to give the IVH intensities in image units), and `target_range_min` and `target_range_max` (the intensity range of the fractions, else `min_val` and `max_val`) |
| `ivh_discretisation` | `None` | A discretisation for the IVH only, with the parameters of `discretise` (method default `"FBS"`) |
| `ivh_use_continuous` | `False` | The IVH of the image before the discretisation |
| `texture_matrix_params` | `None` | The texture options (IBSI 1): `ngldm_alpha` (the NGLDM coarseness, default 0), `glcm_distance` (the GLCM pair distance in each of the 13 directions), `ngtdm_distance` and `ngldm_distance` (the Chebyshev distance of the neighbourhood). The distances are whole numbers of 1 or more (default 1) |
| `spatial_intensity_params`, `local_intensity_params` | `None` | Options of the spatial and local intensity functions |

The feature families:

| Family | Features | Description |
|:--|:--|:--|
| `"intensity"` | 18 | The first-order statistics: mean, variance, skewness and more |
| `"spatial_intensity"` | 2 | Moran's I and Geary's C |
| `"local_intensity"` | 2 | The local and the global intensity peak |
| `"morphology"` | 27 | The shape and size features: volume, surface, sphericity and more |
| `"histogram"` | 23 | The statistics of the discretised intensities |
| `"ivh"` | 7 | The intensity-volume histogram |
| `"texture"` | 95 | All six texture families below |
| `"glcm"` | 25 | The grey level co-occurrence matrix |
| `"glrlm"` | 16 | The grey level run length matrix |
| `"glszm"` | 16 | The grey level size zone matrix |
| `"gldzm"` | 16 | The grey level distance zone matrix |
| `"ngtdm"` | 5 | The neighbourhood grey tone difference matrix |
| `"ngldm"` | 17 | The neighbouring grey level dependence matrix |

- `"texture_glcm"` and the other `"texture_..."` names are the same as `"glcm"` and the others.
- Each texture family uses the 3D merged matrix of IBSI. Ask only for the texture families that you need: the pipeline computes only their matrices.
- The spatial and local intensity features take more time on large ROIs. The spatial features use an FFT on large ROIs. When the FFT needs more than 16 GB, or more than half of the memory, a slower loop runs, with a warning.
- A configuration can have more than one `extract_features` step, for example one before and one after a filter. The feature names of the steps must differ, because a later value replaces an earlier one with the same name.
