# MR Radiomics with Normalisation

MR intensities have no fixed units: the same tissue can have other values on another scanner or day. A `normalise` step maps the intensities with one linear map, `(x - center) / scale`, before the features. IBSI asks you to report the method of the normalisation; it sets no rule.

## 1. Normalise in a Configuration

```python
from pictologics import RadiomicsPipeline

pipeline = RadiomicsPipeline(load_standard=False)
pipeline.add_config("mr_zscore", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "normalise", "params": {"method": "zscore", "region": "image", "range_min": 10}},
    {"step": "resegment", "params": {"range_min": -3, "range_max": 3}},     # outliers out
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 0.25}},
    {"step": "extract_features", "params": {"families": ["intensity", "texture"]}},
])
results = pipeline.run("t2.nii.gz", "tumour.nii.gz", config_names=["mr_zscore"])
```

The steps work in this order:

1. `resample` makes cubic voxels of 1 mm.
2. `normalise` takes the mean and the standard deviation of all image voxels with a value of 10 or more (so the dark background is out), and maps the image to `(x - mean) / sd`.
3. `resegment` removes the ROI voxels outside -3 to 3 standard deviations. It also gives the FBS bins their start: -3.
4. `discretise` makes bins of 0.25 standard deviations.

## 2. Choose the Method and the Region

| Parameter | Values | Meaning |
|:--|:--|:--|
| `method` | `"zscore"` | Center = the mean, scale = the standard deviation. The region gets mean 0 and standard deviation 1 |
| | `"percentile"` | Center = the lower percentile, scale = the distance to the upper percentile. That range becomes 0 to 1 |
| `region` | `"image"` | All voxels with image data (see `source_mode`) |
| | `"roi"` | The voxels of the intensity mask |
| `percentiles` | `[1, 99]` (default) | The two percentiles of `"percentile"`; `[0, 100]` takes the minimum and the maximum |
| `range_min`, `range_max` | numbers | Leave the voxels outside this range out of the statistics |

!!! warning "The ROI as its own reference"
    With `region` `"roi"`, the ROI gets mean 0 and standard deviation 1. Its first-order features then hold no information about its intensity. A whole-image region (without the background) or a reference tissue keeps this information.

## 3. A Reference Tissue

To normalise by another mask (for example healthy muscle), call the function before the pipeline:

```python
from pictologics import load_image
from pictologics.preprocessing import normalise_image

image = load_image("t2.nii.gz")
muscle = load_image("muscle.nii.gz", reference_image=image)
normalised = normalise_image(image, "zscore", mask=muscle)
results = pipeline.run(normalised, "tumour.nii.gz", config_names=["your_config"])
```

The configuration then needs no `normalise` step.

## Notes

- The map changes every voxel of the image, and it keeps the geometry.
- The step must come before `discretise`.
- The step cancels the FBS start of an earlier `resegment` step, as a filter does, because the units change. A later `resegment` step or `min_val` gives the FBS bins their start again.
- Region `"image"` needs the whole grid, so the steps before it do not cut the image to the ROI region. It is slower than region `"roi"`.
- The log entry of the step records the map: `center_effective` and `scale_effective`.

See also: [`normalise` in the step reference](../user_guide/pipeline_steps.md#normalise).
