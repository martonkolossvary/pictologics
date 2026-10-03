# Image Filtering

A filter makes a **response map** from an image: it brings out edges, blobs, textures, directions or frequencies. Radiomic features of the response map then describe these patterns. Pictologics has the filters of IBSI 2, and it checks them against the IBSI 2 reference maps.

## Overview

A filtered configuration usually has these steps:

1. **Resample** the image to cubic voxels, for example 1 × 1 × 1 mm. IBSI filters a resampled image.
2. **Resegment** the ROI by intensity (for example a HU range), when you need it. Put this step before the filter: after a filter, a `resegment` step reads the response values, not the HU values.
3. **Filter** the image.
4. **Discretise** the response map, for texture, histogram and IVH features.
5. **Extract** the features of the response map.

Every filter returns a float32 response map, half the memory of float64. For a float64 image, the filter passes run in float64, and the last pass writes float32.

## Filter Reference

| Pipeline `type` | Function | IBSI code | Required parameters | Boundary default (pipeline / direct call) | With a source mask | IBSI 2 Phase 1 |
|:--|:--|:--|:--|:--|:--|:--|
| [`"mean"`](#mean-filter) | `mean_filter` | S60F | none (`support` 15) | mirror / zero | normalized convolution | 1.a.1 to 1.a.4 pass |
| [`"gaussian"`](#gaussian-filter) | `gaussian_filter` | 8BC3 | `sigma_mm` | mirror / zero | normalized convolution | no test |
| [`"log"`](#laplacian-of-gaussian-log) | `laplacian_of_gaussian` | L6PA | `sigma_mm` | mirror / zero | normalized convolution | 2.a, 2.b pass |
| [`"laws"`](#laws-texture-energy) | `laws_filter` | JTXT | none (`kernel` `"L5E5E5"`) | mirror / zero | normalized convolution (zero-fill when rotation invariant) | 3.a.1 to 3.b.3 pass |
| [`"gabor"`](#gabor-filter) | `gabor_filter` | Q88H | `sigma_mm`, `lambda_mm` | mirror / zero | zero-fill | 4.a.1 to 4.b.2 pass |
| [`"wavelet"`](#separable-wavelets) | `wavelet_transform` | | none (`"db2"`, level 1, `"LHL"`) | mirror / zero | zero-fill | 5.a to 7.a pass |
| [`"simoncelli"`](#simoncelli-wavelet) | `simoncelli_wavelet` | PRT7 | none (level 1) | periodic / periodic | zero-fill | 8.a.1 to 8.a.3 pass |
| [`"riesz"`](#riesz-transform) | `riesz_transform`, `riesz_log`, `riesz_simoncelli` | AYRS | `order` (`"base"`), `sigma_mm` (`"log"`) | periodic / periodic | zero-fill | 9.a, 9.b.1, 10.b.1 pass |

- **Direct calls**: the functions are in `pictologics.filters`. A direct call takes `spacing_mm=1.0` by default, so `sigma_mm` then counts voxels. Give the spacing of your image. The pipeline gives it for Gaussian, LoG, Gabor and Riesz-LoG.
- **IBSI 2 Phase 1**: 28 of 28 compared tests pass. 5 tests of 2D filters are not run: Pictologics filters 3D images (the Gabor filter works on 2D planes, as IBSI defines it). 3 tests have no reference map: 9.b.2 and 10.b.2 need structure-tensor steering, which Pictologics does not have, and the map of 10.a is missing. See the [IBSI 2 Phase 1 report](../ibsi2_compliance.md).

## Boundaries and Padding

A filter reads voxels past the image edge. The boundary sets their values:

| `boundary` | The voxels past the edge |
|:--|:--|
| `"mirror"` | The image, mirrored at its edge (IBSI recommends it for most tests) |
| `"nearest"` | The value of the nearest edge voxel |
| `"periodic"` | The image from its other side (the FFT filters work so) |
| `"zero"` (or `"constant"`) | A constant: 0, or `padding_value` |

- **Defaults**: a pipeline filter step uses `"mirror"`, except the Simoncelli and Riesz filters, which use `"periodic"`. A direct call uses `"zero"` (spatial filters) or `"periodic"` (FFT filters). Give `boundary` in every step to make it clear.
- **Constant value padding** (IBSI 2 Z3VE): with `"constant"`, `padding_value` sets the constant, for example -1000 to pad a CT image with air:

    ```python
    {"step": "filter", "params": {
        "type": "log",
        "sigma_mm": 2.0,
        "boundary": "constant",
        "padding_value": -1000.0,
    }}
    ```

    Every filter takes `padding_value`. A value other than 0 needs the `"constant"` (or `"zero"`) boundary; with another boundary, the filter raises `ValueError`. A spatial filter gives the response of the image padded with the constant as far as the filter reads. A Laws energy step then pads the response with 0.

### FFT Filter Boundaries

The Simoncelli and Riesz filters work in the Fourier domain, which is periodic:

- **`"periodic"` (default)**: the FFT works on the image as it is. This is the fastest path, and the one that IBSI specifies for the Simoncelli tests.
- **`"zero"`, `"nearest"`, `"mirror"`**: the filter pads the image with the requested mode, filters, and cuts the result back. A composite filter (`riesz_log`, `riesz_simoncelli`) pads once around the whole chain.

!!! note "Requested and effective boundary"
    The transform stays periodic on the padded image, so a non-periodic boundary is honoured only approximately: the Riesz kernel has long tails, and no finite pad removes all of its error. The capability metadata calls this `as_specified_via_padding` (see [`FILTER_CAPABILITIES`](../api/filters.md#capability-metadata)), and the log entry of each filter step records the requested and the effective boundary. The boundary that a test specifies gives better agreement with its IBSI reference map: for IBSI 2 Phase 1 test 10.b.1, `"nearest"` makes the largest error about four times smaller.

!!! warning "Cost of a non-periodic boundary"
    The pad makes the array larger. On a 64³ image, Simoncelli at level 1 then works on 2.0 times the voxels and takes 1.5 times as long; Riesz 3.4 and 3.0 times; Riesz-Simoncelli at level 1 5.4 and 3.1 times; Riesz-LoG at σ 2 mm 5.4 and 2.2 times. The pad grows with the scale of the filter. Use a non-periodic boundary only when you need it.

## Filters in a Pipeline

- **Checks**: `add_config` checks every filter step. An unknown parameter gets a hint (`"kernels"` gives "did you mean 'kernel'?"). A missing `sigma_mm`, `lambda_mm` or `order` is an error, and so is a bad wavelet level or decomposition, a bad Riesz order, an unknown Gabor `response`, or a `padding_value` without the constant boundary.
- **FBS after a filter**: the response has other units than the image, so a filter step cancels the FBS start of an earlier `resegment` step. An FBS `discretise` step after a filter needs `min_val`, or a `resegment` step after the filter (IBSI asks for a fixed start).
- **The region around the ROI**: when no later step reads the image away from the ROI, the mean, Gaussian, LoG, Laws, wavelet and Gabor filters compute only the region around the ROI box (grown by the reach of the filter). The features are the same, bit for bit, and a small ROI in a large image costs much less time. A later filter, resample, `binarize_mask` that keeps the value 0, or whole-image `normalise` step needs the whole grid. See [Tips for Speed and Memory](../tutorials/performance.md).
- **The log**: the log entry of each filter step records `boundary_requested`, `boundary_effective`, `params_requested` and `params_effective` (with the spacing and the boundary that the pipeline gave).

## Mean Filter

The mean filter replaces each voxel with the mean of the cube of `support` voxels around it. It smooths the image and reduces noise.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `support` | `int` | `15` | The edge of the cube in voxels (odd), for example `3` for 3 × 3 × 3 |

=== "Pipeline"

    ```python
    {"step": "filter", "params": {"type": "mean", "support": 3, "boundary": "mirror"}}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import mean_filter

    response = mean_filter(image_array, support=3, boundary="mirror")
    ```

## Gaussian Filter

The Gaussian filter smooths the image with a Gaussian kernel of scale σ (in mm). It keeps the coarse structure and removes fine detail and noise.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `sigma_mm` | `float` | required | The scale of the Gaussian (mm) |
| `truncate` | `float` | `4.0` | The kernel ends at this many σ |
| `spacing_mm` | `float` or `tuple` | `1.0` | The voxel spacing (the pipeline gives it) |

=== "Pipeline"

    ```python
    {"step": "filter", "params": {"type": "gaussian", "sigma_mm": 2.0, "boundary": "mirror"}}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import gaussian_filter

    response = gaussian_filter(image_array, sigma_mm=2.0, spacing_mm=(0.8, 0.8, 2.0), boundary="mirror")
    ```

## Laplacian of Gaussian (LoG)

The LoG filter smooths the image with a Gaussian of scale σ, and then takes the Laplacian (the second derivative). It brings out edges and blobs of a size near σ: a larger σ finds larger blobs.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `sigma_mm` | `float` | required | The scale of the Gaussian (mm) |
| `truncate` | `float` | `4.0` | The kernel ends at this many σ (IBSI) |
| `spacing_mm` | `float` or `tuple` | `1.0` | The voxel spacing (the pipeline gives it) |

=== "Pipeline"

    ```python
    {"step": "filter", "params": {"type": "log", "sigma_mm": 3.0, "boundary": "mirror"}}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import laplacian_of_gaussian

    response = laplacian_of_gaussian(image_array, sigma_mm=3.0, spacing_mm=(1.0, 1.0, 1.0), boundary="mirror")
    ```

!!! tip "Many scales"
    Add one configuration for each σ (for example 1, 2, 3, 4 and 5 mm). Configurations with the same first steps share them, so the resample runs once.

## Laws Texture Energy

Laws filters combine three 1D kernels (L: level, E: edge, S: spot, W: wave, R: ripple) into one 3D kernel, for example `"L5E5S5"`. Each kernel brings out one type of texture.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `kernel` (pipeline), `kernels` (direct call) | `str` | `"L5E5E5"` in the pipeline | The 3D kernel. `pictologics.filters.LAWS_KERNELS` lists the 1D kernels |
| `rotation_invariant` | `bool` | `False` | Pool the response over the 24 right-angle rotations of the kernel |
| `pooling` | `str` | `"max"` | How to pool the rotations: `"max"`, `"min"` or `"average"` |
| `compute_energy` | `bool` | `False` | Give the texture energy: the mean of the absolute response in a cube of 2 × `energy_distance` + 1 voxels |
| `energy_distance` | `int` | `7` | The half-width of the energy cube (Chebyshev distance) |

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "laws",
        "kernel": "E5L5S5",
        "rotation_invariant": True,
        "pooling": "max",
        "compute_energy": True,
    }}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import laws_filter

    response = laws_filter(image_array, kernels="E5L5S5", rotation_invariant=True, compute_energy=True)
    ```

## Gabor Filter

A Gabor filter is a complex sine wave in a Gaussian envelope. It brings out texture of one wavelength and one direction. Pictologics applies 2D Gabor kernels in the axial plane, or in all three orthogonal planes, as IBSI defines them.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `sigma_mm` | `float` | required | The width of the Gaussian envelope (mm) |
| `lambda_mm` | `float` | required | The wavelength of the sine wave (mm) |
| `gamma` | `float` | `1.0` | The aspect ratio of the envelope. The kernel radius is ceil(6σ / min(1, γ)) |
| `theta` | `float` | `0.0` | The direction of the filter, in radians |
| `rotation_invariant` | `bool` | `False` | Pool the response over the directions 0, `delta_theta`, 2 × `delta_theta`, ... |
| `delta_theta` | `float` | required with `rotation_invariant` | The step between the directions, in radians. A step near 2π / n (within 1e-3) gives exactly n directions, so a rounded π/4 works |
| `pooling` | `str` | `"average"` | How to pool the directions: `"average"`, `"max"` or `"min"` |
| `average_over_planes` | `bool` | `False` | Average the response over the three orthogonal planes (else the axial plane only) |
| `response` | `str` | `"modulus"` | The part of the complex response (IBSI 2 5P3T): `"modulus"`, `"angle"` (radians), `"real"` or `"imaginary"`. Pooling pools this part |
| `spacing_mm` | `float` or `tuple` | `1.0` | The voxel spacing (the pipeline gives it). Each plane scales its kernel by its own in-plane spacing |

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "gabor",
        "sigma_mm": 5.0,
        "lambda_mm": 2.0,
        "rotation_invariant": True,
        "delta_theta": 0.7853981633974483,  # pi/4: 8 directions
    }}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import gabor_filter

    response = gabor_filter(
        image_array,
        sigma_mm=5.0,
        lambda_mm=2.0,
        gamma=1.0,
        spacing_mm=(1.0, 1.0, 1.0),
        rotation_invariant=True,
        delta_theta=0.7853981633974483,
    )
    ```

## Separable Wavelets

A separable wavelet transform splits each axis into a low (L) and a high (H) frequency part. A 3D image so gives 8 sub-bands (LLL, LLH, ..., HHH). Pictologics uses the undecimated (stationary) transform, so the response map has the size of the image.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `wavelet` | `str` | `"db2"` | The wavelet: `"haar"`, `"dbN"` (for example `"db3"`), `"coifN"` (for example `"coif1"`), and the other PyWavelets names |
| `level` | `int` | `1` | The decomposition level: a whole number of 1 or more |
| `decomposition` | `str` | `"LHL"` | The sub-band: three letters of L or H (lower case works) |
| `rotation_invariant` | `bool` | `False` | Pool the response over the 24 right-angle rotations of the image |
| `pooling` | `str` | `"average"` | How to pool the rotations: `"average"`, `"max"` or `"min"` |

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "wavelet",
        "wavelet": "coif1",
        "level": 1,
        "decomposition": "LLH",
        "rotation_invariant": True,
    }}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import wavelet_transform

    response = wavelet_transform(image_array, wavelet="coif1", decomposition="LLH", level=1, rotation_invariant=True)
    ```

## Simoncelli Wavelet

The Simoncelli wavelet is a non-separable, isotropic wavelet: it has no preferred direction. Use it when the direction of a texture does not matter.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `level` | `int` | `1` | The decomposition level: a whole number of 1 or more |

=== "Pipeline"

    ```python
    {"step": "filter", "params": {"type": "simoncelli", "level": 1}}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import simoncelli_wavelet

    response = simoncelli_wavelet(image_array, level=1)
    ```

!!! note "IBSI 2 Phase 3"
    The IBSI 2 Phase 3 manual names the mirror boundary for its Simoncelli filter, but the Phase 3 checks of Pictologics use the periodic default, as the Phase 1 tests do. See the [Phase 3 report](../ibsi2_phase3_compliance.md).

## Riesz Transform

The Riesz transform is a higher-order derivative in all directions at once. Its order `(l1, l2, l3)` gives the derivative order along each axis. Pictologics combines it with a LoG or a Simoncelli filter. It does not steer the filters (no structure-tensor alignment).

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `variant` | `str` | `"base"` | Pipeline only: `"base"` calls `riesz_transform`, `"log"` calls `riesz_log`, `"simoncelli"` calls `riesz_simoncelli` |
| `order` | `tuple` | required for `"base"`; `(1, 0, 0)` otherwise | One whole number of 0 or more for each axis, with a sum above 0 |
| `sigma_mm`, `spacing_mm`, `truncate` | | `sigma_mm` required for `"log"` | The LoG of the `"log"` variant (the pipeline gives the spacing) |
| `level` | `int` | `1` | The Simoncelli level of the `"simoncelli"` variant |

With the periodic boundary, the LoG stage of `riesz_log` uses the zero boundary. With another boundary, the whole chain uses the requested boundary.

=== "Pipeline"

    ```python
    {"step": "filter", "params": {"type": "riesz", "variant": "log", "order": [2, 0, 0], "sigma_mm": 3.0}}
    ```

=== "Direct call"

    ```python
    from pictologics.filters import riesz_log, riesz_simoncelli, riesz_transform

    response = riesz_transform(image_array, order=(2, 0, 0))
    response = riesz_log(image_array, sigma_mm=3.0, spacing_mm=(1.0, 1.0, 1.0), order=(1, 0, 0))
    response = riesz_simoncelli(image_array, level=1, order=(1, 0, 0))
    ```

## Source Masks for Sentinel Values

Some images hold a **sentinel value** where they have no data, for example -2048 HU outside the field of view. These values would spoil a filter response. Every filter takes a `source_mask` (True: a voxel with data):

| Filters | Method |
|:--|:--|
| Mean, Gaussian, LoG, Laws | Normalized convolution: each response voxel is the weighted mean of the valid voxels only |
| Laws (rotation invariant), Gabor, wavelets, Simoncelli, Riesz | Zero-fill: the invalid voxels become 0 before the filter |

### In a Pipeline

The pipeline makes the source mask from the `source_mode` of the configuration, and gives it to the filter. In `"auto"` mode with `sentinel_value`, the valid voxels are those without the sentinel value; in `"roi_only"` mode, the voxels of the ROI.

```python
pipeline.add_config(
    "filtered_features",
    [
        {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
        {"step": "resegment", "params": {"range_min": -1000, "range_max": 3000}},  # HU, before the filter
        {"step": "filter", "params": {"type": "log", "sigma_mm": 3.0, "boundary": "mirror"}},
        {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
        {"step": "extract_features", "params": {"families": ["intensity", "texture"]}},
    ],
    source_mode="auto",
    sentinel_value=-2048,
)
```

A resample step also removes the voxels without data from the masks. Without a resample step, a `resegment` step before the filter keeps the sentinel voxels out of the ROI.

### In a Direct Call

```python
from pictologics.filters import gabor_filter, laplacian_of_gaussian, mean_filter

valid = image_array > -2000  # True: a voxel with data

# Mean, Gaussian, LoG and Laws return (response, valid mask) with a source mask
response, response_valid = mean_filter(image_array, support=15, source_mask=valid)
response, response_valid = laplacian_of_gaussian(image_array, sigma_mm=3.0, source_mask=valid)

# The other filters return the response only
response = gabor_filter(image_array, sigma_mm=5.0, lambda_mm=2.0, source_mask=valid)
```

See also: [Source Modes and Sentinel Values](pipeline.md#source-modes-and-sentinel-values), the [filter API](../api/filters.md) and the [IBSI 2 reports](../ibsi2_compliance.md).
