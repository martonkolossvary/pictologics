# Image Filtering

Pictologics provides a comprehensive suite of **IBSI 2 compliant image filters**. These filters are used to generate **response maps** from the original image, highlighting specific textures, edges, or frequencies. Radiomic features are then extracted from these response maps to quantify patterns invisible to the naked eye.

## Overview

Applying filters is a key step in advanced radiomics. The process generally involves:

1.  **Preprocessing**: The image is resampled to isotropic spacing (e.g., 1x1x1 mm) and intensities are often rounded.
2.  **Filtering**: A convolutional filter (e.g., LoG, Wavelet) is applied to the preprocessed image.
3.  **Feature Extraction**: Features (intensity, texture, etc.) are calculated from the *filtered* image.

!!! note "IBSI 2 Compliance"
    To maintain IBSI 2 compliance, filters should be applied **after** interpolation to isotropic spacing and **before** discretisation. Most filters also require specific boundary conditions (usually `mirror`).

## Available Filters

| Filter | Code (IBSI) | Description |
|:-------|:------------|:------------|
| [**Mean**](#mean-filter) | S60F | Averages intensities in a local neighborhood. |
| [**Laplacian of Gaussian**](#laplacian-of-gaussian-log) | L6PA | Detects edges and blobs at specific scales. |
| [**Laws Texture Energy**](#laws-texture-energy) | JTXT | Measures texture energy using 1D kernels. |
| [**Gabor**](#gabor-filter) | Q88H | Detects frequency content at specific orientations. |
| [**Separable Wavelet**](#separable-wavelets) | - | Decomposes image into frequency sub-bands (Haar, db, etc.). |
| [**Simoncelli Wavelet**](#simoncelli-wavelet) | PRT7 | Isotropic, non-separable wavelet transform. |
| [**Riesz Transform**](#riesz-transform) | AYRS | Steerable filter bank for texture orientation. |

## Mean Filter

The Mean filter replaces each voxel's intensity with the average intensity of its neighborhood. It smooths the image and reduces noise.

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `support` | `int` | The size of the kernel (e.g., `3` for a 3x3x3 kernel). |
| `boundary` | `str` | Boundary condition, resolved by enum name (`"mirror"`, `"nearest"`, `"periodic"`, `"zero"`). Default: `"zero"`. |

### Usage

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "mean",
        "support": 3,
        "boundary": "mirror"
    }}
    ```

=== "Direct API"

    ```python
    from pictologics.filters import mean_filter
    
    response = mean_filter(image_array, support=3, boundary="mirror")
    ```

## Laplacian of Gaussian (LoG)

The Laplacian of Gaussian (LoG) filter highlights regions of rapid intensity change (edges) and blobs. It first smooths the image with a Gaussian kernel (scale σ) and then calculates the Laplacian (second derivative).

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `sigma_mm` | `float` | Scale of the Gaussian in physical units (mm). Larger values detect larger blobs. |
| `truncate` | `float` | Cutoff for the Gaussian kernel in standard deviations. Default: `4.0` (IBSI recommended). |
| `spacing_mm` | `tuple` | Voxel spacing of the image (handled automatically by Pipeline). |

### Usage

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "log",
        "sigma_mm": 3.0,  # e.g., coarse texture
        "truncate": 4.0
    }}
    ```

=== "Direct API"

    ```python
    from pictologics.filters import laplacian_of_gaussian
    
    response = laplacian_of_gaussian(
        image_array, 
        sigma_mm=3.0, 
        spacing_mm=(1.0, 1.0, 1.0),
        truncate=4.0
    )
    ```

## Laws Texture Energy

Laws filters use a set of 1D kernels (Level, Edge, Spot, Wave, Ripple) combined to form 3D masks. These masks detect specific types of texture energy.

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `kernels` | `str` | The 3D kernel name (e.g., `"E5L5S5"`). See `pictologics.filters.laws.LAWS_KERNELS` for list. Direct-API kwarg; the Pipeline `"laws"` step uses the key `"kernel"` (singular). |
| `rotation_invariant` | `bool` | If `True`, averages response across rotationally symmetric kernels (e.g., E5L5S5, L5E5S5, L5S5E5). |
| `compute_energy` | `bool` | If `True`, computes local energy (average absolute deviation) in a window. Default: `False`. |
| `energy_distance` | `int` | Distance for the energy window (Chebyshev distance). Default: `7`. |

### Usage

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "laws",
        "kernel": "E5L5S5",
        "rotation_invariant": True,
        "compute_energy": True
    }}
    ```

=== "Direct API"

    ```python
    from pictologics.filters import laws_filter
    
    # Rotation invariant energy map
    response = laws_filter(
        image_array, 
        kernels="E5L5S5", 
        rotation_invariant=True,
        compute_energy=True
    )
    ```

## Gabor Filter

Gabor filters are sinusoidal waves modulated by a Gaussian envelope. They are excellent for analyzing texture frequency and directionality.

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `sigma_mm` | `float` | Spatial scale (Gaussian width) in mm. |
| `lambda_mm` | `float` | Wavelength of the sinusoid in mm. |
| `gamma` | `float` | Spatial aspect ratio. Default: `1.0`. |
| `theta` | `float` | Orientation angle (if not rotation invariant). |
| `rotation_invariant` | `bool` | If `True`, aggregates responses over multiple orientations. |
| `delta_theta` | `float` | Orientation step in radians. **Required** when `rotation_invariant=True` (raises `ValueError` otherwise). |
| `average_over_planes` | `bool` | If `True`, averages the response over the three orthogonal planes. Default: `False` (axial plane only). |
| `spacing_mm` | `float` or `tuple` | Voxel spacing. The 2D kernel is scaled using the **true in-plane spacing of each plane**, so anisotropic voxels are handled correctly (this matters mainly when `average_over_planes=True`, where two of the three planes contain the through-plane axis). |

### Usage

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "gabor",
        "sigma_mm": 5.0,
        "lambda_mm": 2.0,
        "rotation_invariant": True,
        "delta_theta": 0.7853981633974483  # pi/4
    }}
    ```

=== "Direct API"

    ```python
    from pictologics.filters import gabor_filter
    
    response = gabor_filter(
        image_array,
        sigma_mm=5.0,
        lambda_mm=2.0,
        gamma=1.0,
        spacing_mm=(1.0, 1.0, 1.0),
        rotation_invariant=True,
        delta_theta=0.7853981633974483,  # pi/4
    )
    ```

## Separable Wavelets

Separable wavelet transforms decompose the image into low-frequency (L) and high-frequency (H) components along each axis (x, y, z). This results in 8 sub-bands (LLL, LLH, ..., HHH).

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `wavelet` | `str` | Wavelet family: `"haar"`, `"dbX"` (e.g., `db3`), `"coifX"` (e.g., `coif1`). |
| `level` | `int` | Decomposition level (usually `1`). |
| `decomposition` | `str` | Specific sub-band relative to the level (e.g., `"LLH"`, `"HHL"`). |
| `rotation_invariant` | `bool` | If `True`, averages responses across rotationally symmetric sub-bands (e.g., HLL, LHL, LLH). |

### Usage

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "wavelet",
        "wavelet": "coif1",
        "level": 1,
        "decomposition": "LLH",
        "rotation_invariant": True
    }}
    ```

=== "Direct API"

    ```python
    from pictologics.filters import wavelet_transform
    
    response = wavelet_transform(
        image_array,
        wavelet="coif1",
        decomposition="LLH",
        level=1,
        rotation_invariant=True
    )
    ```

## Simoncelli Wavelet

The Simoncelli wavelet (non-separable) provides isotropic texture analysis. It is valuable when the direction of texture is unknown or irrelevant.

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `level` | `int` | Decomposition level (1, 2, ...). |
| `boundary` | `str` | Boundary condition (`"periodic"`, `"zero"`, `"nearest"`, `"mirror"`). Default: `"periodic"`. See [FFT filter boundaries](#fft-filter-boundaries) below. |

### Usage

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "simoncelli",
        "level": 1
    }}
    ```

=== "Direct API"

    ```python
    from pictologics.filters import simoncelli_wavelet
    
    response = simoncelli_wavelet(image_array, level=1)
    ```

## Riesz Transform

The Riesz transform provides a steerable filter bank. It can be combined with other filters (like LoG or Simoncelli) to analyze texture orientation and phase.

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `order` | `Tuple[int, ...]` | Order tuple `(l1, l2, l3)` specifying derivative order per axis, e.g. `(1, 0, 0)`. |
| `variant` | `str` | Pipeline-level dispatch key (not a `riesz_transform` kwarg): `"base"` calls `riesz_transform` directly, `"log"` dispatches to `riesz_log` (requires `sigma_mm`), `"simoncelli"` dispatches to `riesz_simoncelli`. |
| `boundary` | `str` | Boundary condition (`"periodic"`, `"zero"`, `"nearest"`, `"mirror"`). Default: `"periodic"`. Accepted by all three variants; for `"log"` it is also forwarded to the internal LoG stage. See [FFT filter boundaries](#fft-filter-boundaries) below. |

### Usage

=== "Pipeline"

    ```python
    {"step": "filter", "params": {
        "type": "riesz",
        "order": [2, 0, 0],
        "variant": "log",
        "sigma_mm": 3.0  # required by riesz_log
    }}
    ```

=== "Direct API"

    ```python
    from pictologics.filters import riesz_transform
    
    response = riesz_transform(image_array, order=(2, 0, 0))
    ```

    The `"log"` and `"simoncelli"` variants dispatched by the Pipeline's `variant` parameter are
    also directly callable — `riesz_log()` requires `sigma_mm`:

    ```python
    from pictologics.filters import riesz_log, riesz_simoncelli

    # Riesz transform of a LoG-filtered image
    response = riesz_log(
        image_array,
        sigma_mm=3.0,
        spacing_mm=(1.0, 1.0, 1.0),
        order=(1, 0, 0),
    )

    # Riesz transform of a Simoncelli-wavelet-filtered image
    response = riesz_simoncelli(image_array, level=1, order=(1, 0, 0))
    ```

## FFT Filter Boundaries

The Simoncelli and Riesz filters operate in the Fourier domain, which is inherently
periodic. They still accept a `boundary` argument, honoured as follows:

- **`"periodic"` (default)** — the FFT is applied directly to the image. This is the
  fastest path and the one IBSI specifies for the Simoncelli tests.
- **`"zero"` / `"nearest"` / `"mirror"`** — the volume is padded with the requested
  mode, filtered, and cropped back (a *pad-filter-crop* procedure). For composite
  filters (`riesz_log`, `riesz_simoncelli`) the padding is applied once around the
  whole chain.

!!! note "Requested vs. effective boundary"
    Because the transform remains periodic on the *padded* domain, a non-periodic
    boundary is honoured **approximately** rather than exactly — the Riesz kernel in
    particular has power-law tails, so no finite pad removes truncation error
    entirely. The capability metadata reports this as `as_specified_via_padding`
    (see [`FILTER_CAPABILITIES`](../api/filters.md#capability-metadata)), and each
    pipeline filter step records both the requested and the effective boundary in the
    run log.

    Requesting the boundary a test specifies measurably improves agreement with the
    IBSI reference response maps — for example, using nearest-value padding for IBSI 2
    Phase 1 test `10.b.1` reduces the maximum absolute error roughly four-fold.

An unsupported boundary value raises `ValueError` rather than silently falling back.

!!! warning "Cost of a non-periodic boundary"
    Padding enlarges the array before the transform, so a non-periodic boundary is
    substantially slower than the periodic default — on a 64³ input the padded volume
    is roughly 5× the voxels and the call takes about 3–4× as long. The padding width
    grows with the filter's scale (Simoncelli level, LoG sigma), and accuracy improves
    steadily with it, so this is a genuine accuracy-versus-speed trade rather than
    avoidable overhead. Leave the boundary periodic unless you specifically need
    another one.

## Source Masking for Sentinel Values

When images contain **sentinel values** (e.g., -2048 HU for outside-FOV regions), these artificial values can contaminate filter responses. All Pictologics filters support a `source_mask` parameter to handle this. Note that source masking protects **filtering (and resampling) only** — you still need a `resegment` step to exclude sentinel voxels from the **ROI** used for feature extraction (see [Data Loading → Sentinel Handling](data_loading.md#handling-sentinel-na-values)).

### How It Works

| Filter Type | Strategy | Filters |
|:------------|:---------|:--------|
| **Spatial (normalized convolution)** | Weights convolution by valid-voxel contribution, then normalizes | Mean, LoG, Laws |
| **FFT-based (zero-fill)** | Zeros out sentinel voxels before frequency-domain processing | Gabor, Simoncelli, Riesz |
| **Separable wavelet** | Zeros out sentinel voxels before decomposition | Wavelet |

### Pipeline Usage

When using the pipeline, source masking is handled **automatically** based on the `source_mode` configuration. No changes to filter steps are needed — but remember to include `resegment` to also exclude sentinels from the ROI:

```python
pipeline.add_config("filtered_features", [
    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0)}},
    {"step": "filter", "params": {"type": "log", "sigma_mm": 3.0}},  # source_mask passed automatically
    {"step": "resegment", "params": {"range_min": -100, "range_max": 3000}},  # exclude sentinels from ROI
    {"step": "extract_features", "params": {"families": ["intensity", "texture"]}},
], source_mode="auto")
```

### Direct API Usage

When using filters directly, pass `source_mask` explicitly:

```python
from pictologics.filters import mean_filter, laplacian_of_gaussian

# Create a boolean mask (True = valid)
valid_mask = image_array > -1000

# Spatial filters return (response, valid_mask) when source_mask is provided
response, output_valid = mean_filter(image_array, support=15, source_mask=valid_mask)
response, output_valid = laplacian_of_gaussian(image_array, sigma_mm=3.0, source_mask=valid_mask)

# FFT-based filters accept source_mask but return only the response
from pictologics.filters import gabor_filter
response = gabor_filter(image_array, sigma_mm=5.0, lambda_mm=2.0, source_mask=valid_mask)
```

See the **[Quick Start](quick_start.md)** guide for a complete sentinel workflow example.
