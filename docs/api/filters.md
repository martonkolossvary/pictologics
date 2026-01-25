# Filters API

Pictologics provides IBSI 2-compliant convolutional filters for image response map generation.

## Overview

All filters can be used via the `RadiomicsPipeline` `filter` step or called directly:

```python
# Pipeline usage
{"step": "filter", "params": {"type": "log", "sigma_mm": 1.5}}

# Direct usage
from pictologics.filters import laplacian_of_gaussian, BoundaryCondition
response = laplacian_of_gaussian(image.array, sigma_mm=1.5, spacing_mm=image.spacing)
```

## Available Filters

| Filter | Function | Use Case |
|:-------|:---------|:---------|
| Mean | `mean_filter` | Local averaging |
| LoG | `laplacian_of_gaussian` | Edge/blob detection |
| Laws | `laws_filter` | Texture energy |
| Gabor | `gabor_filter` | Directional patterns |
| Wavelet | `wavelet_transform` | Multi-resolution analysis |
| Simoncelli | `simoncelli_wavelet` | Non-separable wavelet |
| Riesz | `riesz_transform`, `riesz_log`, `riesz_simoncelli` | Rotation-equivariant transforms |

## Boundary Conditions

::: pictologics.filters.BoundaryCondition

::: pictologics.filters.FilterResult

::: pictologics.filters.LAWS_KERNELS

## Filter Functions

::: pictologics.filters.mean_filter

::: pictologics.filters.laplacian_of_gaussian

::: pictologics.filters.laws_filter

::: pictologics.filters.gabor_filter

::: pictologics.filters.wavelet_transform

::: pictologics.filters.simoncelli_wavelet

::: pictologics.filters.riesz_transform

::: pictologics.filters.riesz_log

::: pictologics.filters.riesz_simoncelli
