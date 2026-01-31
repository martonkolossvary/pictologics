# Benchmarks


## Performance Benchmarks

### Benchmark Configuration

Comparisons between **Pictologics** and **PyRadiomics** (single-thread parity). 

**Test Data Generation:**

- **Texture**: 3D correlated noise generated using Gaussian smoothing.
- **Mask**: Blob-like structures generated via thresholded smooth noise with random holes.
- **Voxel Distribution**: Mean=486.04, Std=90.24, Min=0.00, Max=1000.00.

### Hardware Used for Calculations

- **Hardware**: Apple M4 Pro, 14 cores, 48 GB
- **OS**: macOS 26.2 (arm64)
- **Python**: 3.12.10
- **Core deps**: pictologics 0.3.1, numpy 2.2.6, scipy 1.17.0, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
- **PyRadiomics stack (parity runs)**: pyradiomics 3.1.1.dev111+g8ed579383, SimpleITK 2.5.3
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)

Note: the benchmark script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation overhead in the measured runtimes. All calculations are repeated 5 times and the average runtime is reported.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0308 s                | 0.84 MB                |
|     25 | FBS 25.0         | 0.0299 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0291 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0285 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0293 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0299 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.2846 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.2816 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.2814 s                | 6.33 MB                |
|     50 | FBN 16           | 1.2738 s                | 6.33 MB                |
|     50 | FBN 32           | 1.2653 s                | 6.33 MB                |
|     50 | FBN 64           | 1.2743 s                | 6.33 MB                |
|     75 | FBS 10.0         | Not calculated          | Not calculated         |
|     75 | FBS 25.0         | Not calculated          | Not calculated         |
|     75 | FBS 50.0         | Not calculated          | Not calculated         |
|     75 | FBN 16           | Not calculated          | Not calculated         |
|     75 | FBN 32           | Not calculated          | Not calculated         |
|     75 | FBN 64           | Not calculated          | Not calculated         |
|    100 | FBS 10.0         | Not calculated          | Not calculated         |
|    100 | FBS 25.0         | Not calculated          | Not calculated         |
|    100 | FBS 50.0         | Not calculated          | Not calculated         |
|    100 | FBN 16           | Not calculated          | Not calculated         |
|    100 | FBN 32           | Not calculated          | Not calculated         |
|    100 | FBN 64           | Not calculated          | Not calculated         |


### Morphology

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Morphology time](assets/benchmarks/morphology_execution_time_log.png)](assets/benchmarks/morphology_execution_time_log.png) | [![Morphology speedup](assets/benchmarks/morphology_speedup_factor.png)](assets/benchmarks/morphology_speedup_factor.png) |

**Pictologics-only morphology families (intensity-weighted morphology):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0041 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0040 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0041 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0041 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0041 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0043 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0109 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0106 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0110 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0108 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0110 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0108 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0171 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0173 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0172 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0174 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0173 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0172 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0330 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0336 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0337 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0333 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0336 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0338 s                | 20.46 MB               |


### Texture

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Texture time](assets/benchmarks/texture_execution_time_log.png)](assets/benchmarks/texture_execution_time_log.png) | [![Texture speedup](assets/benchmarks/texture_speedup_factor.png)](assets/benchmarks/texture_speedup_factor.png) |

**Pictologics-only texture families (GLDZM):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0002 s                | 0.15 MB                |
|     25 | FBS 25.0         | 0.0002 s                | 0.13 MB                |
|     25 | FBS 50.0         | 0.0002 s                | 0.13 MB                |
|     25 | FBN 16           | 0.0002 s                | 0.13 MB                |
|     25 | FBN 32           | 0.0002 s                | 0.13 MB                |
|     25 | FBN 64           | 0.0002 s                | 0.14 MB                |
|     50 | FBS 10.0         | 0.0003 s                | 0.16 MB                |
|     50 | FBS 25.0         | 0.0002 s                | 0.14 MB                |
|     50 | FBS 50.0         | 0.0003 s                | 0.13 MB                |
|     50 | FBN 16           | 0.0002 s                | 0.13 MB                |
|     50 | FBN 32           | 0.0002 s                | 0.14 MB                |
|     50 | FBN 64           | 0.0002 s                | 0.15 MB                |
|     75 | FBS 10.0         | 0.0003 s                | 0.21 MB                |
|     75 | FBS 25.0         | 0.0003 s                | 0.16 MB                |
|     75 | FBS 50.0         | 0.0003 s                | 0.14 MB                |
|     75 | FBN 16           | 0.0003 s                | 0.14 MB                |
|     75 | FBN 32           | 0.0003 s                | 0.15 MB                |
|     75 | FBN 64           | 0.0003 s                | 0.18 MB                |
|    100 | FBS 10.0         | 0.0004 s                | 0.20 MB                |
|    100 | FBS 25.0         | 0.0004 s                | 0.16 MB                |
|    100 | FBS 50.0         | 0.0003 s                | 0.14 MB                |
|    100 | FBN 16           | 0.0003 s                | 0.14 MB                |
|    100 | FBN 32           | 0.0005 s                | 0.15 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.17 MB                |


### Filters

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Filters time](assets/benchmarks/filters_execution_time_log.png)](assets/benchmarks/filters_execution_time_log.png) | [![Filters speedup](assets/benchmarks/filters_speedup_factor.png)](assets/benchmarks/filters_speedup_factor.png) |

**Pictologics-only filters (Gabor, Laws, Simoncelli, Riesz, Mean):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0088 s                | 1.70 MB                |
|     25 | FBS 25.0         | 0.0092 s                | 1.70 MB                |
|     25 | FBS 50.0         | 0.0086 s                | 1.70 MB                |
|     25 | FBN 16           | 0.0094 s                | 1.70 MB                |
|     25 | FBN 32           | 0.0091 s                | 1.70 MB                |
|     25 | FBN 64           | 0.0088 s                | 1.70 MB                |
|     50 | FBS 10.0         | 0.0285 s                | 13.50 MB               |
|     50 | FBS 25.0         | 0.0273 s                | 13.50 MB               |
|     50 | FBS 50.0         | 0.0272 s                | 13.50 MB               |
|     50 | FBN 16           | 0.0280 s                | 13.50 MB               |
|     50 | FBN 32           | 0.0285 s                | 13.50 MB               |
|     50 | FBN 64           | 0.0277 s                | 13.50 MB               |
|     75 | FBS 10.0         | 0.0555 s                | 45.50 MB               |
|     75 | FBS 25.0         | 0.0652 s                | 45.52 MB               |
|     75 | FBS 50.0         | 0.0562 s                | 45.50 MB               |
|     75 | FBN 16           | 0.0557 s                | 45.50 MB               |
|     75 | FBN 32           | 0.0563 s                | 45.50 MB               |
|     75 | FBN 64           | 0.0561 s                | 45.50 MB               |
|    100 | FBS 10.0         | 0.1023 s                | 107.80 MB              |
|    100 | FBS 25.0         | 0.1029 s                | 107.80 MB              |
|    100 | FBS 50.0         | 0.1026 s                | 107.80 MB              |
|    100 | FBN 16           | 0.1041 s                | 107.80 MB              |
|    100 | FBN 32           | 0.1053 s                | 107.80 MB              |
|    100 | FBN 64           | 0.1061 s                | 107.80 MB              |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Filters    |     25 | FBN 16           | 0.0011 s           | 0.0045 s           | 4.12x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 32           | 0.0011 s           | 0.0044 s           | 3.98x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 64           | 0.0010 s           | 0.0048 s           | 4.73x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 10.0         | 0.0011 s           | 0.0045 s           | 4.20x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 25.0         | 0.0011 s           | 0.0045 s           | 4.22x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 50.0         | 0.0010 s           | 0.0046 s           | 4.70x     | 0.48 MB           | 0.94 MB           |
| Filters    |     50 | FBN 16           | 0.0058 s           | 0.0099 s           | 1.69x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 32           | 0.0056 s           | 0.0096 s           | 1.72x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 64           | 0.0055 s           | 0.0098 s           | 1.77x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 10.0         | 0.0056 s           | 0.0097 s           | 1.72x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 25.0         | 0.0055 s           | 0.0096 s           | 1.75x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 50.0         | 0.0056 s           | 0.0097 s           | 1.73x     | 3.82 MB           | 6.68 MB           |
| Filters    |     75 | FBN 16           | 0.0184 s           | 0.0273 s           | 1.49x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 32           | 0.0182 s           | 0.0255 s           | 1.40x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 64           | 0.0186 s           | 0.0257 s           | 1.38x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 10.0         | 0.0184 s           | 0.0255 s           | 1.39x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 25.0         | 0.0185 s           | 0.0261 s           | 1.41x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 50.0         | 0.0183 s           | 0.0258 s           | 1.41x     | 12.88 MB          | 23.38 MB          |
| Filters    |    100 | FBN 16           | 0.0427 s           | 0.0521 s           | 1.22x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 32           | 0.0428 s           | 0.0522 s           | 1.22x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 64           | 0.0429 s           | 0.0519 s           | 1.21x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 10.0         | 0.0427 s           | 0.0518 s           | 1.21x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 25.0         | 0.0429 s           | 0.0526 s           | 1.22x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 50.0         | 0.0430 s           | 0.0520 s           | 1.21x     | 30.52 MB          | 53.41 MB          |
| Intensity  |     25 | FBN 16           | 0.0008 s           | 0.0128 s           | 16.54x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0132 s           | 16.49x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0009 s           | 0.0142 s           | 16.46x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0020 s           | 0.0139 s           | 6.93x     | 0.25 MB           | 0.75 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0132 s           | 15.98x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0008 s           | 0.0128 s           | 16.29x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0029 s           | 0.0653 s           | 22.61x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0029 s           | 0.0634 s           | 21.77x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0029 s           | 0.0647 s           | 22.13x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0031 s           | 0.0673 s           | 21.81x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0029 s           | 0.0627 s           | 21.91x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0028 s           | 0.0629 s           | 22.65x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0106 s           | 0.2478 s           | 23.32x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0116 s           | 0.2518 s           | 21.68x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0111 s           | 0.2549 s           | 22.91x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0115 s           | 0.2546 s           | 22.09x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0110 s           | 0.2505 s           | 22.72x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0108 s           | 0.2501 s           | 23.17x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0205 s           | 0.5010 s           | 24.41x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0215 s           | 0.5092 s           | 23.72x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0224 s           | 0.5277 s           | 23.54x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0233 s           | 0.5191 s           | 22.26x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0214 s           | 0.5186 s           | 24.18x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0210 s           | 0.5089 s           | 24.28x    | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0041 s           | 0.0552 s           | 13.34x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0039 s           | 0.0555 s           | 14.29x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0041 s           | 0.0600 s           | 14.51x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0059 s           | 0.0578 s           | 9.80x     | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0038 s           | 0.0552 s           | 14.39x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0040 s           | 0.0536 s           | 13.43x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0104 s           | 0.9549 s           | 91.63x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0107 s           | 0.9442 s           | 88.64x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0108 s           | 0.9481 s           | 87.89x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0109 s           | 0.9493 s           | 87.17x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0105 s           | 0.9370 s           | 89.51x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0108 s           | 0.9404 s           | 86.74x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0170 s           | 1.6906 s           | 99.40x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0180 s           | 1.6791 s           | 93.43x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0169 s           | 1.6808 s           | 99.51x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0171 s           | 1.7180 s           | 100.54x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0170 s           | 1.7095 s           | 100.64x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0172 s           | 1.6944 s           | 98.26x    | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0334 s           | 8.3787 s           | 250.84x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0332 s           | 8.3449 s           | 251.43x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0338 s           | 8.3437 s           | 247.12x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0333 s           | 8.5437 s           | 256.69x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0331 s           | 8.4334 s           | 254.90x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0333 s           | 8.4154 s           | 252.61x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0050 s           | 0.0489 s           | 9.83x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0054 s           | 0.0523 s           | 9.63x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0056 s           | 0.0672 s           | 11.95x    | 4.53 MB           | 1.95 MB           |
| Texture    |     25 | FBS 10.0         | 0.0072 s           | 0.0957 s           | 13.29x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0052 s           | 0.0537 s           | 10.26x    | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0051 s           | 0.0488 s           | 9.59x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0223 s           | 0.2953 s           | 13.25x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0236 s           | 0.2995 s           | 12.67x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0177 s           | 0.3159 s           | 17.85x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0168 s           | 0.3482 s           | 20.76x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0222 s           | 0.3027 s           | 13.63x    | 19.30 MB          | 6.10 MB           |
| Texture    |     50 | FBS 50.0         | 0.0256 s           | 0.2978 s           | 11.65x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0772 s           | 1.1898 s           | 15.40x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0762 s           | 1.1950 s           | 15.69x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0712 s           | 1.2108 s           | 17.01x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0483 s           | 1.2486 s           | 25.83x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0703 s           | 1.2092 s           | 17.21x    | 89.44 MB          | 25.80 MB          |
| Texture    |     75 | FBS 50.0         | 0.0741 s           | 1.2021 s           | 16.23x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1634 s           | 2.4462 s           | 14.97x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.1812 s           | 2.4564 s           | 13.56x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.1989 s           | 2.4925 s           | 12.53x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.0992 s           | 2.5416 s           | 25.61x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.1862 s           | 2.4889 s           | 13.36x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1802 s           | 2.4781 s           | 13.75x    | 229.77 MB         | 68.48 MB          |

