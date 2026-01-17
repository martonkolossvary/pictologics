# Benchmarks


## Performance Benchmarks

### Benchmark Configuration

Comparisons between **Pictologics** and **PyRadiomics** (single-thread parity). 

**Test Data Generation:**

- **Texture**: 3D correlated noise generated using Gaussian smoothing.
- **Mask**: Blob-like structures generated via thresholded smooth noise with random holes.
- **Voxel Distribution**: Mean=486.04, Std=90.24, Min=0.00, Max=1000.00.

### HARDWARE USED FOR CALCULATIONS

- **Hardware**: Apple M4 Pro, 14 cores, 48 GB
- **OS**: macOS 26.2 (arm64)
- **Python**: 3.12.10
- **Core deps**: pictologics 0.2.0, numpy 2.3.5, scipy 1.16.3, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
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
|     25 | FBS 10.0         | 0.0344 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0349 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0367 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0391 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0344 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0333 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.5596 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.5884 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.5698 s                | 6.33 MB                |
|     50 | FBN 16           | 1.6430 s                | 6.33 MB                |
|     50 | FBN 32           | 1.4384 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3535 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0045 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0045 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0055 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0050 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0042 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0047 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0118 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0118 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0116 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0118 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0116 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0114 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0186 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0183 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0179 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0182 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0185 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0178 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0375 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.3629 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0378 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0355 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0349 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0409 s                | 20.46 MB               |


### Texture

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Texture time](assets/benchmarks/texture_execution_time_log.png)](assets/benchmarks/texture_execution_time_log.png) | [![Texture speedup](assets/benchmarks/texture_speedup_factor.png)](assets/benchmarks/texture_speedup_factor.png) |

**Pictologics-only texture families (GLDZM):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0002 s                | 0.08 MB                |
|     25 | FBS 25.0         | 0.0002 s                | 0.07 MB                |
|     25 | FBS 50.0         | 0.0002 s                | 0.07 MB                |
|     25 | FBN 16           | 0.0002 s                | 0.07 MB                |
|     25 | FBN 32           | 0.0002 s                | 0.07 MB                |
|     25 | FBN 64           | 0.0002 s                | 0.07 MB                |
|     50 | FBS 10.0         | 0.0002 s                | 0.10 MB                |
|     50 | FBS 25.0         | 0.0002 s                | 0.08 MB                |
|     50 | FBS 50.0         | 0.0002 s                | 0.07 MB                |
|     50 | FBN 16           | 0.0003 s                | 0.07 MB                |
|     50 | FBN 32           | 0.0002 s                | 0.07 MB                |
|     50 | FBN 64           | 0.0002 s                | 0.09 MB                |
|     75 | FBS 10.0         | 0.0003 s                | 0.15 MB                |
|     75 | FBS 25.0         | 0.0003 s                | 0.10 MB                |
|     75 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|     75 | FBN 16           | 0.0003 s                | 0.08 MB                |
|     75 | FBN 32           | 0.0004 s                | 0.09 MB                |
|     75 | FBN 64           | 0.0003 s                | 0.12 MB                |
|    100 | FBS 10.0         | 0.0004 s                | 0.14 MB                |
|    100 | FBS 25.0         | 0.0004 s                | 0.09 MB                |
|    100 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|    100 | FBN 16           | 0.0003 s                | 0.07 MB                |
|    100 | FBN 32           | 0.0004 s                | 0.09 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.11 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0009 s           | 0.0149 s           | 17.21x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0141 s           | 17.23x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0009 s           | 0.0143 s           | 16.06x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0008 s           | 0.0184 s           | 22.34x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0148 s           | 18.48x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0009 s           | 0.0140 s           | 16.11x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0032 s           | 0.0719 s           | 22.56x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0030 s           | 0.0689 s           | 23.15x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0032 s           | 0.0655 s           | 20.19x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0033 s           | 0.0708 s           | 21.36x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0031 s           | 0.0700 s           | 22.82x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0030 s           | 0.0681 s           | 23.05x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0122 s           | 0.2649 s           | 21.78x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0124 s           | 0.2611 s           | 20.97x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0120 s           | 0.2630 s           | 21.83x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0133 s           | 0.2715 s           | 20.40x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0125 s           | 0.2699 s           | 21.58x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0113 s           | 0.2611 s           | 23.11x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0235 s           | 0.5418 s           | 23.02x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0247 s           | 0.5485 s           | 22.18x    | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBN 64           | 0.0259 s           | 0.5748 s           | 22.18x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0258 s           | 0.5547 s           | 21.51x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0370 s           | 0.5782 s           | 15.61x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0245 s           | 0.5654 s           | 23.12x    | 12.16 MB          | 39.00 MB          |
| Morphology |     25 | FBN 16           | 0.0045 s           | 0.0577 s           | 12.92x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0044 s           | 0.0566 s           | 12.84x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0043 s           | 0.0574 s           | 13.44x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0044 s           | 0.0574 s           | 13.20x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0043 s           | 0.0588 s           | 13.82x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0044 s           | 0.0590 s           | 13.35x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0112 s           | 0.9647 s           | 86.42x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0116 s           | 0.9726 s           | 84.21x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0113 s           | 0.9733 s           | 85.75x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0119 s           | 0.9760 s           | 81.92x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0112 s           | 0.9810 s           | 87.63x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0113 s           | 0.9850 s           | 86.92x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0178 s           | 1.7830 s           | 99.91x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0185 s           | 1.7776 s           | 96.35x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0177 s           | 1.7645 s           | 99.93x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0189 s           | 1.8000 s           | 95.46x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0184 s           | 1.7827 s           | 97.06x    | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0175 s           | 1.7830 s           | 101.90x   | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0353 s           | 9.0724 s           | 257.20x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0358 s           | 9.5607 s           | 266.79x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0357 s           | 9.0394 s           | 253.07x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0361 s           | 9.0817 s           | 251.46x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0482 s           | 9.1583 s           | 189.89x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0367 s           | 9.4211 s           | 256.69x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0058 s           | 0.0494 s           | 8.48x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0058 s           | 0.0528 s           | 9.12x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0061 s           | 0.0690 s           | 11.36x    | 4.53 MB           | 1.94 MB           |
| Texture    |     25 | FBS 10.0         | 0.0083 s           | 0.0976 s           | 11.82x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0062 s           | 0.0569 s           | 9.19x     | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0058 s           | 0.0542 s           | 9.40x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0280 s           | 0.3067 s           | 10.94x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0289 s           | 0.3102 s           | 10.72x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0192 s           | 0.3328 s           | 17.30x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0190 s           | 0.3613 s           | 19.01x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0270 s           | 0.3115 s           | 11.52x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0289 s           | 0.3128 s           | 10.82x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0869 s           | 1.2254 s           | 14.10x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0869 s           | 1.2666 s           | 14.58x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0736 s           | 1.2764 s           | 17.34x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0513 s           | 1.2966 s           | 25.30x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0893 s           | 1.2442 s           | 13.94x    | 89.44 MB          | 25.79 MB          |
| Texture    |     75 | FBS 50.0         | 0.0862 s           | 1.7134 s           | 19.88x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.2031 s           | 2.5784 s           | 12.70x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.2264 s           | 2.6213 s           | 11.58x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.2328 s           | 2.6522 s           | 11.39x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.1117 s           | 2.8208 s           | 25.26x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.4584 s           | 2.6375 s           | 5.75x     | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.2275 s           | 2.5777 s           | 11.33x    | 229.77 MB         | 68.48 MB          |

