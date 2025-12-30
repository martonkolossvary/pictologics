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
- **Core deps**: pictologics 0.1.0, numpy 2.3.5, scipy 1.16.3, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
- **PyRadiomics stack (parity runs)**: pyradiomics 3.1.1.dev111+g8ed579383, SimpleITK 2.5.3
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)

Note: the benchmark script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation overhead in the measured runtimes.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0324 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0267 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0277 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0281 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0289 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0301 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.3024 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.2929 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.2979 s                | 6.33 MB                |
|     50 | FBN 16           | 1.2876 s                | 6.33 MB                |
|     50 | FBN 32           | 1.2959 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3105 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0041 s                | 1.11 MB                |
|     25 | FBS 25.0         | 0.0042 s                | 1.11 MB                |
|     25 | FBS 50.0         | 0.0042 s                | 1.11 MB                |
|     25 | FBN 16           | 0.0044 s                | 1.11 MB                |
|     25 | FBN 32           | 0.0040 s                | 1.10 MB                |
|     25 | FBN 64           | 0.0038 s                | 1.11 MB                |
|     50 | FBS 10.0         | 0.0106 s                | 5.37 MB                |
|     50 | FBS 25.0         | 0.0106 s                | 5.37 MB                |
|     50 | FBS 50.0         | 0.0107 s                | 5.37 MB                |
|     50 | FBN 16           | 0.0106 s                | 5.37 MB                |
|     50 | FBN 32           | 0.0107 s                | 5.37 MB                |
|     50 | FBN 64           | 0.0105 s                | 5.37 MB                |
|     75 | FBS 10.0         | 0.0170 s                | 8.77 MB                |
|     75 | FBS 25.0         | 0.0168 s                | 8.77 MB                |
|     75 | FBS 50.0         | 0.0165 s                | 8.77 MB                |
|     75 | FBN 16           | 0.0159 s                | 8.77 MB                |
|     75 | FBN 32           | 0.0164 s                | 8.77 MB                |
|     75 | FBN 64           | 0.0159 s                | 8.77 MB                |
|    100 | FBS 10.0         | 0.0317 s                | 20.40 MB               |
|    100 | FBS 25.0         | 0.0325 s                | 20.40 MB               |
|    100 | FBS 50.0         | 0.0319 s                | 20.40 MB               |
|    100 | FBN 16           | 0.0320 s                | 20.40 MB               |
|    100 | FBN 32           | 0.0319 s                | 20.40 MB               |
|    100 | FBN 64           | 0.0343 s                | 20.40 MB               |


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
|     50 | FBS 25.0         | 0.0003 s                | 0.08 MB                |
|     50 | FBS 50.0         | 0.0002 s                | 0.07 MB                |
|     50 | FBN 16           | 0.0002 s                | 0.07 MB                |
|     50 | FBN 32           | 0.0003 s                | 0.07 MB                |
|     50 | FBN 64           | 0.0003 s                | 0.09 MB                |
|     75 | FBS 10.0         | 0.0003 s                | 0.15 MB                |
|     75 | FBS 25.0         | 0.0003 s                | 0.10 MB                |
|     75 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|     75 | FBN 16           | 0.0003 s                | 0.08 MB                |
|     75 | FBN 32           | 0.0003 s                | 0.09 MB                |
|     75 | FBN 64           | 0.0003 s                | 0.12 MB                |
|    100 | FBS 10.0         | 0.0004 s                | 0.14 MB                |
|    100 | FBS 25.0         | 0.0003 s                | 0.09 MB                |
|    100 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|    100 | FBN 16           | 0.0003 s                | 0.07 MB                |
|    100 | FBN 32           | 0.0003 s                | 0.09 MB                |
|    100 | FBN 64           | 0.0003 s                | 0.11 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0008 s           | 0.0132 s           | 17.38x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0009 s           | 0.0128 s           | 13.98x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0008 s           | 0.0125 s           | 14.87x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0008 s           | 0.0135 s           | 17.39x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0128 s           | 15.20x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0009 s           | 0.0130 s           | 14.96x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0029 s           | 0.0625 s           | 21.72x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0029 s           | 0.0633 s           | 21.65x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0031 s           | 0.0657 s           | 21.08x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0028 s           | 0.0655 s           | 23.15x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0035 s           | 0.0643 s           | 18.22x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0030 s           | 0.0634 s           | 21.45x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0104 s           | 0.2441 s           | 23.46x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0108 s           | 0.2493 s           | 23.12x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0107 s           | 0.2501 s           | 23.28x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0116 s           | 0.2572 s           | 22.20x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0112 s           | 0.2538 s           | 22.62x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0110 s           | 0.2499 s           | 22.71x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0208 s           | 0.5077 s           | 24.36x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0214 s           | 0.5039 s           | 23.55x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0248 s           | 0.5128 s           | 20.70x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0225 s           | 0.5146 s           | 22.83x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0231 s           | 0.5151 s           | 22.29x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0209 s           | 0.5127 s           | 24.50x    | 12.16 MB          | 39.00 MB          |
| Morphology |     25 | FBN 16           | 0.0038 s           | 0.0521 s           | 13.56x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0040 s           | 0.0512 s           | 12.79x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0038 s           | 0.0501 s           | 13.02x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0041 s           | 0.0510 s           | 12.58x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0041 s           | 0.0507 s           | 12.30x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0037 s           | 0.0520 s           | 13.91x    | 1.11 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0104 s           | 0.9495 s           | 91.09x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0103 s           | 0.9449 s           | 91.49x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0104 s           | 0.9577 s           | 91.89x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0104 s           | 0.9545 s           | 91.58x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0100 s           | 0.9476 s           | 95.23x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0101 s           | 0.9420 s           | 93.32x    | 5.37 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0159 s           | 1.6769 s           | 105.70x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0162 s           | 1.6589 s           | 102.17x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0160 s           | 1.6684 s           | 104.20x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0168 s           | 1.7313 s           | 102.85x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0164 s           | 1.7055 s           | 104.18x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0166 s           | 1.7286 s           | 103.90x   | 8.77 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0310 s           | 8.2976 s           | 267.42x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0315 s           | 8.2656 s           | 262.18x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0338 s           | 8.2164 s           | 242.93x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0310 s           | 8.4072 s           | 270.94x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0343 s           | 8.3736 s           | 244.10x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0315 s           | 8.2861 s           | 262.78x   | 20.40 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0049 s           | 0.0478 s           | 9.73x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0057 s           | 0.0508 s           | 8.94x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0053 s           | 0.0621 s           | 11.77x    | 4.53 MB           | 1.94 MB           |
| Texture    |     25 | FBS 10.0         | 0.0073 s           | 0.0889 s           | 12.25x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0058 s           | 0.0517 s           | 8.89x     | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0050 s           | 0.0482 s           | 9.69x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0220 s           | 0.2931 s           | 13.35x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0237 s           | 0.2999 s           | 12.67x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0180 s           | 0.3189 s           | 17.69x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0178 s           | 0.3517 s           | 19.72x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0237 s           | 0.3033 s           | 12.78x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0237 s           | 0.2968 s           | 12.53x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0702 s           | 1.1862 s           | 16.90x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0781 s           | 1.1797 s           | 15.10x    | 86.59 MB          | 26.22 MB          |
| Texture    |     75 | FBN 64           | 0.0637 s           | 1.2079 s           | 18.96x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0486 s           | 1.2407 s           | 25.54x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0803 s           | 1.1946 s           | 14.87x    | 89.44 MB          | 25.80 MB          |
| Texture    |     75 | FBS 50.0         | 0.0753 s           | 1.2248 s           | 16.27x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1675 s           | 2.4330 s           | 14.52x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.1859 s           | 2.4364 s           | 13.10x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.1922 s           | 2.4449 s           | 12.72x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.0991 s           | 2.5537 s           | 25.77x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.2054 s           | 2.4582 s           | 11.97x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1926 s           | 2.4373 s           | 12.65x    | 229.77 MB         | 68.48 MB          |

