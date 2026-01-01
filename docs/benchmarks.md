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

Note: the benchmark script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation overhead in the measured runtimes.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0348 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0393 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0334 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0324 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0317 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0290 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.4736 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.4931 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.4675 s                | 6.33 MB                |
|     50 | FBN 16           | 1.4439 s                | 6.33 MB                |
|     50 | FBN 32           | 1.2825 s                | 6.33 MB                |
|     50 | FBN 64           | 1.2885 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0046 s                | 1.10 MB                |
|     25 | FBS 25.0         | 0.0046 s                | 1.11 MB                |
|     25 | FBS 50.0         | 0.0045 s                | 1.11 MB                |
|     25 | FBN 16           | 0.0042 s                | 1.11 MB                |
|     25 | FBN 32           | 0.0043 s                | 1.10 MB                |
|     25 | FBN 64           | 0.0039 s                | 1.11 MB                |
|     50 | FBS 10.0         | 0.0112 s                | 5.37 MB                |
|     50 | FBS 25.0         | 0.0111 s                | 5.37 MB                |
|     50 | FBS 50.0         | 0.0108 s                | 5.37 MB                |
|     50 | FBN 16           | 0.0116 s                | 5.37 MB                |
|     50 | FBN 32           | 0.0103 s                | 5.37 MB                |
|     50 | FBN 64           | 0.0105 s                | 5.37 MB                |
|     75 | FBS 10.0         | 0.0169 s                | 8.77 MB                |
|     75 | FBS 25.0         | 0.0166 s                | 8.77 MB                |
|     75 | FBS 50.0         | 0.0169 s                | 8.77 MB                |
|     75 | FBN 16           | 0.0166 s                | 8.77 MB                |
|     75 | FBN 32           | 0.0165 s                | 8.77 MB                |
|     75 | FBN 64           | 0.0163 s                | 8.77 MB                |
|    100 | FBS 10.0         | 0.0317 s                | 20.40 MB               |
|    100 | FBS 25.0         | 0.0313 s                | 20.40 MB               |
|    100 | FBS 50.0         | 0.0317 s                | 20.40 MB               |
|    100 | FBN 16           | 0.0315 s                | 20.40 MB               |
|    100 | FBN 32           | 0.0361 s                | 20.40 MB               |
|    100 | FBN 64           | 0.0369 s                | 20.40 MB               |


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
|     50 | FBS 10.0         | 0.0003 s                | 0.10 MB                |
|     50 | FBS 25.0         | 0.0003 s                | 0.08 MB                |
|     50 | FBS 50.0         | 0.0002 s                | 0.07 MB                |
|     50 | FBN 16           | 0.0002 s                | 0.07 MB                |
|     50 | FBN 32           | 0.0002 s                | 0.07 MB                |
|     50 | FBN 64           | 0.0002 s                | 0.09 MB                |
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
|    100 | FBN 32           | 0.0004 s                | 0.09 MB                |
|    100 | FBN 64           | 0.0005 s                | 0.11 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0009 s           | 0.0133 s           | 15.54x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0009 s           | 0.0136 s           | 15.27x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0008 s           | 0.0137 s           | 16.54x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0010 s           | 0.0155 s           | 16.23x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0009 s           | 0.0147 s           | 15.54x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0010 s           | 0.0149 s           | 14.39x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0030 s           | 0.0668 s           | 22.22x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0030 s           | 0.0634 s           | 21.38x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0029 s           | 0.0631 s           | 21.46x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0034 s           | 0.0686 s           | 20.13x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0030 s           | 0.0684 s           | 22.63x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0030 s           | 0.0659 s           | 22.29x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0108 s           | 0.2480 s           | 22.95x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0111 s           | 0.2519 s           | 22.73x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0111 s           | 0.2526 s           | 22.67x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0117 s           | 0.2568 s           | 21.91x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0111 s           | 0.2545 s           | 22.89x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0112 s           | 0.2494 s           | 22.22x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0207 s           | 0.5035 s           | 24.34x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0251 s           | 0.5509 s           | 21.92x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0249 s           | 0.5636 s           | 22.65x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0228 s           | 0.5186 s           | 22.72x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0215 s           | 0.5095 s           | 23.69x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0208 s           | 0.5011 s           | 24.08x    | 12.16 MB          | 39.00 MB          |
| Morphology |     25 | FBN 16           | 0.0040 s           | 0.0564 s           | 13.95x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0038 s           | 0.0544 s           | 14.46x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0039 s           | 0.0562 s           | 14.25x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0045 s           | 0.0577 s           | 12.90x    | 1.11 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0046 s           | 0.0578 s           | 12.43x    | 1.10 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0044 s           | 0.0563 s           | 12.83x    | 1.11 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0110 s           | 0.9775 s           | 88.67x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0103 s           | 0.9486 s           | 92.36x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0103 s           | 0.9697 s           | 93.88x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0108 s           | 0.9822 s           | 90.94x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0107 s           | 0.9707 s           | 90.60x    | 5.37 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0110 s           | 0.9707 s           | 88.49x    | 5.37 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0165 s           | 1.7305 s           | 104.57x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0167 s           | 1.6964 s           | 101.38x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0163 s           | 1.6886 s           | 103.48x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0164 s           | 1.7370 s           | 106.21x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0161 s           | 1.7359 s           | 107.82x   | 8.77 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0163 s           | 1.7163 s           | 105.52x   | 8.77 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0314 s           | 8.4985 s           | 270.89x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0355 s           | 9.1379 s           | 257.26x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0356 s           | 9.1521 s           | 257.10x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0313 s           | 8.5800 s           | 273.77x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0310 s           | 8.5343 s           | 275.03x   | 20.40 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0314 s           | 8.6135 s           | 274.30x   | 20.40 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0051 s           | 0.0503 s           | 9.89x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0054 s           | 0.0513 s           | 9.57x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0054 s           | 0.0705 s           | 13.11x    | 4.53 MB           | 1.94 MB           |
| Texture    |     25 | FBS 10.0         | 0.0081 s           | 0.1057 s           | 13.08x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0059 s           | 0.0576 s           | 9.74x     | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0059 s           | 0.0526 s           | 8.97x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0251 s           | 0.3035 s           | 12.09x    | 19.85 MB          | 6.28 MB           |
| Texture    |     50 | FBN 32           | 0.0221 s           | 0.2972 s           | 13.47x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0183 s           | 0.3147 s           | 17.21x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0195 s           | 0.3573 s           | 18.33x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0255 s           | 0.3092 s           | 12.13x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0253 s           | 0.3064 s           | 12.12x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0730 s           | 1.1806 s           | 16.18x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0765 s           | 1.1871 s           | 15.51x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0648 s           | 1.2088 s           | 18.65x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0492 s           | 1.2615 s           | 25.64x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0789 s           | 1.1990 s           | 15.20x    | 89.44 MB          | 25.79 MB          |
| Texture    |     75 | FBS 50.0         | 0.0759 s           | 1.1932 s           | 15.71x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1738 s           | 2.6248 s           | 15.10x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.2323 s           | 2.6344 s           | 11.34x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.2226 s           | 2.5986 s           | 11.67x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.0976 s           | 2.4998 s           | 25.61x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.1941 s           | 2.4530 s           | 12.64x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1845 s           | 2.4470 s           | 13.26x    | 229.77 MB         | 68.48 MB          |

