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
- **Core deps**: pictologics 0.2.0, numpy 2.2.6, scipy 1.17.0, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
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
|     25 | FBS 10.0         | 0.0322 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0285 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0313 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0314 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0280 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0277 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.3945 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.3965 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.3628 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3578 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3811 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3948 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0044 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0042 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0045 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0041 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0041 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0045 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0107 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0106 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0109 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0105 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0106 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0122 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0168 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0176 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0173 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0167 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0173 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0163 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0324 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0318 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0325 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0316 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0314 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0316 s                | 20.46 MB               |


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
|     50 | FBS 25.0         | 0.0003 s                | 0.14 MB                |
|     50 | FBS 50.0         | 0.0003 s                | 0.13 MB                |
|     50 | FBN 16           | 0.0002 s                | 0.13 MB                |
|     50 | FBN 32           | 0.0002 s                | 0.14 MB                |
|     50 | FBN 64           | 0.0003 s                | 0.15 MB                |
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
|    100 | FBN 32           | 0.0004 s                | 0.15 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.17 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0009 s           | 0.0134 s           | 14.48x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0132 s           | 16.89x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0008 s           | 0.0130 s           | 16.52x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0011 s           | 0.0144 s           | 12.51x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0131 s           | 16.84x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0008 s           | 0.0128 s           | 15.60x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0030 s           | 0.0625 s           | 20.70x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0030 s           | 0.0641 s           | 21.31x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0030 s           | 0.0666 s           | 22.12x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0032 s           | 0.0650 s           | 20.44x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0029 s           | 0.0654 s           | 22.66x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0030 s           | 0.0653 s           | 22.10x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0110 s           | 0.2501 s           | 22.71x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0117 s           | 0.2552 s           | 21.77x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0116 s           | 0.2525 s           | 21.85x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0119 s           | 0.2600 s           | 21.87x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0122 s           | 0.2594 s           | 21.25x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0112 s           | 0.2481 s           | 22.24x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0204 s           | 0.5099 s           | 24.97x    | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBN 32           | 0.0211 s           | 0.5066 s           | 24.03x    | 12.16 MB          | 39.00 MB          |
| Intensity  |    100 | FBN 64           | 0.0216 s           | 0.5245 s           | 24.29x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0232 s           | 0.5201 s           | 22.38x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0215 s           | 0.5154 s           | 24.00x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0208 s           | 0.5123 s           | 24.66x    | 12.16 MB          | 39.00 MB          |
| Morphology |     25 | FBN 16           | 0.0040 s           | 0.0540 s           | 13.46x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0041 s           | 0.0516 s           | 12.46x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0042 s           | 0.0534 s           | 12.61x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0043 s           | 0.0530 s           | 12.19x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0039 s           | 0.0528 s           | 13.67x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0041 s           | 0.0531 s           | 12.83x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0103 s           | 0.9641 s           | 93.82x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0106 s           | 0.9699 s           | 91.73x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0110 s           | 0.9676 s           | 88.13x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0122 s           | 0.9482 s           | 77.92x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0103 s           | 0.9579 s           | 92.92x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0106 s           | 0.9629 s           | 90.97x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0163 s           | 1.7555 s           | 107.80x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0173 s           | 1.7348 s           | 100.34x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0164 s           | 1.7378 s           | 106.06x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0171 s           | 1.7636 s           | 103.17x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0173 s           | 1.7564 s           | 101.72x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0167 s           | 1.7535 s           | 104.94x   | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0314 s           | 8.3944 s           | 267.61x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0312 s           | 8.3838 s           | 268.90x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0316 s           | 8.4430 s           | 267.56x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0321 s           | 8.5900 s           | 267.67x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0316 s           | 8.4630 s           | 267.99x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0315 s           | 8.4259 s           | 267.37x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0053 s           | 0.0470 s           | 8.85x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0050 s           | 0.0511 s           | 10.19x    | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0056 s           | 0.0693 s           | 12.42x    | 4.53 MB           | 1.95 MB           |
| Texture    |     25 | FBS 10.0         | 0.0081 s           | 0.0944 s           | 11.63x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0054 s           | 0.0553 s           | 10.19x    | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0052 s           | 0.0490 s           | 9.44x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0220 s           | 0.3022 s           | 13.76x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0234 s           | 0.3064 s           | 13.12x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0196 s           | 0.3250 s           | 16.58x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0167 s           | 0.3506 s           | 21.01x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0265 s           | 0.3100 s           | 11.70x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0248 s           | 0.3035 s           | 12.22x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0725 s           | 1.1909 s           | 16.43x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0815 s           | 1.2173 s           | 14.94x    | 86.59 MB          | 26.22 MB          |
| Texture    |     75 | FBN 64           | 0.0674 s           | 1.2218 s           | 18.13x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0490 s           | 1.2728 s           | 26.00x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0797 s           | 1.2329 s           | 15.48x    | 89.44 MB          | 25.79 MB          |
| Texture    |     75 | FBS 50.0         | 0.0799 s           | 1.2019 s           | 15.04x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1692 s           | 2.4352 s           | 14.39x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.1891 s           | 2.4733 s           | 13.08x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.1834 s           | 2.4737 s           | 13.49x    | 204.71 MB         | 60.71 MB          |
| Texture    |    100 | FBS 10.0         | 0.1005 s           | 2.5358 s           | 25.22x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.1875 s           | 2.4714 s           | 13.18x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1938 s           | 2.4664 s           | 12.73x    | 229.77 MB         | 68.48 MB          |

