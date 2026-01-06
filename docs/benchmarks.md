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
|     25 | FBS 10.0         | 0.0274 s                | 0.81 MB                |
|     25 | FBS 25.0         | 0.0271 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0278 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0289 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0286 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0271 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.3368 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.3378 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.3259 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3124 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3183 s                | 6.33 MB                |
|     50 | FBN 64           | 1.3167 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0042 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0039 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0041 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0041 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0041 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0041 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0101 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0102 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0107 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0104 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0105 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0108 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0168 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0166 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0166 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0172 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0165 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0164 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0325 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0317 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0316 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0317 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0325 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0320 s                | 20.46 MB               |


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
|     50 | FBS 50.0         | 0.0003 s                | 0.07 MB                |
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
|    100 | FBS 25.0         | 0.0004 s                | 0.09 MB                |
|    100 | FBS 50.0         | 0.0003 s                | 0.08 MB                |
|    100 | FBN 16           | 0.0004 s                | 0.07 MB                |
|    100 | FBN 32           | 0.0003 s                | 0.09 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.11 MB                |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Intensity  |     25 | FBN 16           | 0.0007 s           | 0.0122 s           | 16.98x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0126 s           | 16.57x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0009 s           | 0.0131 s           | 15.18x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0007 s           | 0.0134 s           | 17.95x    | 0.24 MB           | 0.74 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0125 s           | 15.80x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0007 s           | 0.0126 s           | 17.73x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0029 s           | 0.0625 s           | 21.39x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0028 s           | 0.0628 s           | 22.06x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0029 s           | 0.0642 s           | 21.84x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0029 s           | 0.0637 s           | 22.20x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0028 s           | 0.0624 s           | 22.13x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0030 s           | 0.0646 s           | 21.67x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0106 s           | 0.2456 s           | 23.17x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0111 s           | 0.2502 s           | 22.61x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0114 s           | 0.2509 s           | 21.99x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0115 s           | 0.2526 s           | 21.92x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0110 s           | 0.2515 s           | 22.93x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0110 s           | 0.2547 s           | 23.15x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0204 s           | 0.5069 s           | 24.91x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0216 s           | 0.5116 s           | 23.65x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0218 s           | 0.5145 s           | 23.56x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0228 s           | 0.5270 s           | 23.11x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0213 s           | 0.5037 s           | 23.69x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0204 s           | 0.5191 s           | 25.39x    | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0038 s           | 0.0501 s           | 13.31x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0038 s           | 0.0508 s           | 13.26x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0039 s           | 0.0508 s           | 12.99x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0040 s           | 0.0513 s           | 12.95x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0039 s           | 0.0497 s           | 12.66x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0040 s           | 0.0497 s           | 12.58x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0104 s           | 0.9506 s           | 91.79x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0102 s           | 0.9455 s           | 92.27x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0107 s           | 0.9464 s           | 88.34x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0100 s           | 0.9143 s           | 91.23x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0101 s           | 0.9339 s           | 92.37x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0102 s           | 0.9440 s           | 92.57x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0164 s           | 1.6907 s           | 103.05x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0165 s           | 1.6991 s           | 102.81x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0164 s           | 1.6896 s           | 103.33x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0165 s           | 1.7352 s           | 105.47x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0165 s           | 1.7137 s           | 103.67x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0163 s           | 1.7059 s           | 104.39x   | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0315 s           | 8.4146 s           | 266.92x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0321 s           | 8.4263 s           | 262.89x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0318 s           | 8.4179 s           | 265.02x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0326 s           | 8.5908 s           | 263.90x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0313 s           | 8.4611 s           | 270.35x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0311 s           | 8.4232 s           | 270.57x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0049 s           | 0.0452 s           | 9.30x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0051 s           | 0.0505 s           | 9.80x     | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0056 s           | 0.0635 s           | 11.31x    | 4.53 MB           | 1.94 MB           |
| Texture    |     25 | FBS 10.0         | 0.0072 s           | 0.0916 s           | 12.66x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0051 s           | 0.0513 s           | 10.02x    | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0051 s           | 0.0461 s           | 9.06x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0210 s           | 0.2953 s           | 14.03x    | 19.85 MB          | 6.28 MB           |
| Texture    |     50 | FBN 32           | 0.0220 s           | 0.2981 s           | 13.54x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0182 s           | 0.3131 s           | 17.20x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0166 s           | 0.3431 s           | 20.62x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0217 s           | 0.3101 s           | 14.32x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0254 s           | 0.2953 s           | 11.63x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0705 s           | 1.2014 s           | 17.04x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0768 s           | 1.1928 s           | 15.54x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0646 s           | 1.2124 s           | 18.77x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0479 s           | 1.2394 s           | 25.87x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0763 s           | 1.2012 s           | 15.75x    | 89.44 MB          | 25.79 MB          |
| Texture    |     75 | FBS 50.0         | 0.0737 s           | 1.1793 s           | 15.99x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1701 s           | 2.4634 s           | 14.48x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.1919 s           | 2.4812 s           | 12.93x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.1844 s           | 2.4722 s           | 13.41x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.1012 s           | 2.5159 s           | 24.85x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.1872 s           | 2.4473 s           | 13.08x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.1862 s           | 2.4768 s           | 13.30x    | 229.77 MB         | 68.48 MB          |

