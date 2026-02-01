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
- **Core deps**: pictologics 0.3.2, numpy 2.2.6, scipy 1.17.0, numba 0.62.1, pandas 2.3.3, matplotlib 3.10.7
- **PyRadiomics stack (parity runs)**: pyradiomics 3.1.1.dev111+g8ed579383, SimpleITK 2.5.3
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)

!!! info "Benchmark Methodology"

    **Reported values**: All timings are the **average of 5 runs** to reduce variance. The benchmark
    script explicitly calls `warmup_jit()` before timing to avoid including Numba compilation
    overhead in the measured runtimes.

    **Fair comparison**: For benchmarking, **all features within each family are calculated** to
    ensure a fair comparison with PyRadiomics. This represents the worst-case scenario where every
    feature is requested.

    **Real-world performance**: In practice, Pictologics uses **intelligent deduplication** to avoid
    redundant computation. When you run multiple configurations (e.g., comparing different
    discretisation methods), the pipeline automatically reuses preprocessing steps and feature
    calculations that would produce identical results. This can lead to significantly faster
    run times than shown here, especially when extracting features across many configurations.

### Intensity

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Intensity time](assets/benchmarks/intensity_execution_time_log.png)](assets/benchmarks/intensity_execution_time_log.png) | [![Intensity speedup](assets/benchmarks/intensity_speedup_factor.png)](assets/benchmarks/intensity_speedup_factor.png) |

**Pictologics-only intensity families (IVH + spatial/local intensity):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0340 s                | 0.84 MB                |
|     25 | FBS 25.0         | 0.0280 s                | 0.81 MB                |
|     25 | FBS 50.0         | 0.0297 s                | 0.81 MB                |
|     25 | FBN 16           | 0.0316 s                | 0.81 MB                |
|     25 | FBN 32           | 0.0309 s                | 0.81 MB                |
|     25 | FBN 64           | 0.0305 s                | 0.81 MB                |
|     50 | FBS 10.0         | 1.2988 s                | 6.33 MB                |
|     50 | FBS 25.0         | 1.3367 s                | 6.33 MB                |
|     50 | FBS 50.0         | 1.3608 s                | 6.33 MB                |
|     50 | FBN 16           | 1.3218 s                | 6.33 MB                |
|     50 | FBN 32           | 1.3635 s                | 6.33 MB                |
|     50 | FBN 64           | 1.2953 s                | 6.33 MB                |
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
|     25 | FBS 10.0         | 0.0043 s                | 1.17 MB                |
|     25 | FBS 25.0         | 0.0041 s                | 1.17 MB                |
|     25 | FBS 50.0         | 0.0040 s                | 1.17 MB                |
|     25 | FBN 16           | 0.0039 s                | 1.17 MB                |
|     25 | FBN 32           | 0.0039 s                | 1.17 MB                |
|     25 | FBN 64           | 0.0041 s                | 1.17 MB                |
|     50 | FBS 10.0         | 0.0109 s                | 5.43 MB                |
|     50 | FBS 25.0         | 0.0107 s                | 5.43 MB                |
|     50 | FBS 50.0         | 0.0104 s                | 5.43 MB                |
|     50 | FBN 16           | 0.0110 s                | 5.43 MB                |
|     50 | FBN 32           | 0.0109 s                | 5.43 MB                |
|     50 | FBN 64           | 0.0109 s                | 5.43 MB                |
|     75 | FBS 10.0         | 0.0170 s                | 8.84 MB                |
|     75 | FBS 25.0         | 0.0170 s                | 8.84 MB                |
|     75 | FBS 50.0         | 0.0170 s                | 8.84 MB                |
|     75 | FBN 16           | 0.0168 s                | 8.84 MB                |
|     75 | FBN 32           | 0.0170 s                | 8.84 MB                |
|     75 | FBN 64           | 0.0176 s                | 8.84 MB                |
|    100 | FBS 10.0         | 0.0333 s                | 20.46 MB               |
|    100 | FBS 25.0         | 0.0335 s                | 20.46 MB               |
|    100 | FBS 50.0         | 0.0348 s                | 20.46 MB               |
|    100 | FBN 16           | 0.0349 s                | 20.46 MB               |
|    100 | FBN 32           | 0.0342 s                | 20.46 MB               |
|    100 | FBN 64           | 0.0350 s                | 20.46 MB               |


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
|     50 | FBS 50.0         | 0.0002 s                | 0.13 MB                |
|     50 | FBN 16           | 0.0003 s                | 0.13 MB                |
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
|    100 | FBS 50.0         | 0.0004 s                | 0.14 MB                |
|    100 | FBN 16           | 0.0004 s                | 0.14 MB                |
|    100 | FBN 32           | 0.0004 s                | 0.15 MB                |
|    100 | FBN 64           | 0.0004 s                | 0.17 MB                |


### Filters

| Execution Time (Log-Log) | Speedup |
|:---:|:---:|
| [![Filters time](assets/benchmarks/filters_execution_time_log.png)](assets/benchmarks/filters_execution_time_log.png) | [![Filters speedup](assets/benchmarks/filters_speedup_factor.png)](assets/benchmarks/filters_speedup_factor.png) |

**Pictologics-only filters (Gabor, Laws, Simoncelli, Riesz, Mean):**

|   Size | Discretization   | Pictologics-only Time   | Pictologics-only Mem   |
|-------:|:-----------------|:------------------------|:-----------------------|
|     25 | FBS 10.0         | 0.0090 s                | 1.70 MB                |
|     25 | FBS 25.0         | 0.0087 s                | 1.70 MB                |
|     25 | FBS 50.0         | 0.0089 s                | 1.70 MB                |
|     25 | FBN 16           | 0.0086 s                | 1.70 MB                |
|     25 | FBN 32           | 0.0090 s                | 1.70 MB                |
|     25 | FBN 64           | 0.0086 s                | 1.70 MB                |
|     50 | FBS 10.0         | 0.0290 s                | 13.51 MB               |
|     50 | FBS 25.0         | 0.0298 s                | 13.50 MB               |
|     50 | FBS 50.0         | 0.0296 s                | 13.50 MB               |
|     50 | FBN 16           | 0.0311 s                | 13.50 MB               |
|     50 | FBN 32           | 0.0320 s                | 13.50 MB               |
|     50 | FBN 64           | 0.0310 s                | 13.50 MB               |
|     75 | FBS 10.0         | 0.0590 s                | 45.50 MB               |
|     75 | FBS 25.0         | 0.0592 s                | 45.50 MB               |
|     75 | FBS 50.0         | 0.0578 s                | 45.50 MB               |
|     75 | FBN 16           | 0.0566 s                | 45.50 MB               |
|     75 | FBN 32           | 0.0673 s                | 45.50 MB               |
|     75 | FBN 64           | 0.0601 s                | 45.50 MB               |
|    100 | FBS 10.0         | 0.1114 s                | 107.80 MB              |
|    100 | FBS 25.0         | 0.1071 s                | 107.80 MB              |
|    100 | FBS 50.0         | 0.1161 s                | 107.80 MB              |
|    100 | FBN 16           | 0.1190 s                | 107.80 MB              |
|    100 | FBN 32           | 0.1107 s                | 107.80 MB              |
|    100 | FBN 64           | 0.1192 s                | 107.80 MB              |


### Detailed Parity Results

| Family     |   Size | Discretization   | Pictologics Time   | PyRadiomics Time   | Speedup   | Pictologics Mem   | PyRadiomics Mem   |
|:-----------|-------:|:-----------------|:-------------------|:-------------------|:----------|:------------------|:------------------|
| Filters    |     25 | FBN 16           | 0.0010 s           | 0.0043 s           | 4.27x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 32           | 0.0011 s           | 0.0044 s           | 4.07x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBN 64           | 0.0011 s           | 0.0043 s           | 4.10x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 10.0         | 0.0011 s           | 0.0045 s           | 4.21x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 25.0         | 0.0011 s           | 0.0046 s           | 4.25x     | 0.48 MB           | 0.94 MB           |
| Filters    |     25 | FBS 50.0         | 0.0010 s           | 0.0043 s           | 4.20x     | 0.48 MB           | 0.94 MB           |
| Filters    |     50 | FBN 16           | 0.0060 s           | 0.0097 s           | 1.62x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 32           | 0.0060 s           | 0.0095 s           | 1.58x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBN 64           | 0.0060 s           | 0.0097 s           | 1.62x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 10.0         | 0.0057 s           | 0.0097 s           | 1.68x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 25.0         | 0.0060 s           | 0.0099 s           | 1.66x     | 3.82 MB           | 6.68 MB           |
| Filters    |     50 | FBS 50.0         | 0.0058 s           | 0.0098 s           | 1.69x     | 3.82 MB           | 6.68 MB           |
| Filters    |     75 | FBN 16           | 0.0190 s           | 0.0263 s           | 1.39x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 32           | 0.0200 s           | 0.0269 s           | 1.35x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBN 64           | 0.0192 s           | 0.0275 s           | 1.43x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 10.0         | 0.0185 s           | 0.0262 s           | 1.42x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 25.0         | 0.0187 s           | 0.0258 s           | 1.38x     | 12.88 MB          | 23.38 MB          |
| Filters    |     75 | FBS 50.0         | 0.0191 s           | 0.0254 s           | 1.33x     | 12.88 MB          | 23.38 MB          |
| Filters    |    100 | FBN 16           | 0.0470 s           | 0.0571 s           | 1.21x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 32           | 0.0445 s           | 0.0554 s           | 1.24x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBN 64           | 0.0445 s           | 0.0546 s           | 1.23x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 10.0         | 0.0439 s           | 0.0570 s           | 1.30x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 25.0         | 0.0439 s           | 0.0569 s           | 1.30x     | 30.52 MB          | 53.41 MB          |
| Filters    |    100 | FBS 50.0         | 0.0453 s           | 0.0580 s           | 1.28x     | 30.52 MB          | 53.41 MB          |
| Intensity  |     25 | FBN 16           | 0.0008 s           | 0.0203 s           | 25.64x    | 0.24 MB           | 0.72 MB           |
| Intensity  |     25 | FBN 32           | 0.0008 s           | 0.0133 s           | 17.07x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBN 64           | 0.0009 s           | 0.0131 s           | 15.31x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 10.0         | 0.0019 s           | 0.0140 s           | 7.29x     | 0.24 MB           | 0.75 MB           |
| Intensity  |     25 | FBS 25.0         | 0.0008 s           | 0.0130 s           | 16.44x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     25 | FBS 50.0         | 0.0008 s           | 0.0126 s           | 15.89x    | 0.24 MB           | 0.71 MB           |
| Intensity  |     50 | FBN 16           | 0.0029 s           | 0.0660 s           | 22.96x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 32           | 0.0029 s           | 0.0661 s           | 23.12x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBN 64           | 0.0030 s           | 0.0652 s           | 21.59x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 10.0         | 0.0031 s           | 0.0689 s           | 22.38x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 25.0         | 0.0029 s           | 0.0663 s           | 22.78x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     50 | FBS 50.0         | 0.0029 s           | 0.0667 s           | 23.21x    | 1.40 MB           | 4.61 MB           |
| Intensity  |     75 | FBN 16           | 0.0103 s           | 0.2515 s           | 24.48x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 32           | 0.0117 s           | 0.2713 s           | 23.09x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBN 64           | 0.0112 s           | 0.2625 s           | 23.35x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 10.0         | 0.0117 s           | 0.2629 s           | 22.41x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 25.0         | 0.0112 s           | 0.2573 s           | 22.88x    | 5.81 MB           | 17.95 MB          |
| Intensity  |     75 | FBS 50.0         | 0.0110 s           | 0.2535 s           | 23.09x    | 5.81 MB           | 17.95 MB          |
| Intensity  |    100 | FBN 16           | 0.0248 s           | 0.5414 s           | 21.81x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 32           | 0.0238 s           | 0.5456 s           | 22.97x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBN 64           | 0.0236 s           | 0.5485 s           | 23.28x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 10.0         | 0.0241 s           | 0.5450 s           | 22.58x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 25.0         | 0.0228 s           | 0.5280 s           | 23.19x    | 12.16 MB          | 39.01 MB          |
| Intensity  |    100 | FBS 50.0         | 0.0234 s           | 0.5328 s           | 22.81x    | 12.16 MB          | 39.01 MB          |
| Morphology |     25 | FBN 16           | 0.0038 s           | 0.0539 s           | 14.22x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 32           | 0.0039 s           | 0.0541 s           | 13.74x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBN 64           | 0.0038 s           | 0.0539 s           | 14.11x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 10.0         | 0.0061 s           | 0.0547 s           | 8.97x     | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 25.0         | 0.0040 s           | 0.0542 s           | 13.39x    | 1.17 MB           | 1.18 MB           |
| Morphology |     25 | FBS 50.0         | 0.0039 s           | 0.0541 s           | 13.85x    | 1.17 MB           | 1.18 MB           |
| Morphology |     50 | FBN 16           | 0.0104 s           | 0.9377 s           | 90.08x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 32           | 0.0104 s           | 0.9366 s           | 90.08x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBN 64           | 0.0103 s           | 0.9458 s           | 91.68x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 10.0         | 0.0103 s           | 0.9384 s           | 90.90x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 25.0         | 0.0103 s           | 0.9377 s           | 90.65x    | 5.43 MB           | 8.68 MB           |
| Morphology |     50 | FBS 50.0         | 0.0107 s           | 0.9370 s           | 87.68x    | 5.43 MB           | 8.68 MB           |
| Morphology |     75 | FBN 16           | 0.0170 s           | 1.7065 s           | 100.68x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 32           | 0.0165 s           | 1.7326 s           | 104.81x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBN 64           | 0.0168 s           | 1.7458 s           | 103.87x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 10.0         | 0.0167 s           | 1.7611 s           | 105.37x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 25.0         | 0.0169 s           | 1.7043 s           | 101.14x   | 8.84 MB           | 36.60 MB          |
| Morphology |     75 | FBS 50.0         | 0.0168 s           | 1.6992 s           | 101.40x   | 8.84 MB           | 36.60 MB          |
| Morphology |    100 | FBN 16           | 0.0361 s           | 9.0401 s           | 250.39x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 32           | 0.0343 s           | 8.8616 s           | 258.03x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBN 64           | 0.0339 s           | 8.7991 s           | 259.87x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 10.0         | 0.0334 s           | 8.7507 s           | 261.82x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 25.0         | 0.0323 s           | 8.6826 s           | 269.05x   | 20.46 MB          | 77.49 MB          |
| Morphology |    100 | FBS 50.0         | 0.0344 s           | 9.1090 s           | 265.13x   | 20.46 MB          | 77.49 MB          |
| Texture    |     25 | FBN 16           | 0.0048 s           | 0.0478 s           | 9.91x     | 2.13 MB           | 0.69 MB           |
| Texture    |     25 | FBN 32           | 0.0051 s           | 0.0512 s           | 10.04x    | 2.07 MB           | 0.75 MB           |
| Texture    |     25 | FBN 64           | 0.0057 s           | 0.0655 s           | 11.54x    | 4.53 MB           | 1.95 MB           |
| Texture    |     25 | FBS 10.0         | 0.0072 s           | 0.0921 s           | 12.80x    | 10.10 MB          | 3.77 MB           |
| Texture    |     25 | FBS 25.0         | 0.0052 s           | 0.0540 s           | 10.30x    | 2.16 MB           | 0.91 MB           |
| Texture    |     25 | FBS 50.0         | 0.0051 s           | 0.0488 s           | 9.63x     | 2.13 MB           | 0.71 MB           |
| Texture    |     50 | FBN 16           | 0.0243 s           | 0.2995 s           | 12.31x    | 19.85 MB          | 6.29 MB           |
| Texture    |     50 | FBN 32           | 0.0246 s           | 0.3060 s           | 12.47x    | 19.95 MB          | 6.29 MB           |
| Texture    |     50 | FBN 64           | 0.0191 s           | 0.3193 s           | 16.70x    | 9.83 MB           | 4.82 MB           |
| Texture    |     50 | FBS 10.0         | 0.0177 s           | 0.3543 s           | 20.05x    | 11.96 MB          | 5.41 MB           |
| Texture    |     50 | FBS 25.0         | 0.0254 s           | 0.3098 s           | 12.18x    | 19.30 MB          | 6.09 MB           |
| Texture    |     50 | FBS 50.0         | 0.0247 s           | 0.3007 s           | 12.17x    | 21.08 MB          | 6.60 MB           |
| Texture    |     75 | FBN 16           | 0.0784 s           | 1.2263 s           | 15.64x    | 84.43 MB          | 25.62 MB          |
| Texture    |     75 | FBN 32           | 0.0734 s           | 1.2323 s           | 16.78x    | 86.59 MB          | 26.23 MB          |
| Texture    |     75 | FBN 64           | 0.0770 s           | 1.2478 s           | 16.20x    | 63.72 MB          | 20.14 MB          |
| Texture    |     75 | FBS 10.0         | 0.0498 s           | 1.2860 s           | 25.84x    | 13.36 MB          | 17.08 MB          |
| Texture    |     75 | FBS 25.0         | 0.0756 s           | 1.2271 s           | 16.24x    | 89.44 MB          | 25.80 MB          |
| Texture    |     75 | FBS 50.0         | 0.0721 s           | 1.2117 s           | 16.80x    | 88.88 MB          | 25.65 MB          |
| Texture    |    100 | FBN 16           | 0.1986 s           | 2.5783 s           | 12.98x    | 213.37 MB         | 64.12 MB          |
| Texture    |    100 | FBN 32           | 0.2078 s           | 2.5764 s           | 12.40x    | 224.11 MB         | 66.17 MB          |
| Texture    |    100 | FBN 64           | 0.2104 s           | 2.5853 s           | 12.29x    | 204.71 MB         | 60.72 MB          |
| Texture    |    100 | FBS 10.0         | 0.1030 s           | 2.5900 s           | 25.15x    | 22.17 MB          | 37.15 MB          |
| Texture    |    100 | FBS 25.0         | 0.1980 s           | 2.6275 s           | 13.27x    | 230.74 MB         | 68.10 MB          |
| Texture    |    100 | FBS 50.0         | 0.2021 s           | 2.6235 s           | 12.98x    | 229.77 MB         | 68.48 MB          |

