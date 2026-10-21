# Benchmarks

## Speed of Pictologics

This page gives the speed of Pictologics 0.8.0 on one computer. A later version of this page will compare more programs, data sets and computers.

!!! info "How we measured"
    - **Data**: synthetic and seeded. Correlated noise with blob-shaped ROIs (with holes) for the features; normal noise cubes of 1 mm voxels for the filters; a CT-like image of 512 × 512 × 200 voxels (0.7 × 0.7 × 1.25 mm: air, a body of soft tissue with noise, a lesion of 20 mm radius of 54,746 voxels) for the preprocessing, the pipeline and the loaders.
    - **Time**: the wall-clock time of one call, the median of 5 runs after one warm-up run (the warm-up compiles the numba code). The tables give the median; the list tables also give the fastest and the slowest run.
    - **Threads**: numba uses 10 threads (the default of Pictologics on this computer), except in the thread test.
    - **Features**: the feature functions on their own, without the pipeline. The texture rows use FBN 32, except in the grey level table.

### Computer and Software

- **Hardware**: Apple M4 Pro, 14 cores, 48 GB
- **OS**: macOS 27.0.1 (arm64)
- **Python**: 3.12.10
- **Core deps**: pictologics 0.8.0, numpy 2.5.3, scipy 1.17.0, numba 0.67.0, pandas 2.3.3, matplotlib 3.10.7
- **BLAS/LAPACK**: Apple Accelerate (from `numpy.show_config()`)
- **Numba threads**: 10

### Feature Families

ROI cubes: 25³ (9,861 ROI voxels), 50³ (101,638 ROI voxels), 75³ (315,151 ROI voxels), 100³ (830,970 ROI voxels), 150³ (2,311,384 ROI voxels).

[![Feature families](assets/benchmarks/features.png)](assets/benchmarks/features.png)

| Task | 25³ | 50³ | 75³ | 100³ | 150³ |
|:--|--:|--:|--:|--:|--:|
| Intensity (first order) | 0.0 ms | 0.3 ms | 0.7 ms | 1.9 ms | 4.0 ms |
| Intensity histogram | 0.0 ms | 0.1 ms | 0.2 ms | 0.2 ms | 0.3 ms |
| Intensity-volume histogram | 0.0 ms | 0.1 ms | 0.3 ms | 0.8 ms | 2.0 ms |
| Morphology | 0.3 ms | 1.0 ms | 1.6 ms | 2.2 ms | 4.5 ms |
| Local intensity (peaks) | 0.3 ms | 1.9 ms | 5.0 ms | 11.8 ms | 31.2 ms |
| Spatial intensity (Moran's I, Geary's C) | 1.2 ms | 6.1 ms | 12.8 ms | 30.5 ms | 92.0 ms |
| Texture (all 6 families) | 1.0 ms | 2.6 ms | 4.5 ms | 8.4 ms | 20.0 ms |

### Texture Families

Each family on its own (`calculate_all_texture_features(..., families=[...])`), FBN 32.

[![Texture families](assets/benchmarks/texture_families.png)](assets/benchmarks/texture_families.png)

| Task | 25³ | 50³ | 75³ | 100³ | 150³ |
|:--|--:|--:|--:|--:|--:|
| GLCM | 0.3 ms | 0.6 ms | 1.1 ms | 1.7 ms | 3.5 ms |
| GLRLM | 0.3 ms | 0.7 ms | 1.3 ms | 2.6 ms | 6.7 ms |
| GLSZM | 0.3 ms | 0.9 ms | 1.7 ms | 3.7 ms | 8.5 ms |
| GLDZM | 0.4 ms | 1.4 ms | 2.5 ms | 4.5 ms | 10.7 ms |
| NGTDM | 0.2 ms | 0.4 ms | 0.6 ms | 1.0 ms | 2.4 ms |
| NGLDM | 0.3 ms | 0.4 ms | 0.6 ms | 1.0 ms | 2.1 ms |

### Texture and Grey Levels

All six texture families on the 75³ cube (315,151 ROI voxels) with FBN discretisation.

[![Texture and grey levels](assets/benchmarks/texture_grey_levels.png)](assets/benchmarks/texture_grey_levels.png)

| Task | 8 levels | 16 levels | 32 levels | 64 levels | 128 levels | 256 levels |
|:--|--:|--:|--:|--:|--:|--:|
| Texture (all 6 families) | 4.7 ms | 4.8 ms | 4.5 ms | 4.7 ms | 5.3 ms | 6.4 ms |

### Filters

Image cubes of 1 mm voxels. Convolution filters use the mirror boundary; the FFT filters (Simoncelli, Riesz) use their periodic default.

[![Convolution filters](assets/benchmarks/filters_convolution.png)](assets/benchmarks/filters_convolution.png)

[![Wavelet and FFT filters](assets/benchmarks/filters_wavelet.png)](assets/benchmarks/filters_wavelet.png)

| Task | 64³ | 128³ | 192³ | 256³ |
|:--|--:|--:|--:|--:|
| Mean (support 5) | 0.6 ms | 1.8 ms | 4.7 ms | 10.2 ms |
| Gaussian (sigma 2 mm) | 0.8 ms | 2.7 ms | 8.2 ms | 18.4 ms |
| LoG (sigma 2 mm) | 1.9 ms | 6.5 ms | 16.9 ms | 37.5 ms |
| Laws L5E5E5 with energy | 1.2 ms | 3.4 ms | 9.1 ms | 20.1 ms |
| Laws L5E5E5, rotation invariant, energy | 2.3 ms | 8.1 ms | 22.7 ms | 52.2 ms |
| Gabor (axial, rotation invariant) | 2.6 ms | 8.7 ms | 25.0 ms | 52.4 ms |
| Gabor (3 planes, rotation invariant) | 8.2 ms | 25.6 ms | 68.9 ms | 143.0 ms |
| Wavelet db2 LHL, level 1 | 0.6 ms | 1.5 ms | 3.6 ms | 7.7 ms |
| Wavelet db2 LHL, level 1, rotation invariant | 3.7 ms | 20.1 ms | 49.1 ms | 111.2 ms |
| Simoncelli, level 1 | 1.1 ms | 4.2 ms | 12.6 ms | 30.2 ms |
| Riesz, order (1, 0, 0) | 0.9 ms | 4.1 ms | 12.4 ms | 32.5 ms |
| Riesz-LoG (sigma 2 mm) | 2.9 ms | 10.9 ms | 29.2 ms | 66.4 ms |
| Riesz-Simoncelli, level 1 | 1.0 ms | 4.4 ms | 14.1 ms | 33.6 ms |

### Preprocessing

The functions of `pictologics.preprocessing` on the CT-like image.

[![Preprocessing](assets/benchmarks/preprocessing.png)](assets/benchmarks/preprocessing.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| Resample to 1 mm (linear) | 11.4 ms | 11.4 ms | 11.4 ms |
| Resample the mask to 1 mm (nearest) | 2.7 ms | 2.6 ms | 2.8 ms |
| Discretise, FBN 32 (ROI range) | 8.9 ms | 8.8 ms | 9.3 ms |
| Discretise, FBS 25 HU | 6.1 ms | 6.0 ms | 6.1 ms |
| Resegment, -100 to 400 HU | 3.2 ms | 3.2 ms | 3.3 ms |
| Ring of 3 mm (grow_mask) | 9.6 ms | 9.5 ms | 9.8 ms |
| Keep the largest component | 3.5 ms | 3.3 ms | 3.8 ms |

### Pipeline

Whole pipeline runs on the CT-like image, from the image in memory to the feature Series. The standard configurations resample to 0.5 mm and compute the intensity, morphology, texture, histogram and IVH features; the pipeline resamples only the region around the ROI. The 20 ROIs of `run_rois` are spheres of 8 mm radius.

[![Pipeline runs](assets/benchmarks/pipeline.png)](assets/benchmarks/pipeline.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| run(): one standard configuration (standard_fbn_32) | 9.8 ms | 9.4 ms | 10.3 ms |
| run(): the 6 standard configurations | 34.3 ms | 33.8 ms | 34.5 ms |
| run(): resample to 1 mm, LoG, intensity | 17.0 ms | 16.7 ms | 17.5 ms |
| run_rois(): 20 ROIs, intensity and morphology at 1 mm | 18.3 ms | 18.1 ms | 18.7 ms |

### Threads

`run()` of `standard_fbn_32` on the CT-like image with 1 to 10 numba threads.

[![Thread scaling](assets/benchmarks/threads.png)](assets/benchmarks/threads.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| 1 thread | 23.2 ms | 23.1 ms | 24.0 ms |
| 2 threads | 15.8 ms | 15.2 ms | 16.2 ms |
| 4 threads | 11.4 ms | 11.0 ms | 11.5 ms |
| 8 threads | 9.5 ms | 9.3 ms | 9.9 ms |
| 10 threads | 9.7 ms | 9.1 ms | 10.2 ms |

### Loading

`load_image` of the CT-like image (int16 values), from a temporary folder on the internal disk.

[![Loading](assets/benchmarks/loading.png)](assets/benchmarks/loading.png)

| Task | Time (median) | Fastest | Slowest |
|:--|--:|--:|--:|
| NIfTI, gzip (.nii.gz) | 181.8 ms | 179.5 ms | 186.6 ms |
| NIfTI (.nii) | 22.3 ms | 21.5 ms | 32.0 ms |
| DICOM series (200 files) | 81.7 ms | 80.3 ms | 82.9 ms |
