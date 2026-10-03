# Tips for Speed and Memory

Pictologics computes in parallel and only where it must. These tips help it do so.

## Threads

- Pictologics uses the numba thread count for all its parallel parts: the numba kernels, the FFT filters and the thread pools of the filters. One setting limits them all: the environment variable `NUMBA_NUM_THREADS`, or `numba.set_num_threads()` in your script.
- `run_batch(..., workers=4)` gives each worker process a quarter of the threads.
- See the [benchmark page](../benchmarks.md#threads) for the speed-up by threads.

## The First Run

- The first `import pictologics` compiles the numba code and keeps it in a cache. Later imports read the cache, so they take less time.
- Set `PICTOLOGICS_DISABLE_WARMUP=1` to import without the warm-up. The first call of a function then compiles its code.

## Work Only Near the ROI

- When no later step reads the image away from the ROI, the pipeline resamples, filters and discretises only the region around the ROI box. A small ROI in a large CT then costs much less time and memory.
- These later steps read the image away from the ROI, so an earlier resample or filter makes the whole grid: a `filter`, a `resample`, a `binarize_mask` that keeps the value 0, and a `normalise` step with region `"image"`. For example, in "resample, then filter", the resample makes the whole new grid, and the filter computes only the region around the ROI.

## Memory

- Before a resample, the pipeline checks that the new grid fits in the memory of the computer: a configuration needs at least 24 bytes for each voxel of the new grid. When it does not fit, the configuration stops with a `MemoryError` in its log entry.
- A resample of a whole CT to 0.5 mm makes a grid of hundreds of millions of voxels. With an ROI, the pipeline resamples only its region.
- The filters return float32 maps, half the memory of float64.
- Each `run_batch` worker holds one case at a time: the memory need grows with the number of workers.

## Many Configurations and ROIs

- Configurations that share their first steps share the states after these steps, and families that read the same image and masks share their features (deduplication).
- `run_rois` runs many ROIs of one label map with one image load: much faster than one `run()` for each ROI.
- In direct calls, `calculate_all_texture_features(..., families=["glcm"])` computes only the families that you need.

## Measure

- `pipeline.get_log()` gives the time of each configuration (`elapsed_seconds`).
- The [benchmark page](../benchmarks.md) gives the speed of each part on one computer.
