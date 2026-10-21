`gabor_filter` returns a row-order (C-contiguous) array on every plane, as every other filter does. On the default axial plane it returned a strided view before. Only the axial result is copied.
