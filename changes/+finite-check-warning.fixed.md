The NaN check of a run gives no numpy RuntimeWarning when the sum of the image values is too large for float64 or float32, or when the image has +inf and -inf. Results are unchanged.
