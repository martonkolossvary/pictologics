`load_image()` inflates a gzip NIfTI file in one pass into its stored values: 158 ms instead of 173 ms for a 512×512×200 CT, with the same array, bit for bit.
