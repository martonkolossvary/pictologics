A gzip or bzip2 NRRD file with `byte skip: -1` raises a clear error: the NRRD format allows this value with raw encoding only. Before, the loader read the last byte and reported too few bytes.
