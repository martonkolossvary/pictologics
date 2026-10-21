The NRRD and MetaImage readers give writeable arrays. So the first load of each stored type no longer compiles a second, read-only numba signature (about 0.2 s with a cold cache).
