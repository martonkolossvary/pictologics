# Installation

## Requirements

- Python 3.12, 3.13 or 3.14.
- pip, or another Python package installer.

## Install from PyPI

```bash
pip install pictologics
```

pip also installs the packages that Pictologics needs:

| Package | For |
|:--|:--|
| NumPy, SciPy, Numba | The computations (Numba compiles the fast parts) |
| nibabel | NIfTI files |
| pydicom, python-gdcm, Pillow | DICOM files, also compressed ones (JPEG Lossless, JPEG-LS, JPEG 2000, baseline JPEG) |
| PyWavelets | The wavelet filters |
| pandas | The result tables |
| PyYAML | Configuration files |
| Matplotlib | The slice viewers (`visualize_slices`, `save_slices`) |
| tqdm | The progress bar of `run_batch` |

!!! note "12-bit lossy JPEG DICOM"
    python-gdcm reads all compressed DICOM types except 12-bit lossy JPEG (JPEG Extended). For these files, also install `pylibjpeg` and `pylibjpeg-libjpeg` (GPL-3.0, so Pictologics does not install it for you).

## Install from GitHub

The newest development version:

```bash
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@main"
```

A tag or a commit:

```bash
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@v0.6.0"
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@<commit_sha>"
```

## Development Install

The project uses [Poetry](https://python-poetry.org/) for its development tools (tests, type checks, docs):

```bash
git clone https://github.com/martonkolossvary/pictologics.git
cd pictologics
poetry install
```

`pip install -e .` gives an editable package, but without the test and docs tools.

## The First Import

Numba compiles the fast parts of Pictologics for your computer. So `import pictologics` compiles them on the first import (a warm-up), and keeps the compiled code in a disk cache:

- The first import can take about half a minute (27 s on an Apple M4 Pro).
- Later imports read the cache: about 2 s.
- A new version of Pictologics or of Numba compiles again on its first import.

Set `NUMBA_CACHE_DIR` to keep the cache in another folder, for example when the package folder is read-only.

To import without the warm-up, for example in a script that only prints a version:

```bash
export PICTOLOGICS_DISABLE_WARMUP=1
```

The first call of each function then compiles its code.

## Threads

Pictologics computes in parallel on all cores. Set the number of threads with the environment variable `NUMBA_NUM_THREADS`, or with `numba.set_num_threads()` in your script. One setting limits all parallel parts, also the filters. See [Tips for Speed and Memory](../tutorials/performance.md).

## Check the Installation

```python
import pictologics

print(pictologics.__version__)
```
