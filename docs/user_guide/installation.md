# Installation

## Requirements

- Python 3.12, 3.13 or 3.14.
- On an Intel Mac, and with Intel (x86_64) Python on Apple silicon, for example 3D Slicer under Rosetta: Python 3.12 or 3.13. There pip installs numba 0.62, the last numba with builds for these computers, and NumPy below 2.4, as numba 0.62 needs.
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
| pydicom, python-gdcm, Pillow | DICOM files, also compressed ones (JPEG Lossless, JPEG-LS, JPEG 2000, baseline JPEG). Pillow also writes the files of `save_slices` |
| PyWavelets | The wavelet filters |
| pandas | The result tables |
| PyYAML | Configuration files |
| tqdm | The progress bar of `run_batch` |

The slice viewer `visualize_slices` also needs Matplotlib, the optional extra `viz`:

```bash
pip install "pictologics[viz]"
```

!!! note "12-bit lossy JPEG DICOM"
    python-gdcm reads all compressed DICOM types except 12-bit lossy JPEG (JPEG Extended). For these files, also install `pylibjpeg` and `pylibjpeg-libjpeg` (GPL-3.0, so Pictologics does not install it for you).

## Install from GitHub

The newest development version:

```bash
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@main"
```

A tag or a commit:

```bash
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@v0.7.0"
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

Pictologics computes in parallel on the fast cores of the computer. On a Mac with Apple silicon, these are the performance cores (for example, 10 of the 14 cores of an M4 Pro). On Windows, Pictologics uses all cores. For another number of threads, set an environment variable before the import:

```bash
export PICTOLOGICS_NUM_THREADS=4
```

You can also call `pictologics.set_num_threads(4)` in your script. One setting limits all parallel parts, also the filters. See [Tips for Speed and Memory](../tutorials/performance.md).

## Check the Installation

```python
import pictologics

print(pictologics.__version__)
```
