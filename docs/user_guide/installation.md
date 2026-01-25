# Installation

## Prerequisites

- Python 3.12+
- pip

## Installation via Pip

```bash
pip install pictologics
```

## Installation from GitHub

If you want the latest development version (or you want to install before the
next PyPI release), you can install directly from the GitHub repository.

### Latest from `main`

```bash
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@main"
```

### Pinned to a tag or commit

```bash
# Example: install from a tag
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@v0.1.0"

# Example: install from a commit SHA
pip install "pictologics @ git+https://github.com/martonkolossvary/pictologics.git@<commit_sha>"
```

### Editable install (development)

Use this if you plan to modify the code.

```bash
git clone https://github.com/martonkolossvary/pictologics.git
cd pictologics
pip install -e .
```

## Eager Compilation (Warmup)

Pictologics uses Numba for Just-In-Time (JIT) compilation to accelerate feature extraction. To ensure fast runtime performance, Pictologics performs an **automatic warmup mechanism** during import. This compiles the core functions immediately when `import pictologics` is executed.

!!! note
    This may cause the `import pictologics` statement to take a few seconds (typically 2-10s depending on your CPU) to complete. This is expected behavior and guarantees that subsequent function calls are executed at full speed without initial compilation lag.

### Disabling Warmup

If you need fast import times (e.g., for CLI tools checking versions or lightweight scripts) and are willing to pay the compilation cost at the first function call, you can disable this behavior by setting the environment variable:

```bash
export PICTOLOGICS_DISABLE_WARMUP=1
```
