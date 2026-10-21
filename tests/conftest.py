# tests/conftest.py
"""
Pytest configuration for the pictologics test suite.

This file is automatically loaded by pytest before any test modules.
Setting NUMBA_DISABLE_JIT=1 here ensures that Numba functions run as
regular Python code, allowing unittest.mock.patch to work correctly
on numpy functions called inside Numba-decorated functions.
"""

import os

# Disable Numba JIT compilation BEFORE any imports
# This must happen before any pictologics module is imported
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Suppress numpy reload warnings that can occur with Numba
import warnings

warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def scipy_fuses() -> bool:
    """Whether the scipy build of this machine fuses a multiply-add into one rounding, as
    the numba kernels of the package do (the arm64 wheels fuse, the x86-64 wheels do not).
    A test that asks for scipy's bits compares bytes where scipy fuses, and values within
    a few units in the last place elsewhere."""
    from scipy.ndimage import correlate1d

    # The pair sum 1 + 2^-30 times the weight 1 - 2^-30 is 1 - 2^-60: a fused step keeps
    # it, a rounded product loses it against the centre term -1
    weight = 1.0 - 2.0**-30
    line = np.array([1.0, 1.0, 2.0**-30])
    return bool(correlate1d(line, np.array([weight, -1.0, weight]), mode="constant")[1] != 0.0)
