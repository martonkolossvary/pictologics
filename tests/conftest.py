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
