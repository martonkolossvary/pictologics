# pictologics/filters/__init__.py
"""
IBSI 2 Convolutional Filters for Radiomics.

This module provides standardized image filtering implementations
following the IBSI 2 reference manual specifications.

Filters:
    - Mean filter (S60F)
    - Laplacian of Gaussian (L6PA)
    - Laws kernels (JTXT)
    - Gabor filter (Q88H)
    - Separable wavelets (Haar, Daubechies, Coiflet)
    - Non-separable wavelets / Simoncelli (PRT7)
    - Riesz transform (AYRS)

Example:
    >>> from pictologics.filters import laplacian_of_gaussian
    >>> response = laplacian_of_gaussian(image, sigma_mm=5.0, spacing_mm=2.0)
"""

from .base import BoundaryCondition, FilterResult
from .gabor import gabor_filter
from .laws import LAWS_KERNELS, laws_filter
from .log import laplacian_of_gaussian
from .mean import mean_filter
from .riesz import riesz_log, riesz_simoncelli, riesz_transform
from .wavelets import simoncelli_wavelet, wavelet_transform

__all__ = [
    # Base types
    "BoundaryCondition",
    "FilterResult",
    # Filters
    "mean_filter",
    "laplacian_of_gaussian",
    "laws_filter",
    "LAWS_KERNELS",
    "gabor_filter",
    "wavelet_transform",
    "simoncelli_wavelet",
    "riesz_transform",
    "riesz_log",
    "riesz_simoncelli",
]
