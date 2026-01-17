# pictologics/filters/base.py
"""Base classes and utilities for IBSI 2 filter implementations."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import numpy as np


class BoundaryCondition(Enum):
    """
    IBSI 2 boundary conditions for image padding (GBYQ).

    Maps to scipy.ndimage mode parameter values.
    """

    ZERO = "constant"  # Zero padding (Z3VE)
    NEAREST = "nearest"  # Nearest value padding (SIJG)
    PERIODIC = "wrap"  # Periodic/wrap padding (Z7YO)
    MIRROR = "reflect"  # Mirror/symmetric padding (ZDTV)


@dataclass
class FilterResult:
    """Container for filter response maps and metadata."""

    response_map: np.ndarray
    filter_name: str
    filter_params: Dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the response map."""
        return self.response_map.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the response map."""
        return self.response_map.dtype


def ensure_float32(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is at least 32-bit floating point precision.

    Per IBSI 2: "The phantom data need to be converted from an integer
    data type to at least 32 bit floating point precision, prior to filtering."

    Args:
        image: Input image array

    Returns:
        Image as float32 (or higher precision if already float64)
    """
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32) if image.dtype == np.float16 else image
    return image.astype(np.float32)


def get_scipy_mode(boundary: BoundaryCondition) -> str:
    """Convert BoundaryCondition to scipy.ndimage mode string."""
    return boundary.value
