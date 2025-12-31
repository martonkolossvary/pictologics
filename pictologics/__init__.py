"""
Pictologics: IBSI-compliant radiomic feature extraction from medical images.
"""

__version__ = "0.2.0"

from .loader import (
    Image,
    create_full_mask,
    load_and_merge_images,
    load_image,
)
from .pipeline import RadiomicsPipeline
from .results import format_results, save_results
from .warmup import warmup_jit

# Perform automatic JIT warmup on import
# This can be disabled by setting PICTOLOGICS_DISABLE_WARMUP=1
warmup_jit()

__all__ = [
    "load_image",
    "Image",
    "create_full_mask",
    "load_and_merge_images",
    "RadiomicsPipeline",
    "format_results",
    "save_results",
]
