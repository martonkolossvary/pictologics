"""
Pictologics: IBSI-compliant radiomic feature extraction from medical images.
"""

# The number of threads is set before the imports that start numba.
# ruff: noqa: E402

__version__ = "0.7.0"

from .threads import _configure, get_num_threads, set_num_threads

_configure()  # numba reads its number of threads once, at its import

from .deduplication import (
    CURRENT_RULES_VERSION,
    RULES_REGISTRY,
    ConfigurationAnalyzer,
    DeduplicationPlan,
    DeduplicationRules,
    PreprocessingSignature,
    get_default_rules,
)
from .loader import (
    Image,
    create_full_mask,
    load_and_merge_images,
    load_image,
    save_image,
)
from .loaders import load_rtstruct, load_seg
from .pipeline import RadiomicsPipeline, SourceMode
from .results import format_results, save_results
from .warmup import warmup_jit

# Perform automatic JIT warmup on import
# This can be disabled by setting PICTOLOGICS_DISABLE_WARMUP=1
warmup_jit()

__all__ = [
    # Core
    "load_image",
    "save_image",
    "load_seg",
    "load_rtstruct",
    "Image",
    "create_full_mask",
    "load_and_merge_images",
    "RadiomicsPipeline",
    "SourceMode",
    "format_results",
    "save_results",
    # Threads
    "get_num_threads",
    "set_num_threads",
    # Deduplication
    "ConfigurationAnalyzer",
    "DeduplicationPlan",
    "DeduplicationRules",
    "PreprocessingSignature",
    "CURRENT_RULES_VERSION",
    "RULES_REGISTRY",
    "get_default_rules",
]
