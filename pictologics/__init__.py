"""
Pictologics: IBSI-compliant radiomic feature extraction from medical images.
"""

__version__ = "0.3.4"

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
)
from .loaders import load_seg
from .pipeline import RadiomicsPipeline, SourceMode
from .results import format_results, save_results
from .warmup import warmup_jit

# Perform automatic JIT warmup on import
# This can be disabled by setting PICTOLOGICS_DISABLE_WARMUP=1
warmup_jit()

__all__ = [
    # Core
    "load_image",
    "load_seg",
    "Image",
    "create_full_mask",
    "load_and_merge_images",
    "RadiomicsPipeline",
    "SourceMode",
    "format_results",
    "save_results",
    # Deduplication
    "ConfigurationAnalyzer",
    "DeduplicationPlan",
    "DeduplicationRules",
    "PreprocessingSignature",
    "CURRENT_RULES_VERSION",
    "RULES_REGISTRY",
    "get_default_rules",
]
