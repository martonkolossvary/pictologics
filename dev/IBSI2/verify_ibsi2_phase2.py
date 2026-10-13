#!/usr/bin/env python3
"""
IBSI 2 Phase 2 Compliance Verification Script.

This script validates feature extraction from filtered response maps against
IBSI 2 Phase 2 reference values using the pictologics RadiomicsPipeline.

Uses the pipeline's `filter` step for all image filtering.

Reference: IBSI 2 Reference Manual, Section 9 (Phase 2)

Usage:
    python verify_ibsi2_phase2.py [--output docs/ibsi2_phase2_compliance.md]
"""

import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pictologics import RadiomicsPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev.compliance_utils import FEATURE_MAP, WORKERS, get_rounding_decimals  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

# IBSI 2 Phase 2 Config B: Preprocessing steps (common to all filters)
IBSI2_PREPROCESS_STEPS = [
    {
        "step": "resample",
        "params": {
            "new_spacing": (1.0, 1.0, 1.0),
            "interpolation": "cubic",  # tricubic spline
            "mask_interpolation": "linear",  # trilinear for mask
            "mask_threshold": 0.5,
        },
    },
    {"step": "round_intensities", "params": {}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},
]

# Feature extraction step (intensity only for Phase 2)
EXTRACT_INTENSITY_STEP = {
    "step": "extract_features",
    "params": {"families": ["intensity"]},
}

# Feature name mapping: pictologics name -> IBSI tag


# =============================================================================
# Filter Definitions (as pipeline steps)
# =============================================================================


def get_filter_configs() -> dict[str, dict[str, Any]]:
    """
    Get all 3D filter configurations for IBSI 2 Phase 2 Config B.

    Returns:
        Dict mapping filter_id to config dict with 'name', 'step', and 'notes'.
    """
    return {
        # 1.B: No filter (baseline)
        "1.B": {
            "name": "None",
            "step": None,  # No filter step
            "notes": "Baseline (no filter)",
        },
        # 2.B: Mean filter, support M=5
        "2.B": {
            "name": "Mean",
            "step": {
                "step": "filter",
                "params": {"type": "mean", "support": 5},
            },
            "notes": "3D, support=5",
        },
        # 3.B: LoG, σ=1.5mm, truncate=4σ
        "3.B": {
            "name": "LoG",
            "step": {
                "step": "filter",
                "params": {"type": "log", "sigma_mm": 1.5, "truncate": 4.0},
            },
            "notes": "σ=1.5mm, truncate=4σ",
        },
        # 4.B: Laws L5E5E5, rotation-invariant max, energy δ=7
        "4.B": {
            "name": "Laws",
            "step": {
                "step": "filter",
                "params": {
                    "type": "laws",
                    "kernel": "L5E5E5",
                    "rotation_invariant": True,
                    "pooling": "max",
                    "compute_energy": True,
                    "energy_distance": 7,
                },
            },
            "notes": "L5E5E5, rot-inv max, energy δ=7",
        },
        # 5.B: Gabor, rotation-invariant average, average over planes
        "5.B": {
            "name": "Gabor",
            "step": {
                "step": "filter",
                "params": {
                    "type": "gabor",
                    "sigma_mm": 5.0,
                    "lambda_mm": 2.0,
                    "gamma": 1.5,
                    "rotation_invariant": True,
                    "delta_theta": np.pi / 8,
                    "pooling": "average",
                    "average_over_planes": True,
                },
            },
            "notes": "σ=5mm, λ=2mm, γ=1.5, rot-inv avg",
        },
        # 6.B: Daubechies 3, LLH level 1, rotation-invariant average
        "6.B": {
            "name": "Daubechies 3",
            "step": {
                "step": "filter",
                "params": {
                    "type": "wavelet",
                    "wavelet": "db3",
                    "level": 1,
                    "decomposition": "LLH",
                    "rotation_invariant": True,
                    "pooling": "average",
                },
            },
            "notes": "LLH level 1, rot-inv avg",
        },
        # 7.B: Daubechies 3, HHH level 2, rotation-invariant average
        "7.B": {
            "name": "Daubechies 3",
            "step": {
                "step": "filter",
                "params": {
                    "type": "wavelet",
                    "wavelet": "db3",
                    "level": 2,
                    "decomposition": "HHH",
                    "rotation_invariant": True,
                    "pooling": "average",
                },
            },
            "notes": "HHH level 2, rot-inv avg",
        },
        # 8.B: Simoncelli B map level 1
        "8.B": {
            "name": "Simoncelli",
            "step": {
                "step": "filter",
                "params": {"type": "simoncelli", "level": 1},
            },
            "notes": "B map level 1",
        },
        # 9.B: Simoncelli B map level 2
        "9.B": {
            "name": "Simoncelli",
            "step": {
                "step": "filter",
                "params": {"type": "simoncelli", "level": 2},
            },
            "notes": "B map level 2",
        },
    }


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FeatureResult:
    """Result for a single feature comparison."""

    feature_name: str
    ibsi_tag: str
    computed: float
    reference: float
    tolerance: float
    passed: bool
    error: float
    decimals: int = 0
    has_reference: bool = True

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureResult":
        return cls(**data)


@dataclass
class TestResult:
    """Result of a single filter test configuration."""

    filter_id: str
    filter_name: str
    passed_count: int
    failed_count: int
    total_count: int
    time_seconds: float = 0.0
    features: list[FeatureResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TestResult":
        # Handle nested FeatureResult objects
        data["features"] = [
            FeatureResult.from_dict(f) for f in data.get("features", [])
        ]
        return cls(**data)


# =============================================================================
# Main Validator Class
# =============================================================================


def run_single_test(
    filter_id: str,
    filter_config: dict,
    image_path: str,
    mask_path: str,
    references: dict,
) -> TestResult:
    """Run a single filter test using the pipeline."""
    if filter_id not in references:
        return TestResult(
            filter_id=filter_id,
            filter_name=filter_config["name"],
            passed_count=0,
            failed_count=0,
            total_count=0,
            notes="SKIPPED: No reference values",
        )

    # Build pipeline steps
    steps = list(IBSI2_PREPROCESS_STEPS)
    if filter_config["step"] is not None:
        steps.append(filter_config["step"])
    steps.append(EXTRACT_INTENSITY_STEP)

    # Create and run pipeline
    pipeline = RadiomicsPipeline()
    config_name = f"ibsi2_{filter_id}"
    pipeline.add_config(config_name, steps)

    start_time = time.perf_counter()
    try:
        results = pipeline.run(image_path, mask_path, config_names=[config_name])
        series = results[config_name]
    except Exception as e:
        return TestResult(
            filter_id=filter_id,
            filter_name=filter_config["name"],
            passed_count=0,
            failed_count=0,
            total_count=0,
            notes=f"ERROR: {str(e)}",
        )
    elapsed = time.perf_counter() - start_time

    # Compare with references
    feature_results = []
    passed_count = 0
    failed_count = 0

    filter_refs = references[filter_id]

    for picto_name, ibsi_tag in FEATURE_MAP.items():
        if ibsi_tag not in filter_refs:
            continue

        ref_data = filter_refs[ibsi_tag]
        computed_val = series.get(picto_name, np.nan)

        if not ref_data["has_reference"]:
            # IBSI 2 publishes no standardized consensus value for this
            # feature/filter combination (blank consensus_value/tolerance
            # in the reference CSV). Report as coverage-only: never
            # counted as a numerical pass or fail.
            feature_results.append(
                FeatureResult(
                    feature_name=ref_data["name"],
                    ibsi_tag=ibsi_tag,
                    computed=float(computed_val),
                    reference=float("nan"),
                    tolerance=float("nan"),
                    passed=False,
                    error=float("nan"),
                    has_reference=False,
                )
            )
            continue

        ref_val = ref_data["value"]
        ref_val_raw = ref_data["value_raw"]
        tol = ref_data["tolerance"]

        # Round computed value to same decimals as reference
        decimals = get_rounding_decimals(ref_val_raw)
        computed_rounded = round(computed_val, decimals)

        error = abs(computed_rounded - ref_val)
        passed = error <= tol

        if passed:
            passed_count += 1
        else:
            failed_count += 1

        feature_results.append(
            FeatureResult(
                feature_name=ref_data["name"],
                ibsi_tag=ibsi_tag,
                computed=float(computed_val),
                reference=float(ref_val),
                tolerance=float(tol),
                passed=bool(passed),
                error=float(error),
                decimals=decimals,
            )
        )

    return TestResult(
        filter_id=filter_id,
        filter_name=filter_config["name"],
        passed_count=passed_count,
        failed_count=failed_count,
        total_count=passed_count + failed_count,
        time_seconds=elapsed,
        features=feature_results,
        notes=filter_config["notes"],
    )


# =============================================================================
# Main Validator Class
# =============================================================================


class IBSIPhase2Validator:
    """
    Validator for IBSI 2 Phase 2 feature benchmarks.

    Uses pictologics RadiomicsPipeline for all preprocessing, filtering,
    and feature extraction.

    Example:
        >>> validator = IBSIPhase2Validator("dev/IBSI2")
        >>> results = validator.run_all_tests()
        >>> validator.print_summary()
    """

    def __init__(self, data_dir: Path | str, n_workers: int = 4):
        """
        Initialize the validator.

        Args:
            data_dir: Path to IBSI2 data directory containing:
                - data/ct_phantom/image.nii.gz
                - data/ct_phantom/mask.nii.gz
                - references/feature_values/reference_values.csv
            n_workers: Number of parallel workers (default: 4)
        """
        self.data_dir = Path(data_dir)
        self.ct_dir = self.data_dir / "data" / "ct_phantom"
        self.refs_path = (
            self.data_dir / "references" / "feature_values" / "reference_values.csv"
        )
        self.n_workers = n_workers

        self._image_path: Optional[str] = None
        self._mask_path: Optional[str] = None
        self._references: dict = {}
        self._results: list[TestResult] = []

    def _validate_paths(self) -> tuple[str, str]:
        """Validate and return image/mask paths."""
        if self._image_path is None:
            image_path = self.ct_dir / "image.nii.gz"
            mask_path = self.ct_dir / "mask.nii.gz"

            if not image_path.exists():
                raise FileNotFoundError(f"Lung CT image not found: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"GTV mask not found: {mask_path}")

            self._image_path = str(image_path)
            self._mask_path = str(mask_path)

        return self._image_path, self._mask_path

    def load_references(self) -> dict:
        """Load reference values from CSV file."""
        if not self._references:
            if not self.refs_path.exists():
                raise FileNotFoundError(f"Reference values not found: {self.refs_path}")

            with open(self.refs_path, "r") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    filter_id = row["filter_id"]
                    if not filter_id.endswith(".B"):
                        continue

                    if filter_id not in self._references:
                        self._references[filter_id] = {}

                    # Some rows (e.g. 8.B/stat_qcod) have no standardized
                    # consensus_value/tolerance published by IBSI. Keep the
                    # row (marked has_reference=False) instead of dropping
                    # it, so it can be reported as coverage-only rather
                    # than silently omitted or scored as a pass/fail.
                    has_reference = bool(row["consensus_value"]) and bool(
                        row["tolerance"]
                    )

                    self._references[filter_id][row["feature_tag"]] = {
                        "name": row["feature"],
                        "value": float(row["consensus_value"])
                        if has_reference
                        else None,
                        "value_raw": row["consensus_value"],
                        "tolerance": float(row["tolerance"])
                        if has_reference
                        else None,
                        "has_reference": has_reference,
                    }

        return self._references

    def get_provenance(self) -> dict:
        """Get provenance for the pictologics version and reference data used."""
        import pictologics

        try:
            reference_file = str(self.refs_path.relative_to(Path.cwd()))
        except ValueError:
            reference_file = str(self.refs_path)

        provenance = {
            "pictologics_version": pictologics.__version__,
            "reference_file": reference_file,
            "reference_source": "https://github.com/theibsi/ibsi_2_reference_data",
        }
        if self.refs_path.exists():
            provenance["reference_file_sha256"] = hashlib.sha256(
                self.refs_path.read_bytes()
            ).hexdigest()[:12]

        return provenance

    def run_all_tests(self, verbose: bool = True) -> list[TestResult]:
        """Run all Phase 2 tests."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        filter_configs = get_filter_configs()
        self._results = []

        # Prepare inputs
        references = self.load_references()
        image_path, mask_path = self._validate_paths()

        if verbose:
            print(
                f"Testing {len(filter_configs)} filters on {self.n_workers} workers..."
            )

        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=WORKERS) as executor:
            future_to_id = {
                executor.submit(
                    run_single_test,
                    filter_id,
                    config,
                    image_path,
                    mask_path,
                    references,
                ): filter_id
                for filter_id, config in filter_configs.items()
            }

            iterator = as_completed(future_to_id)
            if not verbose:
                iterator = tqdm(
                    iterator,
                    total=len(future_to_id),
                    desc="Processing Filters",
                    unit="test",
                )

            # Collect results in order of completion (or sort later)
            results_map = {}

            for future in iterator:
                filter_id = future_to_id[future]
                try:
                    result = future.result()
                    results_map[filter_id] = result

                    if verbose:
                        if "SKIPPED" in result.notes or "ERROR" in result.notes:
                            print(f"  {filter_id}: {result.notes}")
                        else:
                            status = "PASS" if result.failed_count == 0 else "FAIL"
                            print(
                                f"  ✓ {filter_id} {status} ({result.passed_count}/{result.total_count}, {result.time_seconds:.1f}s)"
                            )

                except Exception as e:
                    print(f"  ✗ {filter_id}: Unexpected error: {e}")

        # Store in original order
        for filter_id in filter_configs:
            if filter_id in results_map:
                self._results.append(results_map[filter_id])

        return self._results

    def values(self) -> dict[str, dict[str, float]]:
        """Our feature values per filter test, for the gate's value check."""
        return {
            r.filter_id: {f.ibsi_tag: f.computed for f in r.features}
            for r in sorted(self._results, key=lambda r: r.filter_id)
        }

    def get_summary(self) -> dict:
        """Get structured summary dictionary."""
        tests = {}
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        for r in self._results:
            if "SKIPPED" in r.notes:
                total_skipped += 1
                status = "SKIP"
            elif r.failed_count == 0 and r.total_count > 0:
                total_passed += 1
                status = "PASS"
            else:
                total_failed += 1
                status = "FAIL"

            tests[r.filter_id] = {
                "filter": r.filter_name,
                "status": status,
                "passed": r.passed_count,
                "failed": r.failed_count,
                "total": r.total_count,
                "time_ms": round(r.time_seconds * 1000, 1),
                "notes": r.notes,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "total": len(self._results),
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "provenance": self.get_provenance(),
            "tests": tests,
        }

    def save_cache(self, output_path: Path | str):
        """Save results to JSON cache."""
        data = {
            "results": [r.to_dict() for r in self._results],
            "timestamp": time.time(),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_cache(self, input_path: Path | str) -> bool:
        """Load results from JSON cache."""
        path = Path(input_path)
        if not path.exists():
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self._results = [TestResult.from_dict(d) for d in data.get("results", [])]
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def print_summary(self) -> None:
        """Print test summary to console."""
        summary = self.get_summary()
        print()
        print("-" * 60)
        print(
            f"Total: {summary['total']} | Passed: {summary['passed']} | "
            f"Failed: {summary['failed']} | Skipped: {summary['skipped']}"
        )
        print("-" * 60)

    def generate_docs(self, output_path: Path | str) -> None:
        """Generate full IBSI 2 Phase 2 compliance documentation."""
        output_path = Path(output_path)

        lines = []
        # 1. Header & Overview
        lines.append("# IBSI 2 Phase 2 Compliance: Radiomic Features")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(
            "IBSI 2 Phase 2 focuses on evaluating the **reproducibility of radiomic feature"
            "s extracted from filtered images**. In this phase, specific filters (Mean, LoG"
            ", Laws, Gabor, Wavelets) are applied to a digital phantom (Lung Cancer CT), a"
            "nd intensity features are extracted from the resulting response maps."
        )
        lines.append("")
        lines.append(
            "This page documents Pictologics' compliance with **IBSI 2 Phase 2 (Configuration B)**,"
            " which covers 3D volumetric processing."
        )
        lines.append("")

        # 2. Instructions
        lines.append("## How to Run the Benchmarks")
        lines.append("")
        lines.append("### 1. Download the Data")
        lines.append("")
        lines.append(
            "The IBSI 2 reference datasets are available on the [IBSI GitHub repository]"
            "(https://github.com/theibsi/data_sets)."
        )
        lines.append("")
        lines.append(
            "-   **CT Phantom**: Download the `PAT1_CT.nii.gz` and `PAT1_GTV.nii.gz` (or si"
            "milarly named `image.nii.gz` / `mask.nii.gz`) from the `ibsi_2_phase_2` (or "
            "`ibsi_2_validation`) folder."
        )
        lines.append("")
        lines.append(
            "Place these files in a local directory (e.g., `data/ibsi2/data/ct_phantom/`) t"
            "o run the benchmarks."
        )
        lines.append("")
        lines.append(
            "### 2. Run Configurations Programmatically using `RadiomicsPipeline`"
        )
        lines.append("")
        lines.append(
            "You can verify any IBSI 2 configuration programmatically using the `RadiomicsP"
            "ipeline`."
        )
        lines.append("")
        lines.append("```python")
        lines.append("from pictologics import RadiomicsPipeline")
        lines.append("")
        lines.append("# Path to your downloaded phantom")
        lines.append('image_path = "data/ibsi2/data/ct_phantom/image.nii.gz"')
        lines.append('mask_path = "data/ibsi2/data/ct_phantom/mask.nii.gz"')
        lines.append("")
        lines.append("pipeline = RadiomicsPipeline()")
        lines.append("")
        lines.append("# Define common IBSI 2 Preprocessing (Config B)")
        lines.append("preprocess_steps = [")
        lines.append(
            "    # 1. Resample to 1x1x1 mm using Tricubic Spline interpolation"
        )
        lines.append('    {"step": "resample", "params": {')
        lines.append('        "new_spacing": (1.0, 1.0, 1.0), ')
        lines.append('        "interpolation": "cubic", ')
        lines.append('        "mask_interpolation": "linear", ')
        lines.append('        "mask_threshold": 0.5')
        lines.append("    }},")
        lines.append("    # 2. Round intensities to nearest integer")
        lines.append('    {"step": "round_intensities", "params": {}},')
        lines.append("    # 3. Resegment range [-1000, 400] HU")
        lines.append(
            '    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},'
        )
        lines.append("]")
        lines.append("")
        lines.append("# Define a filter configuration (e.g., Test 3.B: LoG)")
        lines.append("log_config = preprocess_steps + [")
        lines.append('    {"step": "filter", "params": {')
        lines.append('        "type": "log",')
        lines.append('        "sigma_mm": 1.5,')
        lines.append('        "truncate": 4.0')
        lines.append("    }},")
        lines.append(
            '    {"step": "extract_features", "params": {"families": ["intensity"]}}'
        )
        lines.append("]")
        lines.append("")
        lines.append('pipeline.add_config("ibsi2_test_3b", log_config)')
        lines.append("")
        lines.append("# Run pipeline")
        lines.append(
            'results = pipeline.run(image_path, mask_path, config_names=["ibsi2_test_3b"])'
        )
        lines.append('print(results["ibsi2_test_3b"])')
        lines.append("```")
        lines.append("")

        # 3. Provenance
        provenance = self.get_provenance()
        lines.append("## Provenance")
        lines.append("")
        lines.append(
            f"-   **Pictologics version**: `{provenance['pictologics_version']}`"
        )
        lines.append(
            "-   **IBSI 2 reference manual**: version 9 (the revision bundled under "
            "`dev/IBSI2/documentation/`)"
        )
        lines.append(
            f"-   **Reference data**: `{provenance['reference_file']}` "
            f"(IBSI 2 reference feature values, [{provenance['reference_source']}]"
            f"({provenance['reference_source']}))"
        )
        if "reference_file_sha256" in provenance:
            lines.append(
                "-   **Reference data SHA-256** (first 12 hex chars): "
                f"`{provenance['reference_file_sha256']}`"
            )
        lines.append("")

        # 4. Results Overview
        active_results = [r for r in self._results if "SKIPPED" not in r.notes]
        passed_test_count = sum(1 for r in active_results if r.failed_count == 0)
        total_test_count = len(active_results)

        total_features_passed = sum(r.passed_count for r in active_results)
        total_features_checked = sum(r.total_count for r in active_results)
        total_no_reference = sum(
            1 for r in active_results for f in r.features if not f.has_reference
        )

        lines.append("## Phase 2 Results")
        lines.append("")
        lines.append(
            f"**Summary**: {passed_test_count}/{total_test_count} Tests Passed (3D Configuration)."
            f" Total Features: {total_features_passed}/{total_features_checked} passed."
        )
        if total_no_reference:
            lines.append("")
            lines.append(
                f"*Note*: {total_no_reference} feature/filter combination(s) have no "
                "IBSI-2 standardized consensus value published (blank `consensus_value`"
                "/`tolerance` in the reference dataset) and are excluded from the "
                "pass/fail counts above. They are reported as coverage-only rows in "
                "the tables below (see Test 8.B: `stat_qcod`) and are never counted as "
                "a numerical pass or fail."
            )
        lines.append("")

        # 5. Detailed Tables
        for result in self._results:
            if "SKIPPED" in result.notes:
                continue

            lines.append(f"### Test {result.filter_id}: {result.filter_name}")
            lines.append("")
            lines.append(f"**Configuration**: {result.notes}")
            lines.append("")
            lines.append("| Feature | Code | Calc | Ref | Tol | Status |")
            lines.append("|---|---|---|---|---|---|")

            for f in result.features:
                if not f.has_reference:
                    # No standardized IBSI-2 consensus value exists for
                    # this feature/filter combination: report coverage
                    # only, never as a numerical pass or fail.
                    calc_str = f"{f.computed:.3g}"
                    lines.append(
                        f"| {f.feature_name} | {f.ibsi_tag} | {calc_str} | N/A | N/A "
                        "| ⚪ NO REFERENCE (coverage only) |"
                    )
                    continue

                status = "✅ PASS" if f.passed else "❌ FAIL"

                # Check if scientific notation is needed/better matching reference style
                # Actually, simply using the decimals count for 'f' or 'e' as appropriate
                # If reference string was '1.23e+04', decimals is 2.
                # If we use 'g' or 'e', handling decimals is tricky.
                # Simplest approach: use general formatting but try to match precision if possible,
                # or just use standard formatting relative to the value size?
                # The prompt requested dynamic formatting based on reference.
                # Let's rely on standard formatting but with precision control.

                def format_val(val, decimals):
                    # Simple heuristic: if val matches 'g' compactness, use it, else 'f'
                    # But we want to match reference precision roughly.
                    return f"{val:.3g}"  # Fallback

                # Better approach:
                # If decimals was derived from '1.50', decimals=2. -> f"{val:.2f}"
                # If derived from '1.5e3', decimals=1 -> f"{val:.1e}" (maybe)

                # Implementation map from manual file:
                # "2.07e+11" -> seems like .2e
                # "0.0384" -> .4f or .3g

                # Let's use a helper to format based on the reference value structure if possible
                # But here we only have 'decimals'.
                # Let's use a generic formatter that respects significant figures (3 sig figs was mentioned)
                # "comprehensive results tables with 3 significant digits" -> User prompt

                # Let's use 3 significant digits for everything unless exact matching requires more?
                # The manual tables show e.g. "6.52e+04" (3 sig figs), "1.15" (3 sig figs), "-715" (3 sig figs)
                # So I will enforce 3 significant digits.

                calc_str = f"{f.computed:.3g}"
                ref_str = f"{f.reference:.3g}"
                tol_str = f"{f.tolerance:.3g}"

                lines.append(
                    f"| {f.feature_name} | {f.ibsi_tag} | {calc_str} | {ref_str} | {tol_str} | {status} |"
                )

            lines.append("")

        # 6. Skipped Tests Section
        lines.append("## Known Deviations")
        lines.append("")
        lines.append("### 2D Configuration (Config A) Not Implemented")
        lines.append("")
        lines.append(
            "The following tests check filter performance in 2D mode (Config A). These are "
            "**skipped** because Pictologics calculates radiomic features over a 3D "
            "volumetric ROI and does not provide a slice-wise 2D radiomics workflow. "
            "(Config B is the volumetric pathway reported above; the Gabor filter remains "
            "an explicit filter-level exception that uses 2D plane kernels.)"
        )
        lines.append("")
        lines.append("-   **Tests 1.A - 9.A**: 2D versions of the above tests.")
        lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Generated compliance report at {output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for Phase 2 validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="IBSI 2 Phase 2 Compliance Verification (3D Only)"
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("IBSI 2 Phase 2 Compliance Verification")
    print("=" * 60)
    print()

    validator = IBSIPhase2Validator(args.data_dir)

    print("Running Phase 2 tests...")
    validator.run_all_tests(verbose=not args.quiet)
    validator.print_summary()

    if args.output:
        validator.generate_docs(args.output)
        print(f"\nDocumentation saved to: {args.output}")

    if args.baseline:
        summary = validator.get_summary()
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Baseline saved to: {args.baseline}")

    summary = validator.get_summary()
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
