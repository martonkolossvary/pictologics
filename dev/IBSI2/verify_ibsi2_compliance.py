#!/usr/bin/env python3
"""
IBSI 2 Phase 1 Compliance Verification Script.

This script validates all filter implementations against IBSI 2 Phase 1 reference
response maps. Each test compares our computed response map against the reference
using the IBSI tolerance: max_diff ≤ 1% of intensity range.

Reference: IBSI 2 Reference Manual, Table 6.1
"""

import argparse
import hashlib
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import json

import numpy as np
import nibabel as nib

import pictologics

# Import filter functions
from pictologics.filters import (
    mean_filter,
    laplacian_of_gaussian,
    laws_filter,
    gabor_filter,
    wavelet_transform,
    simoncelli_wavelet,
    riesz_log,
    riesz_simoncelli,
    BoundaryCondition,
)

# Source of the reference response maps (`references/response_maps/*-ValidCRM.nii`)
# used for comparison. See dev/IBSI2/download_ibsi2_data.py.
IBSI2_REFERENCE_DATA_REPO_URL = "https://github.com/theibsi/ibsi_2_reference_data"


@dataclass
class TestResult:
    """Result of a single IBSI test."""

    test_id: str
    filter_name: str
    phantom: str
    passed: bool
    max_diff: float
    tolerance: float
    error_pct: float
    time_seconds: float = 0.0
    memory_mb: float = 0.0
    notes: str = ""
    # Explicit, first-class skip marker. A test can only be SKIPPED via this flag
    # (never inferred from free-text `notes`), and it is checked before any
    # numeric pass/fail verdict — see IBSIPhase1Validator.run_test().
    skipped: bool = False
    # Provenance: reference file used for comparison and a short content hash
    # (sha256, first 12 hex chars) of that file, populated only for tests that
    # actually loaded and compared against a reference (see run_test()).
    ref_file: str = ""
    ref_sha256: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TestResult":
        # Handle backward compatibility or type conversion if needed
        return cls(**data)


@dataclass
class TestConfig:
    """Configuration for an IBSI Phase 1 test."""

    test_id: str
    filter_name: str
    phantom: str
    ref_file: str
    compute_fn: Callable
    notes: str = ""
    # Explicit, first-class placeholder/skip marker (see TestResult.skipped).
    skipped: bool = False


class IBSIPhase1Validator:
    """Validator for IBSI 2 Phase 1 filter benchmarks."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.phantoms_dir = self.data_dir / "data" / "phantoms"
        self.refs_dir = self.data_dir / "references" / "response_maps"
        self._phantoms: dict = {}
        self._results: list[TestResult] = []
        # Response maps of the compared tests, for the gate's value check
        self._maps: dict[str, np.ndarray] = {}

    def load_phantom(self, name: str) -> np.ndarray:
        """Load and cache a phantom image."""
        if name not in self._phantoms:
            path = self.phantoms_dir / f"{name}.nii.gz"
            if not path.exists():
                raise FileNotFoundError(f"Phantom not found: {path}")
            img = nib.load(path)
            self._phantoms[name] = img.get_fdata().astype(np.float32)
        return self._phantoms[name]

    def load_reference(self, ref_file: str) -> Optional[np.ndarray]:
        """Load a reference response map if it exists."""
        path = self.refs_dir / ref_file
        if not path.exists():
            return None
        return nib.load(path).get_fdata().astype(np.float32)

    @staticmethod
    def _repo_path(path: Path) -> str:
        """A path relative to the repository (pages and results must not show local folders)."""
        root = Path(__file__).resolve().parents[2]
        path = Path(path).resolve()
        return str(path.relative_to(root)) if path.is_relative_to(root) else path.name

    @staticmethod
    def _hash_file(path: Path, length: int = 12) -> str:
        """Short-form sha256 hex digest of a file's bytes, for provenance."""
        return hashlib.sha256(path.read_bytes()).hexdigest()[:length]

    def run_test(self, config: TestConfig) -> TestResult:
        """Run a single test configuration."""
        # Skip is an explicit, first-class property checked BEFORE any numeric
        # verdict is computed, so a placeholder (identity) test can never be
        # reported as PASS/FAIL — see get_all_tests() and TestConfig.skipped.
        if config.skipped:
            return TestResult(
                test_id=config.test_id,
                filter_name=config.filter_name,
                phantom=config.phantom,
                passed=False,
                max_diff=0,
                tolerance=0,
                error_pct=0,
                time_seconds=0,
                skipped=True,
                notes=config.notes,
            )

        try:
            # Load phantom
            phantom = self.load_phantom(config.phantom)

            # Load reference
            ref = self.load_reference(config.ref_file)
            if ref is None:
                # No reference map: still run a real filter (not a placeholder) once, so
                # that a crash or a non-finite output fails the test.
                if not config.notes.startswith("SKIPPED"):
                    response = config.compute_fn(phantom)
                    if not np.all(np.isfinite(response)):
                        raise ValueError("non-finite values in the response map")
                return TestResult(
                    test_id=config.test_id,
                    filter_name=config.filter_name,
                    phantom=config.phantom,
                    passed=False,
                    max_diff=0,
                    tolerance=0,
                    error_pct=0,
                    time_seconds=0,
                    notes="MISSING_REF",
                )

            # Compute response with timing and memory profiling
            tracemalloc.start()
            start_time = time.perf_counter()

            response = config.compute_fn(phantom)

            elapsed = time.perf_counter() - start_time
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_mb = peak_memory / (1024 * 1024)

            # Validate (a response of another shape must not broadcast against the map)
            if response.shape != ref.shape:
                raise ValueError(f"response shape {response.shape}, reference shape {ref.shape}")
            self._maps[config.test_id] = np.asarray(response, dtype=np.float32)
            diff = np.abs(response - ref)
            intensity_range = ref.max() - ref.min()
            tolerance = 0.01 * intensity_range
            max_diff = diff.max()
            error_pct = 100 * max_diff / intensity_range if intensity_range > 0 else 0
            passed = bool(max_diff <= tolerance)

            # Provenance: record which reference file was actually compared
            # against and a short content hash, for traceability.
            ref_sha256 = self._hash_file(self.refs_dir / config.ref_file)

            return TestResult(
                test_id=config.test_id,
                filter_name=config.filter_name,
                phantom=config.phantom,
                passed=passed,
                max_diff=float(max_diff),
                tolerance=float(tolerance),
                error_pct=float(error_pct),
                time_seconds=elapsed,
                memory_mb=memory_mb,
                notes=config.notes,
                ref_file=config.ref_file,
                ref_sha256=ref_sha256,
            )

        except Exception as e:
            return TestResult(
                test_id=config.test_id,
                filter_name=config.filter_name,
                phantom=config.phantom,
                passed=False,
                max_diff=0,
                tolerance=0,
                error_pct=0,
                time_seconds=0,
                notes=f"Error: {str(e)}",
            )

    def get_all_tests(self) -> list[TestConfig]:
        """Get all Phase 1 test configurations from Table 6.1."""
        tests = []

        # ===== 1. MEAN FILTER =====
        # 1.a.1-4: checkerboard, different padding
        tests.append(
            TestConfig(
                "1.a.1",
                "Mean",
                "checkerboard",
                "1_a_1-ValidCRM.nii",
                lambda p: mean_filter(p, support=15, boundary=BoundaryCondition.ZERO),
                "M=15, zero padding",
            )
        )
        tests.append(
            TestConfig(
                "1.a.2",
                "Mean",
                "checkerboard",
                "1_a_2-ValidCRM.nii",
                lambda p: mean_filter(
                    p, support=15, boundary=BoundaryCondition.NEAREST
                ),
                "M=15, nearest padding",
            )
        )
        tests.append(
            TestConfig(
                "1.a.3",
                "Mean",
                "checkerboard",
                "1_a_3-ValidCRM.nii",
                lambda p: mean_filter(
                    p, support=15, boundary=BoundaryCondition.PERIODIC
                ),
                "M=15, periodic padding",
            )
        )
        tests.append(
            TestConfig(
                "1.a.4",
                "Mean",
                "checkerboard",
                "1_a_4-ValidCRM.nii",
                lambda p: mean_filter(p, support=15, boundary=BoundaryCondition.MIRROR),
                "M=15, mirror padding",
            )
        )
        # 1.b.1: 2D filter - SKIPPED (3D only)
        tests.append(
            TestConfig(
                "1.b.1",
                "Mean (2D)",
                "impulse_response",
                "1_b_1-ValidCRM.nii",
                lambda p: p,  # Placeholder - 2D not implemented
                "SKIPPED: 2D filter not implemented (3D only)",
                skipped=True,
            )
        )

        # ===== 2. LoG FILTER =====
        tests.append(
            TestConfig(
                "2.a",
                "LoG",
                "impulse_response",
                "2_a-ValidCRM.nii",
                lambda p: laplacian_of_gaussian(
                    p,
                    sigma_mm=3.0,
                    spacing_mm=2.0,
                    truncate=4.0,
                    boundary=BoundaryCondition.ZERO,
                ),
                "σ*=3mm, 4σ cutoff, zero padding",
            )
        )
        tests.append(
            TestConfig(
                "2.b",
                "LoG",
                "checkerboard",
                "2_b-ValidCRM.nii",
                lambda p: laplacian_of_gaussian(
                    p,
                    sigma_mm=5.0,
                    spacing_mm=2.0,
                    truncate=4.0,
                    boundary=BoundaryCondition.MIRROR,
                ),
                "σ*=5mm, 4σ cutoff, mirror padding",
            )
        )
        # 2.c: 2D filter - SKIPPED
        tests.append(
            TestConfig(
                "2.c",
                "LoG (2D)",
                "checkerboard",
                "2_c-ValidCRM.nii",
                lambda p: p,  # Placeholder
                "SKIPPED: 2D filter not implemented",
                skipped=True,
            )
        )

        # ===== 3. LAWS KERNELS =====
        # 3.a.x: impulse, E5L5S5
        tests.append(
            TestConfig(
                "3.a.1",
                "Laws",
                "impulse_response",
                "3_a_1-ValidCRM.nii",
                lambda p: laws_filter(p, "E5L5S5", boundary=BoundaryCondition.ZERO),
                "E5L5S5, zero padding",
            )
        )
        tests.append(
            TestConfig(
                "3.a.2",
                "Laws",
                "impulse_response",
                "3_a_2-ValidCRM.nii",
                lambda p: laws_filter(
                    p,
                    "E5L5S5",
                    boundary=BoundaryCondition.ZERO,
                    rotation_invariant=True,
                    pooling="max",
                ),
                "E5L5S5, zero, rot inv max",
            )
        )
        tests.append(
            TestConfig(
                "3.a.3",
                "Laws",
                "impulse_response",
                "3_a_3-ValidCRM.nii",
                lambda p: laws_filter(
                    p,
                    "E5L5S5",
                    boundary=BoundaryCondition.ZERO,
                    rotation_invariant=True,
                    pooling="max",
                    compute_energy=True,
                    energy_distance=7,
                ),
                "E5L5S5, zero, rot inv max, energy δ=7",
            )
        )
        # 3.b.x: checkerboard, E3W5R5
        tests.append(
            TestConfig(
                "3.b.1",
                "Laws",
                "checkerboard",
                "3_b_1-ValidCRM.nii",
                lambda p: laws_filter(p, "E3W5R5", boundary=BoundaryCondition.MIRROR),
                "E3W5R5, mirror padding",
            )
        )
        tests.append(
            TestConfig(
                "3.b.2",
                "Laws",
                "checkerboard",
                "3_b_2-ValidCRM.nii",
                lambda p: laws_filter(
                    p,
                    "E3W5R5",
                    boundary=BoundaryCondition.MIRROR,
                    rotation_invariant=True,
                    pooling="max",
                ),
                "E3W5R5, mirror, rot inv max",
            )
        )
        tests.append(
            TestConfig(
                "3.b.3",
                "Laws",
                "checkerboard",
                "3_b_3-ValidCRM.nii",
                lambda p: laws_filter(
                    p,
                    "E3W5R5",
                    boundary=BoundaryCondition.MIRROR,
                    rotation_invariant=True,
                    pooling="max",
                    compute_energy=True,
                    energy_distance=7,
                ),
                "E3W5R5, mirror, rot inv max, energy δ=7",
            )
        )
        # 3.c.x: 2D Laws - SKIPPED
        for i in ["1", "2", "3"]:
            tests.append(
                TestConfig(
                    f"3.c.{i}",
                    "Laws (2D)",
                    "checkerboard",
                    f"3_c_{i}-ValidCRM.nii",
                    lambda p: p,  # Placeholder
                    "SKIPPED: 2D filter not implemented",
                    skipped=True,
                )
            )

        # ===== 4. GABOR =====
        tests.append(
            TestConfig(
                "4.a.1",
                "Gabor",
                "impulse_response",
                "4_a_1-ValidCRM.nii",
                lambda p: gabor_filter(
                    p,
                    sigma_mm=10.0,
                    lambda_mm=4.0,
                    gamma=0.5,
                    theta=np.pi / 3,
                    spacing_mm=2.0,
                    boundary=BoundaryCondition.ZERO,
                ),
                "2D, σ*=10mm, λ*=4mm, γ=0.5, θ=π/3",
            )
        )
        tests.append(
            TestConfig(
                "4.a.2",
                "Gabor",
                "impulse_response",
                "4_a_2-ValidCRM.nii",
                lambda p: gabor_filter(
                    p,
                    sigma_mm=10.0,
                    lambda_mm=4.0,
                    gamma=0.5,
                    spacing_mm=2.0,
                    boundary=BoundaryCondition.ZERO,
                    rotation_invariant=True,
                    delta_theta=np.pi / 4,
                    pooling="average",
                    average_over_planes=True,
                ),
                "2D rot inv, Δθ=π/4, avg over planes",
            )
        )
        tests.append(
            TestConfig(
                "4.b.1",
                "Gabor",
                "sphere",
                "4_b_1-ValidCRM.nii",
                lambda p: gabor_filter(
                    p,
                    sigma_mm=20.0,
                    lambda_mm=8.0,
                    gamma=2.5,
                    theta=5 * np.pi / 4,
                    spacing_mm=2.0,
                    boundary=BoundaryCondition.MIRROR,
                ),
                "2D, σ*=20mm, λ*=8mm, γ=2.5, θ=5π/4",
            )
        )
        tests.append(
            TestConfig(
                "4.b.2",
                "Gabor",
                "sphere",
                "4_b_2-ValidCRM.nii",
                lambda p: gabor_filter(
                    p,
                    sigma_mm=20.0,
                    lambda_mm=8.0,
                    gamma=2.5,
                    spacing_mm=2.0,
                    boundary=BoundaryCondition.MIRROR,
                    rotation_invariant=True,
                    delta_theta=np.pi / 8,
                    pooling="average",
                    average_over_planes=True,
                ),
                "2D rot inv, Δθ=π/8, avg over planes",
            )
        )

        # ===== 5. DAUBECHIES 2 =====
        tests.append(
            TestConfig(
                "5.a.1",
                "Daubechies 2",
                "impulse_response",
                "5_a_1-ValidCRM.nii",
                lambda p: wavelet_transform(
                    p,
                    wavelet="db2",
                    level=1,
                    decomposition="LHL",
                    boundary=BoundaryCondition.ZERO,
                ),
                "LHL level 1, zero padding",
            )
        )
        tests.append(
            TestConfig(
                "5.a.2",
                "Daubechies 2",
                "impulse_response",
                "5_a_2-ValidCRM.nii",
                lambda p: wavelet_transform(
                    p,
                    wavelet="db2",
                    level=1,
                    decomposition="LHL",
                    boundary=BoundaryCondition.ZERO,
                    rotation_invariant=True,
                    pooling="average",
                ),
                "LHL level 1, zero, rot inv avg",
            )
        )

        # ===== 6. COIFLET 1 =====
        tests.append(
            TestConfig(
                "6.a.1",
                "Coiflet 1",
                "sphere",
                "6_a_1-ValidCRM.nii",
                lambda p: wavelet_transform(
                    p,
                    wavelet="coif1",
                    level=1,
                    decomposition="HHL",
                    boundary=BoundaryCondition.PERIODIC,
                ),
                "HHL level 1, periodic padding",
            )
        )
        tests.append(
            TestConfig(
                "6.a.2",
                "Coiflet 1",
                "sphere",
                "6_a_2-ValidCRM.nii",
                lambda p: wavelet_transform(
                    p,
                    wavelet="coif1",
                    level=1,
                    decomposition="HHL",
                    boundary=BoundaryCondition.PERIODIC,
                    rotation_invariant=True,
                    pooling="average",
                ),
                "HHL level 1, periodic, rot inv avg",
            )
        )

        # ===== 7. HAAR =====
        tests.append(
            TestConfig(
                "7.a.1",
                "Haar",
                "checkerboard",
                "7_a_1-ValidCRM.nii",
                lambda p: wavelet_transform(
                    p,
                    wavelet="haar",
                    level=2,
                    decomposition="LLL",
                    boundary=BoundaryCondition.MIRROR,
                    rotation_invariant=True,
                    pooling="average",
                ),
                "LLL level 2, mirror, rot inv avg",
            )
        )
        tests.append(
            TestConfig(
                "7.a.2",
                "Haar",
                "checkerboard",
                "7_a_2-ValidCRM.nii",
                lambda p: wavelet_transform(
                    p,
                    wavelet="haar",
                    level=2,
                    decomposition="HHH",
                    boundary=BoundaryCondition.MIRROR,
                    rotation_invariant=True,
                    pooling="average",
                ),
                "HHH level 2, mirror, rot inv avg",
            )
        )

        # ===== 8. SIMONCELLI =====
        tests.append(
            TestConfig(
                "8.a.1",
                "Simoncelli",
                "checkerboard",
                "8_a_1-ValidCRM.nii",
                lambda p: simoncelli_wavelet(p, level=1),
                "B map level 1, periodic",
            )
        )
        tests.append(
            TestConfig(
                "8.a.2",
                "Simoncelli",
                "checkerboard",
                "8_a_2-ValidCRM.nii",
                lambda p: simoncelli_wavelet(p, level=2),
                "B map level 2, periodic",
            )
        )
        tests.append(
            TestConfig(
                "8.a.3",
                "Simoncelli",
                "checkerboard",
                "8_a_3-ValidCRM.nii",
                lambda p: simoncelli_wavelet(p, level=3),
                "B map level 3, periodic",
            )
        )

        # ===== 9. RIESZ-LoG =====
        tests.append(
            TestConfig(
                "9.a",
                "Riesz-LoG",
                "impulse_response",
                "9_a-ValidCRM.nii",
                lambda p: riesz_log(
                    p,
                    sigma_mm=3.0,
                    spacing_mm=2.0,
                    order=(1, 0, 0),
                    truncate=4.0,
                    boundary=BoundaryCondition.ZERO,
                ),
                "σ=3mm, l=(1,0,0), zero padding",
            )
        )
        tests.append(
            TestConfig(
                "9.b.1",
                "Riesz-LoG",
                "sphere",
                "9_b_1-ValidCRM.nii",
                lambda p: riesz_log(
                    p,
                    sigma_mm=3.0,
                    spacing_mm=2.0,
                    order=(0, 2, 0),
                    truncate=4.0,
                    boundary=BoundaryCondition.ZERO,
                ),
                "σ=3mm, l=(0,2,0), zero padding",
            )
        )
        # 9.b.2: with structure tensor alignment - not implemented
        tests.append(
            TestConfig(
                "9.b.2",
                "Riesz-LoG (aligned)",
                "sphere",
                "9_b_2-ValidCRM.nii",
                lambda p: p,  # Placeholder
                "SKIPPED: Structure tensor alignment not implemented",
            )
        )

        # ===== 10. RIESZ-SIMONCELLI =====
        tests.append(
            TestConfig(
                "10.a",
                "Riesz-Simoncelli",
                "impulse_response",
                "10_a-ValidCRM.nii",
                lambda p: riesz_simoncelli(
                    p, level=1, order=(1, 0, 0), boundary=BoundaryCondition.ZERO
                ),
                "B map level 1, l=(1,0,0), zero padding",
            )
        )
        tests.append(
            TestConfig(
                "10.b.1",
                "Riesz-Simoncelli",
                "pattern_1",
                "10_b_1-ValidCRM.nii",
                lambda p: riesz_simoncelli(
                    p, level=1, order=(0, 2, 0), boundary=BoundaryCondition.NEAREST
                ),
                "B map level 1, l=(0,2,0), nearest padding",
            )
        )
        # 10.b.2: with structure tensor alignment - not implemented
        tests.append(
            TestConfig(
                "10.b.2",
                "Riesz-Simoncelli (aligned)",
                "pattern_1",
                "10_b_2-ValidCRM.nii",
                lambda p: p,  # Placeholder
                "SKIPPED: Structure tensor alignment not implemented",
            )
        )

        return tests

    def run_all_tests(
        self, verbose: bool = True, filter_pattern: Optional[str] = None
    ) -> list[TestResult]:
        """Run all Phase 1 tests."""
        tests = self.get_all_tests()
        if filter_pattern:
            tests = [
                c for c in tests if filter_pattern.lower() in c.filter_name.lower()
            ]

        self._results = []

        if verbose:
            print("=" * 70)
            print("IBSI 2 Phase 1 Filter Validation")
            print("=" * 70)
            print()

        for config in tests:
            result = self.run_test(config)
            self._results.append(result)

            if verbose:
                if result.skipped:
                    status = "⏭"
                elif (
                    result.notes.startswith("Error")
                    or result.notes == "Reference file not found"
                ):
                    status = "⚠"
                elif result.notes == "MISSING_REF":
                    status = "❗"
                elif result.passed:
                    status = "✓"
                else:
                    status = "✗"

                time_str = (
                    f"{result.time_seconds*1000:7.1f}ms"
                    if result.time_seconds > 0
                    else "    N/A"
                )
                mem_str = (
                    f"{result.memory_mb:6.1f}MB" if result.memory_mb > 0 else "   N/A"
                )
                print(
                    f"{result.test_id:8s} {result.filter_name:20s} "
                    f"diff={result.max_diff:8.4f} tol={result.tolerance:6.3f} "
                    f"err={result.error_pct:5.2f}% {time_str} {mem_str} {status}"
                )

        # Self-check: skip is checked before the numeric verdict (see run_test()),
        # so a SKIPPED test must never also be reported as passed.
        for r in self._results:
            assert not (r.skipped and r.passed), (
                f"Test {r.test_id} is marked both skipped and passed - "
                "skip must take priority over the numeric verdict."
            )

        if verbose:
            print()
            print("=" * 80)
            passed = sum(1 for r in self._results if r.passed)
            skipped = sum(1 for r in self._results if r.skipped)
            missing = sum(1 for r in self._results if r.notes == "MISSING_REF")
            failed = sum(
                1
                for r in self._results
                if not r.passed and not r.skipped and r.notes != "MISSING_REF"
            )
            print(
                f"Summary: {passed} passed, {failed} failed, {skipped} skipped, {missing} missing ref"
            )
            print("=" * 80)

        return self._results

    def save_cache(self, output_path: Path | str):
        """Save results to JSON cache."""
        # Convert TestResults to dicts
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

    def maps(self) -> dict[str, np.ndarray]:
        """Our response maps of the compared tests, for the gate's value check."""
        return dict(sorted(self._maps.items()))

    def save_results(self, output_path: Path):
        """Save results to JSON."""
        results_dict = self.get_results_dict()
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    def get_results_dict(self) -> dict:
        """Get structured results dictionary for baseline comparison."""
        passed = sum(1 for r in self._results if r.passed)
        skipped = sum(1 for r in self._results if r.skipped)
        missing = sum(1 for r in self._results if r.notes == "MISSING_REF")
        failed = sum(
            1
            for r in self._results
            if not r.passed and not r.skipped and r.notes != "MISSING_REF"
        )

        # Build per-test dict with detailed metrics
        tests = {}
        for r in self._results:
            tests[r.test_id] = {
                "filter": r.filter_name,
                "phantom": r.phantom,
                "status": (
                    "SKIP"
                    if r.skipped
                    else (
                        "PASS"
                        if r.passed
                        else ("MISSING" if r.notes == "MISSING_REF" else "FAIL")
                    )
                ),
                "error_pct": float(round(r.error_pct, 4)),
                "time_ms": float(round(r.time_seconds * 1000, 1)),
                "memory_mb": float(round(r.memory_mb, 1)),
                "ref_file": r.ref_file,
                "ref_sha256": r.ref_sha256,
            }

        return {
            "total": len(self._results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "missing": missing,
            "provenance": {
                "pictologics_version": pictologics.__version__,
                "reference_dataset_source": (
                    f"{IBSI2_REFERENCE_DATA_REPO_URL} (subfolder: reference_response_maps)"
                ),
                "reference_dir": self._repo_path(self.refs_dir),
            },
            "tests": tests,
        }

    def generate_docs(self, output_path: Path):
        """Generate IBSI 2 compliance documentation page."""
        output_path = Path(output_path)

        # Counts reused across several sections below.
        passed = sum(1 for r in self._results if r.passed)
        skipped = sum(1 for r in self._results if r.skipped)
        missing = sum(1 for r in self._results if r.notes == "MISSING_REF")
        failed = sum(
            1
            for r in self._results
            if not r.passed and not r.skipped and r.notes != "MISSING_REF"
        )
        compared = passed + failed
        gabor_passed = sum(
            1 for r in self._results if r.passed and r.filter_name == "Gabor"
        )
        strict3d_passed = passed - gabor_passed

        # Lookup for the exact worked example (Test 8.a.1), used both in the
        # "How to Run the Benchmarks" example and the Provenance section.
        test_8a1 = next((r for r in self._results if r.test_id == "8.a.1"), None)
        ref_8a1_hash = test_8a1.ref_sha256 if test_8a1 else ""

        # Header and Overview
        lines = []
        lines.append("# IBSI 2 Compliance: Convolutional Filters")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(
            "The Image Biomarker Standardisation Initiative Chapter 2 (IBSI 2) focuses on st"
            "andardizing convolutional image filters for radiomics. This page documents Pic"
            "tologics' compliance with **IBSI 2 Phase 1**: technical validation using digi"
            "tal phantoms."
        )
        lines.append("")
        lines.append("!!! important")
        lines.append(
            "    **Pictologics accepts volumetric images and calculates radiomic features "
            "over a 3D ROI.** Its IBSI Gabor implementation uses 2D plane kernels and can "
            "average their response maps over three orthogonal planes; this is not a native "
            "3D Gabor convolution, and it does not imply support for a general slice-wise "
            "2D radiomics workflow."
        )
        lines.append("    ")
        lines.append(
            "    The library is designed specifically for volumetric medical imaging analysis"
            " (CT, MRI, PET). A slice-by-slice 2D radiomics workflow is not supported, as it"
            " loses spatial information needed for accurate volumetric feature extraction."
        )
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append("## How to Run the Benchmarks")
        lines.append("")
        lines.append("### 1. Download the Data")
        lines.append("")
        lines.append(
            "The IBSI 2 reference datasets (digital phantoms) are available on the [IBSI G"
            "itHub repository](https://github.com/theibsi/data_sets)."
        )
        lines.append("")
        lines.append(
            "- **Digital Phantoms**: Download the phantom NIfTI files (e.g., `checkerboard."
            "nii.gz`, `impulse_response.nii.gz`) from the `ibsi_2_validation` folder."
        )
        lines.append("")
        lines.append(
            "Place these files in a local directory (e.g., `data/ibsi2/`) to run the benchm"
            "arks."
        )
        lines.append("")
        lines.append(
            "### 2. Run Configurations Programmatically using `RadiomicsPipeline`"
        )
        lines.append("")
        lines.append(
            "You can run IBSI 2 filter configurations programmatically using the `Radiomics"
            "Pipeline` class."
        )
        lines.append("")
        lines.append("```python")
        lines.append("from pictologics import RadiomicsPipeline")
        lines.append("")
        lines.append("pipeline = RadiomicsPipeline()")
        lines.append("")
        lines.append("# Define a Gabor filter configuration (generic example)")
        lines.append("# 2D plane kernels, rotation invariant, averaged over the")
        lines.append("# three orthogonal planes; features are calculated over the 3D ROI.")
        lines.append("gabor_config = [")
        lines.append("    # 1. IBSI 2 Preprocessing")
        lines.append(
            '    {"step": "resample", "params": {"new_spacing": (1.0, 1.0, 1.0), "interp'
            'olation": "cubic"}},'
        )
        lines.append('    {"step": "round_intensities", "params": {}},')
        lines.append(
            '    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400}},'
        )
        lines.append("    ")
        lines.append("    # 2. Apply Gabor filter")
        lines.append("    #    Generic illustrative settings - NOT an official IBSI test")
        lines.append("    #    configuration. rotation_invariant=True requires delta_theta.")
        lines.append('    {"step": "filter", "params": {')
        lines.append('        "type": "gabor",')
        lines.append('        "sigma_mm": 5.0,')
        lines.append('        "lambda_mm": 4.0,')
        lines.append('        "gamma": 0.5,')
        lines.append('        "rotation_invariant": True,')
        lines.append('        "delta_theta": 0.7853981633974483,  # pi/4')
        lines.append('        "average_over_planes": True,')
        lines.append('        "pooling": "average"')
        lines.append("    }},")
        lines.append("    ")
        lines.append("    # 3. Extract Intensity Features from the response map")
        lines.append(
            '    {"step": "extract_features", "params": {"families": ["intensity"]}}'
        )
        lines.append("]")
        lines.append("")
        lines.append('pipeline.add_config("ibsi2_gabor_demo", gabor_config)')
        lines.append("")
        lines.append("# Run on an image")
        lines.append(
            'results = pipeline.run("path/to/phantom.nii.gz", config_names=["ibsi2_gabor_demo"])'
        )
        lines.append('print(results["ibsi2_gabor_demo"])')
        lines.append("```")
        lines.append("")
        lines.append("!!! note")
        lines.append(
            "    The example above shows a **Gabor filter** configuration. This is just one"
            " example. You can configure any IBSI 2 compliant filter (Mean, LoG, Laws, Wav"
            "elet, etc.) similarly. For full specifications of filter parameters, please r"
            "efer to the [Image Filtering](user_guide/image_filtering.md) guide and the [I"
            "BSI 2 Reference Manual](https://arxiv.org/abs/2006.05470)."
        )
        lines.append("")
        lines.append("### 3. Reproduce an Exact IBSI Phase 1 Test (Test 8.a.1)")
        lines.append("")
        lines.append(
            "The example above is a **generic illustration** of the `RadiomicsPipeline` API"
            ", not an official IBSI test. The example below is different: it is one **exact,"
            " named IBSI 2 Phase 1 test** — reproduced precisely as the compliance harness "
            "(`dev/IBSI2/verify_ibsi2_compliance.py`) runs it — that you can check yourself "
            "against the official reference."
        )
        lines.append("")
        lines.append("- **Official test ID**: `8.a.1` (Simoncelli wavelet, Table 6.1)")
        lines.append(
            "- **Phantom file**: `checkerboard.nii.gz` (see [Download the Data](#1-download"
            "-the-data) above)"
        )
        lines.append(
            "- **Preprocessing**: none. The harness loads the phantom NIfTI directly "
            "(`nib.load(...).get_fdata().astype(np.float32)`) and passes it straight to the "
            "filter — there is no resampling, intensity rounding, or resegmentation for "
            "Phase 1 filter tests."
        )
        lines.append(
            "- **Exact filter call**: `simoncelli_wavelet(phantom, level=1)` — `boundary` "
            "defaults to `BoundaryCondition.PERIODIC`, the filter's inherent FFT-based "
            "boundary handling."
        )
        lines.append(
            "- **Comparison rule**: `max_diff = abs(response - reference).max()`; the test "
            "passes if `max_diff <= 0.01 * (reference.max() - reference.min())` — the same "
            "tolerance basis given in [Tolerance Criteria](#tolerance-criteria) below."
        )
        ref_note = (
            f" (the file validated below has sha256 short-hash `{ref_8a1_hash}`; see "
            "[Provenance](#provenance))"
            if ref_8a1_hash
            else ""
        )
        lines.append(
            "- **Reference/response map**: `8_a_1-ValidCRM.nii`, obtained from the "
            "`reference_response_maps` subfolder of the [IBSI 2 reference data reposito"
            f"ry]({IBSI2_REFERENCE_DATA_REPO_URL}); place it under `references/response_"
            f"maps/` alongside the phantoms{ref_note}."
        )
        lines.append("")
        lines.append("```python")
        lines.append("import numpy as np")
        lines.append("import nibabel as nib")
        lines.append("from pictologics.filters import simoncelli_wavelet")
        lines.append("")
        lines.append("# Official IBSI 2 Phase 1 Test 8.a.1 - exact, not illustrative")
        lines.append(
            'phantom = nib.load("data/phantoms/checkerboard.nii.gz").get_fdata().astype(np.float32)'
        )
        lines.append(
            "response = simoncelli_wavelet(phantom, level=1)  # boundary defaults to PERIODIC"
        )
        lines.append("")
        lines.append("# Compare against the official IBSI reference response map")
        lines.append(
            'reference = nib.load("references/response_maps/8_a_1-ValidCRM.nii").get_fdata()'
            ".astype(np.float32)"
        )
        lines.append("max_diff = np.abs(response - reference).max()")
        lines.append("tolerance = 0.01 * (reference.max() - reference.min())")
        lines.append("assert max_diff <= tolerance  # IBSI 2 Phase 1 pass criterion")
        lines.append("```")
        lines.append("")
        lines.append("## Phase 1 Results")
        lines.append("")
        lines.append("### Filter Performance Overview")
        lines.append("")
        lines.append("| Test | Filter | Phantom | Error % | Time | Memory | Status |")
        lines.append("|------|--------|---------|---------|------|--------|--------|")

        for result in self._results:
            if result.skipped:
                status = "⏭ SKIP"
                error = "-"
                time_str = "-"
                mem_str = "-"
            elif result.notes == "MISSING_REF":
                status = "❗ REF."
                error = "-"
                time_str = "-"
                mem_str = "-"
            elif result.passed:
                status = "✅ PASS"
                error = f"{result.error_pct:.2f}%"
                time_str = f"{result.time_seconds*1000:.0f}ms"
                mem_str = f"{result.memory_mb:.1f}MB"
            else:
                status = "❌ FAIL"
                error = f"{result.error_pct:.2f}%"
                time_str = f"{result.time_seconds*1000:.0f}ms"
                mem_str = f"{result.memory_mb:.1f}MB"

            lines.append(
                f"| {result.test_id} | {result.filter_name} | {result.phantom} | {error} | {time_str} | {mem_str} | {status} |"
            )

        lines.append("")
        lines.append("### Tolerance Criteria")
        lines.append("")
        lines.append("All tests use the IBSI 2 standard tolerance:")
        lines.append("```")
        lines.append("max_difference ≤ 0.01 × (reference_max - reference_min)")
        lines.append("```")
        lines.append("")
        lines.append("### Provenance")
        lines.append("")
        lines.append('!!! info "Reproducibility Provenance"')
        lines.append(f"    - **Pictologics version**: `{pictologics.__version__}`")
        lines.append(
            "    - **IBSI 2 reference manual**: version 9 (the revision bundled under "
            "`dev/IBSI2/documentation/`)"
        )
        lines.append(
            "    - **Reference dataset source**: `reference_response_maps` subfolder of "
            f"the [IBSI 2 reference data repository]({IBSI2_REFERENCE_DATA_REPO_URL})"
        )
        lines.append(f"    - **Local reference directory**: `{self._repo_path(self.refs_dir)}`")
        ref_example = (
            f" (e.g. Test `8.a.1` was validated against `8_a_1-ValidCRM.nii`, "
            f"sha256 short-hash `{ref_8a1_hash}`)"
            if ref_8a1_hash
            else ""
        )
        lines.append(
            f"    - Per-test reference filenames and short (12-character) SHA-256 content "
            f"hashes are recorded for all {compared} compared tests (passed or failed) in "
            "the JSON results this script writes (`--output <file>.json`), keyed by test "
            f"ID under `ref_file` / `ref_sha256`{ref_example}."
        )
        lines.append("")
        lines.append("## Known Deviations")
        lines.append("")
        lines.append("### 2D Filters Not Implemented (5 Tests Skipped)")
        lines.append("")
        lines.append("!!! note")
        lines.append("    **Design Decision: 3D Volumetric Processing Only**")
        lines.append("    ")
        lines.append(
            "    Pictologics calculates radiomic features over a **3D volumetric ROI** and "
            "does not provide a general slice-wise 2D radiomics workflow. This is a "
            "deliberate design choice for clinical radiomics workflows with volumetric "
            "medical imaging data (CT, MRI, PET scans). Note that the Gabor filter is an "
            "explicit exception at the *filter* level: it applies 2D plane kernels and can "
            "average the resulting response maps over three orthogonal planes, as IBSI "
            "defines. Feature statistics are still calculated over the volumetric ROI."
        )
        lines.append("")
        lines.append(
            "The following tests are intentionally skipped because they require 2D filter "
            "implementations: **Test 1.b.1** (Mean Filter 2D), **Test 2.c** (LoG 2D), and "
            "**Tests 3.c.1-3.c.3** (Laws 2D)."
        )
        lines.append("")
        lines.append("### Structure Tensor Alignment (Reference Missing)")
        lines.append("")
        lines.append(
            "The following tests are currently not implemented because they require structu"
            "re tensor alignment (steering the Riesz filter response towards the local "
            "dominant orientation) **and the IBSI 2 reference dataset does not contain the "
            "corresponding validity response maps**:"
        )
        lines.append("")
        lines.append(
            "- **Tests 9.b.2, 10.b.2**: Riesz Filter Alignment (ValidCRM missing)"
        )
        lines.append("")
        lines.append("!!! warning")
        lines.append("    **Reference Data Unavailable**")
        lines.append("    ")
        lines.append(
            "    The official IBSI 2 reference dataset **does not contain** the reference va"
            "lidity maps for these tests (`9_b_2-ValidCRM.nii`, `10_b_2-ValidCRM.nii`). "
            "Therefore, these tests cannot be validated and are marked as **❗ REF.** "
            "(Reference Missing)."
        )
        lines.append("")
        lines.append("### Reference Missing (Unsteered Riesz-Simoncelli)")
        lines.append("")
        lines.append(
            "Unlike the two tests above, **Test 10.a is not a steering test**: it is an "
            "implemented, unsteered Riesz-Simoncelli response map "
            "(`riesz_simoncelli(level=1, order=(1, 0, 0))`, no structure-tensor alignment). "
            "It is grouped separately here because the IBSI 2 reference dataset simply does "
            "not distribute a validity response map for it (`10_a-ValidCRM.nii` is absent), "
            "so it cannot be validated for a reason unrelated to structure-tensor alignment:"
        )
        lines.append("")
        lines.append(
            "- **Test 10.a**: unsteered Riesz-Simoncelli response map (ValidCRM missing)"
        )
        lines.append("")
        lines.append("### Summary")
        lines.append("")

        lines.append(
            f"Under **strict 3D validation** (excluding the IBSI-defined 2D Gabor "
            f"response-map tests), **{strict3d_passed} tests passed**. Including the "
            f"**{gabor_passed} 2D Gabor tests** (`4.a.1`, `4.a.2`, `4.b.1`, `4.b.2` — "
            "computed via 2D plane kernels, optionally averaged over the three orthogonal "
            f"planes, as Pictologics implements them), **{passed} of {compared} compared "
            "tests passed** in total; these are the same underlying runs, only split by "
            "category, not a separate claim. "
            f"**{skipped} tests were skipped** as they relate to 2D filters (Mean, LoG, "
            "Laws) which are not applicable to this 3D-focused library. "
            f"**{missing} tests are marked as missing reference** because the IBSI 2 "
            "reference dataset does not distribute a validity response map for them — two "
            "because they require structure-tensor alignment (`9.b.2`, `10.b.2`), and one "
            "unsteered test whose reference map is simply absent (`10.a`)."
        )
        lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Generated compliance report at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="IBSI 2 Phase 1 Validation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Path to IBSI2 data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Run only tests matching this filter name (e.g. 'Simoncelli')",
    )
    args = parser.parse_args()

    validator = IBSIPhase1Validator(args.data_dir)
    results = validator.run_all_tests(
        verbose=not args.quiet, filter_pattern=args.filter
    )

    if args.output:
        validator.save_results(args.output)
        print(f"\nResults saved to {args.output}")

    # Return exit code based on results
    failed = sum(
        1 for r in results if not r.passed and not r.skipped and r.notes != "MISSING_REF"
    )
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    exit(main())
