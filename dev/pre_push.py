#!/usr/bin/env python3
"""
Pre-push workflow script for Pictologics.

Runs all quality checks before pushing to git repository:
1. Clear all cache directories, then build the numba cache in the background
2. Run ruff linting
3. Run mypy type checking
4. Check towncrier changelog fragments
5. Run pytest with coverage (it needs no compiled code), then wait for the numba build
6. Run IBSI compliance verification
7. Run performance benchmarks

Usage:
    poetry run python dev/pre_push.py                    # Run all checks
    poetry run python dev/pre_push.py --update-baseline  # Save new baselines (refused after failures)
    poetry run python dev/pre_push.py --allow-missing-data  # Skip IBSI checks whose data is missing
    poetry run python dev/pre_push.py --skip-compliance  # Skip IBSI checks
    poetry run python dev/pre_push.py --skip-benchmarks  # Skip benchmarks
    poetry run python dev/pre_push.py --skip-towncrier   # Skip towncrier check
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared configuration
# Import utilities
from dev import compliance_utils  # noqa: E402
from dev.compliance_utils import (  # noqa: E402
    Colors,
    ComplianceResult,
    compare_maps,
    compare_values,
    environment,
    load_json_baseline,
    other_environment,
    phase1_margin_problems,
    print_compliance_report,
    print_error,
    print_header,
    print_ibsi_summary,
    print_phase3_summary,
    print_step,
    print_success,
    print_warning,
    redirect_caches,
    run_command,
    save_json_baseline,
    save_npz_baseline,
)
from dev.pipeline_config import (  # noqa: E402
    BENCHMARK_BASELINE,
    BENCHMARK_CACHE,
    DOCS_DIR,
    GATE_CACHE_DIR,
    GATE_LOG_DIR,
    IBSI1_CACHE,
    IBSI1_VALUES,
    IBSI2_BASELINE,
    IBSI2_DATA_DIR,
    IBSI2_PHASE1_CACHE,
    IBSI2_PHASE1_MAPS,
    IBSI2_PHASE2_BASELINE,
    IBSI2_PHASE2_CACHE,
    IBSI2_PHASE2_VALUES,
    IBSI2_PHASE3_BASELINE,
    IBSI2_PHASE3_CACHE,
    IBSI2_PHASE3_VALUES,
    IBSI_BASELINE,
    PHASE3_WORKERS,
    REGRESSION_THRESHOLD,
    RESULTS_DIR,
)


def clear_caches() -> bool:
    """Delete the old caches in the project folder. The tools of this run keep their caches
    in GATE_CACHE_DIR (outside OneDrive), which redirect_caches() empties."""
    print_step("Clearing cache directories...")

    paths = [
        PROJECT_ROOT / name
        for name in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "coverage.xml")
    ]
    paths += list(PROJECT_ROOT.glob(".coverage*"))
    for folder in ("pictologics", "tests", "dev"):
        for name in ("__pycache__", ".mypy_cache"):
            paths += list((PROJECT_ROOT / folder).rglob(name))
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)  # another clear may remove files first
        else:
            path.unlink(missing_ok=True)
    left = [path for path in paths if path.exists()]
    if left:
        print_error(f"Could not clear {len(left)} caches, for example {left[0]}")
        return False

    print_success(f"Cache directories cleared (this run keeps its caches in {GATE_CACHE_DIR})")
    return True


def start_numba_build() -> subprocess.Popen[str]:
    """Compile the package warm-up into the (empty) numba cache while the tool steps run.
    One process compiles, so no two processes write the same cache index. A kernel that
    fails to compile is an error here (-W error), not a warning."""
    print_step("Building the numba cache in the background...")
    return subprocess.Popen(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", "import pictologics"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def finish_numba_build(build: subprocess.Popen[str]) -> bool:
    """Wait for the background build before the IBSI steps use the cache."""
    output, _ = build.communicate()
    if build.returncode == 0:
        print_success("Numba cache built: every warm-up kernel compiles with the compiler on")
        return True
    print(output)
    print_error("Numba build failed (see above)")
    return False


def run_ruff() -> bool:
    """Run ruff linting."""
    print_step("Running ruff linting...")

    returncode, _, _ = run_command(
        ["poetry", "run", "ruff", "check", "pictologics", "tests", "scripts"], cwd=PROJECT_ROOT
    )

    if returncode == 0:
        print_success("Ruff: No issues found")
        return True
    else:
        print_error("Ruff: Issues found (see above)")
        return False


def run_mypy() -> bool:
    """Run mypy type checking."""
    print_step("Running mypy type checking...")

    returncode, _, _ = run_command(
        ["poetry", "run", "mypy", "pictologics", "scripts"], cwd=PROJECT_ROOT
    )

    if returncode == 0:
        print_success("Mypy: No errors found")
        return True
    else:
        print_error("Mypy: Errors found (see above)")
        return False


# Without a prompt: a missing key or network gives a warning, not a wait
FETCH_MAIN = [
    "git",
    "-c",
    "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=10",
    "fetch",
    "--quiet",
    "origin",
    "main",
]


def run_towncrier_check() -> bool:
    """Check for unreleased changelog fragments via towncrier."""
    print_step("Checking for changelog fragments (towncrier)...")

    # Compare with the current origin/main, not with the last fetched copy
    if run_command(FETCH_MAIN, cwd=PROJECT_ROOT)[0] != 0:
        print_warning("git fetch origin main failed; comparing with the last fetched origin/main")
    returncode, stdout, _ = run_command(
        ["poetry", "run", "towncrier", "check", "--compare-with", "origin/main"], cwd=PROJECT_ROOT
    )

    if returncode == 0:
        print_success("Towncrier: Changelog fragments found (or on main branch)")
        return True
    else:
        print_warning(
            "Towncrier: No changelog fragments found. "
            "Create one with: poetry run towncrier create +description.added"
        )
        return False


def run_pytest() -> tuple[bool, float]:
    """Run pytest with coverage. Returns (success, coverage_percentage)."""
    print_step("Running pytest with coverage (parallel)...")

    # Run pytest with live output (capture=False)
    # -n auto: Parallel execution (pytest-xdist)
    # -q: Quiet mode (suppress verbose list, override -v in ini)
    coverage_xml = GATE_CACHE_DIR / "coverage.xml"
    coverage_xml.unlink(missing_ok=True)  # a crash must not show the number of an old run
    returncode, _, _ = run_command(
        [
            "poetry",
            "run",
            "pytest",
            "-n",
            "auto",
            "-q",
            "-p",
            "no:cacheprovider",
            "--cov=pictologics",
            "--cov-report=term-missing",
            f"--cov-report=xml:{coverage_xml}",
        ],
        cwd=PROJECT_ROOT,
        capture=False,
    )

    # Parse coverage from XML since we didn't capture stdout
    coverage = 0.0
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(coverage_xml)
        root = tree.getroot()
        line_rate = float(root.attrib.get("line-rate", 0))
        coverage = line_rate * 100
    except (FileNotFoundError, ValueError, ET.ParseError):
        print_warning("Could not parse coverage.xml")

    if returncode == 0:
        print_success(f"Pytest: All tests passed (Coverage: {coverage:.1f}%)")
        return True, coverage
    else:
        print_error("Pytest: Some tests failed")
        return False, coverage


IBSI1_DIR = PROJECT_ROOT / "dev" / "IBSI1"
IBSI_DATA = {
    "IBSI 1": [
        IBSI1_DIR / "data" / f
        for f in ("phantom.nii.gz", "mask.nii.gz", "CT_phantom.nii.gz", "CT_mask.nii.gz")
    ]
    + [
        IBSI1_DIR / "references" / f"refs_{c}.json"
        for c in ("digital_phantom", "config_c", "config_d", "config_e")
    ],
    "IBSI 2 Phase 1": [
        IBSI2_DATA_DIR / "data" / "phantoms",
        IBSI2_DATA_DIR / "references" / "response_maps",
    ],
    "IBSI 2 Phase 2": [
        IBSI2_DATA_DIR / "data" / "ct_phantom" / "image.nii.gz",
        IBSI2_DATA_DIR / "data" / "ct_phantom" / "mask.nii.gz",
        IBSI2_DATA_DIR / "references" / "feature_values" / "reference_values.csv",
    ],
    "IBSI 2 Phase 3": [
        IBSI2_DATA_DIR / "data" / "validation" / d for d in ("ct", "pet", "mri", "masks")
    ]
    + [IBSI2_DATA_DIR / "references" / "phase3_submissions"],
}


def _missing_data(name: str, allow_missing_data: bool) -> ComplianceResult | None:
    """A result for a step whose data or references are missing: a failure, or a skip with
    --allow-missing-data. None when everything is there."""
    missing = [path for path in IBSI_DATA[name] if not path.exists()]
    if not missing:
        return None
    listed = ", ".join(os.path.relpath(path, PROJECT_ROOT) for path in missing)
    if allow_missing_data:
        print_warning(f"{name} skipped: missing {listed}")
        return ComplianceResult(name=name, passed=True)
    print_error(f"{name} failed: missing {listed} (use --allow-missing-data to skip)")
    return ComplianceResult(name=name, passed=False)


def _may_update(result: ComplianceResult, accept_changes: bool) -> bool:
    """Baselines are saved only after a clean check, or when changes are accepted on purpose."""
    if result.passed or accept_changes:
        return True
    print_error(
        f"{result.name}: baseline NOT updated, the check failed (use --accept-changes to accept)"
    )
    return False


def _crashed(name: str, error: Exception) -> tuple[ComplianceResult, dict[str, Any], None]:
    print_error(f"{name} check crashed: {error}")
    import traceback

    traceback.print_exc()
    return ComplianceResult(name=name, passed=False), {}, None


def run_ibsi_compliance(
    update_baseline: bool = False,
    accept_changes: bool = False,
    allow_missing_data: bool = False,
    write_docs: bool = False,
) -> tuple[ComplianceResult, dict[str, Any], Any]:
    """Run IBSI compliance verification and compare individual features to baseline."""
    print_step("Running IBSI compliance verification...")
    missing = _missing_data("IBSI 1", allow_missing_data)
    if missing is not None:
        return missing, {}, None

    try:
        from dev.verify_ibsi_compliance import TestIBSICompliance

        # Setup and run via centralized method
        test = TestIBSICompliance()
        test.setUp()

        # This now triggers run_all_configs() internally if not cached
        feature_results, total_time, max_mem = test.get_detailed_results()

        current_results = {
            "timestamp": datetime.now().isoformat(),
            "features": feature_results,
            "total_time_seconds": total_time,
            "memory_mb": max_mem,
        }
        values = test.values()
        baseline_values = load_json_baseline(IBSI1_VALUES) if IBSI1_VALUES.exists() else None

        # Load baseline and standard report
        baseline = load_json_baseline(IBSI_BASELINE)
        stats: dict[str, float] = {}
        result = print_compliance_report(
            "IBSI 1",
            current_results,
            baseline,
            key_label="Feature Key",
            value_changes=None
            if baseline_values is None
            else compare_values(values, baseline_values, stats=stats),
            value_stats=stats,
            other=other_environment(baseline),
        )

        update = update_baseline and _may_update(result, accept_changes)
        if update:
            test.save_cache(str(IBSI1_CACHE))
            save_json_baseline(IBSI_BASELINE, {**current_results, "environment": environment()})
            save_json_baseline(IBSI1_VALUES, values)
            print_success("IBSI 1 baseline, values and cache updated.")
        if write_docs or update:
            (DOCS_DIR / "ibsi1_compliance.md").write_text(test.generate_report())

        return result, current_results, test

    except Exception as e:
        return _crashed("IBSI 1", e)


def run_ibsi2_compliance(
    update_baseline: bool = False,
    accept_changes: bool = False,
    allow_missing_data: bool = False,
    write_docs: bool = False,
) -> tuple[ComplianceResult, dict[str, Any], Any]:
    """Run IBSI 2 Phase 1 compliance verification with baseline comparison."""
    print_step("Running IBSI 2 Phase 1 compliance verification...")
    missing = _missing_data("IBSI 2 Phase 1", allow_missing_data)
    if missing is not None:
        return missing, {}, None

    try:
        from dev.IBSI2.verify_ibsi2_compliance import IBSIPhase1Validator

        # Run validation
        validator = IBSIPhase1Validator(IBSI2_DATA_DIR)
        validator.run_all_tests(verbose=False)

        # Get structured results
        current_results = validator.get_results_dict()
        current_results["timestamp"] = datetime.now().isoformat()

        # Add a total time metric to the top level for standard reporting
        if "total_time_seconds" not in current_results:
            total_ms = sum(t.get("time_ms", 0) for t in current_results.get("tests", {}).values())
            current_results["total_time_seconds"] = total_ms / 1000.0

        # Load baseline
        baseline = load_json_baseline(IBSI2_BASELINE)

        # Standard Report
        maps = validator.maps()
        stats: dict[str, float] = {}
        result = print_compliance_report(
            "IBSI 2 Phase 1",
            current_results,
            baseline,
            key_label="Test ID",
            value_changes=compare_maps(maps, IBSI2_PHASE1_MAPS, stats=stats),
            value_stats=stats,
            other=other_environment(baseline),
        )
        margin = phase1_margin_problems(current_results)
        if margin:
            print_error(f"IBSI 2 Phase 1: {len(margin)} maps pass with an error above the margin:")
            for problem in margin:
                print(f"    {problem}")
            result.passed = False
            result.new_failures += len(margin)

        # Add note about skipped tests
        skipped = current_results.get("skipped", 0)
        if skipped > 0:
            print()
            print(
                f"{Colors.CYAN}Note: {skipped} tests skipped (2D filters); "
                f"{current_results.get('missing', 0)} without a reference map{Colors.ENDC}"
            )
            print(
                f"{Colors.CYAN}      Pictologics implements 3D filters only. See docs for details.{Colors.ENDC}"
            )

        update = update_baseline and _may_update(result, accept_changes)
        if update:
            validator.save_cache(IBSI2_PHASE1_CACHE)
            save_json_baseline(IBSI2_BASELINE, {**current_results, "environment": environment()})
            save_npz_baseline(IBSI2_PHASE1_MAPS, maps)
            print_success("IBSI 2 Phase 1 baseline, maps and cache updated.")
        if write_docs or update:
            validator.generate_docs(DOCS_DIR / "ibsi2_compliance.md")

        return result, current_results, validator

    except Exception as e:
        return _crashed("IBSI 2 Phase 1", e)


def run_ibsi2_phase2_compliance(
    update_baseline: bool = False,
    accept_changes: bool = False,
    allow_missing_data: bool = False,
    write_docs: bool = False,
) -> tuple[ComplianceResult, dict[str, Any], Any]:
    """Run IBSI 2 Phase 2 compliance verification (features from filtered response maps)."""
    print_step("Running IBSI 2 Phase 2 compliance verification...")
    missing = _missing_data("IBSI 2 Phase 2", allow_missing_data)
    if missing is not None:
        return missing, {}, None

    try:
        from dev.IBSI2.verify_ibsi2_phase2 import IBSIPhase2Validator

        validator = IBSIPhase2Validator(IBSI2_DATA_DIR)
        validator.run_all_tests(verbose=False)

        current_results = validator.get_summary()
        current_results["timestamp"] = datetime.now().isoformat()

        if "total_time_seconds" not in current_results:
            total_ms = sum(t.get("time_ms", 0) for t in current_results.get("tests", {}).values())
            current_results["total_time_seconds"] = total_ms / 1000.0

        values = validator.values()
        baseline_values = (
            load_json_baseline(IBSI2_PHASE2_VALUES) if IBSI2_PHASE2_VALUES.exists() else None
        )
        baseline = load_json_baseline(IBSI2_PHASE2_BASELINE)
        stats: dict[str, float] = {}
        result = print_compliance_report(
            "IBSI 2 Phase 2",
            current_results,
            baseline,
            key_label="Filter ID",
            value_changes=None
            if baseline_values is None
            else compare_values(values, baseline_values, stats=stats),
            value_stats=stats,
            other=other_environment(baseline),
        )

        update = update_baseline and _may_update(result, accept_changes)
        if update:
            validator.save_cache(IBSI2_PHASE2_CACHE)
            save_json_baseline(IBSI2_PHASE2_BASELINE, {**current_results, "environment": environment()})
            save_json_baseline(IBSI2_PHASE2_VALUES, values)
            print_success("IBSI 2 Phase 2 baseline, values and cache updated.")
        if write_docs or update:
            validator.generate_docs(DOCS_DIR / "ibsi2_phase2_compliance.md")

        return result, current_results, validator

    except Exception as e:
        return _crashed("IBSI 2 Phase 2", e)


def run_ibsi2_phase3_compliance(
    update_baseline: bool = False,
    accept_changes: bool = False,
    allow_missing_data: bool = False,
    write_docs: bool = False,
) -> tuple[ComplianceResult, dict[str, Any], Any]:
    """Run IBSI 2 Phase 3 compliance verification."""
    print_step("Running IBSI 2 Phase 3 compliance verification...")
    missing = _missing_data("IBSI 2 Phase 3", allow_missing_data)
    if missing is not None:
        return missing, {}, None

    try:
        from dev.IBSI2.verify_ibsi2_phase3 import IBSIPhase3Validator

        validator = IBSIPhase3Validator(IBSI2_DATA_DIR, n_workers=PHASE3_WORKERS)
        validator.run_all_patients(verbose=False)
        validator.compare_with_teams()

        # Get structured results
        current_results = validator.get_summary()
        values = validator.values()
        baseline_values = (
            load_json_baseline(IBSI2_PHASE3_VALUES) if IBSI2_PHASE3_VALUES.exists() else None
        )

        # Load baseline
        baseline = load_json_baseline(IBSI2_PHASE3_BASELINE)

        # Phase 3 Reproducibility Report (Custom)
        stats: dict[str, float] = {}
        result = print_phase3_summary(
            current_results,
            baseline,
            value_changes=None
            if baseline_values is None
            else compare_values(values, baseline_values, stats=stats),
            value_stats=stats,
            other=other_environment(baseline),
        )

        update = update_baseline and _may_update(result, accept_changes)
        if update:
            validator.save_cache(IBSI2_PHASE3_CACHE)
            save_json_baseline(IBSI2_PHASE3_BASELINE, {**current_results, "environment": environment()})
            save_json_baseline(IBSI2_PHASE3_VALUES, values)
            print_success("IBSI 2 Phase 3 baseline, values and cache updated.")
        if write_docs or update:
            validator.generate_docs(DOCS_DIR / "ibsi2_phase3_compliance.md")

        return result, current_results, validator

    except Exception as e:
        return _crashed("IBSI 2 Phase 3", e)


def run_benchmarks(
    update_baseline: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """
    Run performance benchmarks and compare to baseline.

    Returns: (success, results)
    """
    print_step("Running performance benchmarks...")

    try:
        from dev.run_benchmark import (
            run_performance_benchmarks,
        )

        # Run benchmarks
        df, voxel_stats = run_performance_benchmarks(
            output_plot_dir=str(RESULTS_DIR / "plots"),
            suppress_external_output=True,
        )

        # Extract timing metrics per family (averaged across all sizes and configs)
        current_metrics: dict[str, dict[str, float]] = {}

        for family in ["Intensity", "Morphology", "Texture"]:
            family_data = df[(df["Family"] == family) & (df["Group"] == "Parity")]
            if not family_data.empty:
                picto_data = family_data[family_data["Library"] == "Pictologics"]
                if not picto_data.empty:
                    current_metrics[family] = {
                        "time_ms": float(picto_data["Time_Mean"].mean()) * 1000,
                        "memory_mb": float(picto_data["Memory_Peak"].mean()),
                    }

        current_results = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "metrics": current_metrics,
            "voxel_stats": voxel_stats,
        }

        # Compare to baseline
        baseline = load_json_baseline(BENCHMARK_BASELINE)

        if baseline and "metrics" in baseline:
            print("\n" + "-" * 50)
            print("Performance Comparison (vs baseline):")
            print("-" * 50)
            print(f"{'Family':<15} {'Baseline (ms)':<15} {'Current (ms)':<15} {'Change':<15}")
            print("-" * 50)

            for family, metrics in current_metrics.items():
                baseline_time = baseline["metrics"].get(family, {}).get("time_ms", 0)
                current_time = metrics["time_ms"]

                if baseline_time > 0:
                    change = (current_time - baseline_time) / baseline_time
                    change_str = f"{change:+.1%}"

                    if change > REGRESSION_THRESHOLD:
                        change_str = f"{Colors.FAIL}{change_str} REGRESSION{Colors.ENDC}"
                    elif change < -0.05:
                        change_str = f"{Colors.GREEN}{change_str} IMPROVED{Colors.ENDC}"
                else:
                    change_str = "N/A (new)"

                print(f"{family:<15} {baseline_time:<15.2f} {current_time:<15.2f} {change_str}")

            print("-" * 50)

        # Only save cache and update baseline when explicitly requested
        if update_baseline:
            save_json_baseline(BENCHMARK_CACHE, current_results)
            save_json_baseline(BENCHMARK_BASELINE, current_results)
            print_success("Benchmark baseline and cache updated")

        print_success("Benchmarks completed")
        return True, current_results

    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return False, {}


def print_summary(results: dict[str, bool]) -> None:
    """Print final summary and next steps."""
    print_header("Summary")

    all_passed = all(results.values())

    for check, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"  {check:<30} [{status}]")

    print()

    if all_passed:
        print_success("All checks passed!")
        print()
        print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print()
        print("  1. Commit the files by name, then push:")
        print(f"     {Colors.CYAN}git add <files> && git commit && git push{Colors.ENDC}")
        print()
        print("  2. Publish with a GitHub release: a pre-release goes to TestPyPI, a release")
        print("     to PyPI. The publish workflows run the tests and check the version first.")
    else:
        print_error("Some checks failed. Please fix the issues above before pushing.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-push workflow for Pictologics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip performance benchmarks",
    )
    parser.add_argument(
        "--skip-compliance",
        action="store_true",
        help="Skip IBSI compliance checks",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "Save the IBSI results, values and caches as the new baselines (the old files are "
            "copied first), and write the IBSI docs pages; refused after failures"
        ),
    )
    parser.add_argument(
        "--accept-changes",
        action="store_true",
        help="With --update-baseline: save the new baselines even though the checks failed",
    )
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Skip an IBSI check whose data or references are missing (default: it fails)",
    )
    parser.add_argument(
        "--skip-cache-clear",
        action="store_true",
        help="Skip clearing cache directories",
    )
    parser.add_argument(
        "--skip-towncrier",
        action="store_true",
        help="Skip towncrier changelog fragment check",
    )
    parser.add_argument(
        "--compliance-only",
        action="store_true",
        help=(
            "Build the numba cache and run the IBSI checks only (as in CI, which runs ruff, mypy "
            "and pytest itself): no cache clear, lint, types, towncrier, pytest or benchmarks"
        ),
    )
    return parser.parse_args()


def run_with_log() -> int:
    """Run the gate in a child process and copy its output to the terminal and to a log file
    per run in GATE_LOG_DIR (outside OneDrive)."""
    GATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = GATE_LOG_DIR / f"gate-{datetime.now():%Y%m%d-%H%M%S}.log"
    env = {**os.environ, "PICTOLOGICS_GATE_LOGGED": "1", "PYTHONUNBUFFERED": "1"}
    with (
        open(log, "w") as f,
        subprocess.Popen(
            [sys.executable, __file__, *sys.argv[1:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        ) as child,
    ):
        assert child.stdout is not None
        for line in child.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
    print(f"Log: {log}")
    return child.returncode


def main() -> int:
    """Main entry point."""
    args = parse_args()
    redirect_caches(clean=not args.skip_cache_clear)

    print_header("Pictologics Pre-Push Workflow")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results: dict[str, bool] = {}
    tools = not args.compliance_only
    compliance_utils.SHOW_TIMES = tools

    # 1. Clear caches, then compile the numba cache while the tool steps run
    if tools and not args.skip_cache_clear:
        results["Cache Clear"] = clear_caches()
    numba_build = start_numba_build()

    # 2. Run ruff
    if tools:
        results["Ruff Linting"] = run_ruff()

    # 3. Run mypy
    if tools:
        results["Mypy Type Check"] = run_mypy()

    # 4. Check towncrier fragments
    if tools and not args.skip_towncrier:
        results["Towncrier Check"] = run_towncrier_check()
    elif tools:
        print_warning("Towncrier check skipped")

    # 5. Run pytest
    if tools:
        results["Pytest"] = run_pytest()[0]
    results["Numba Build"] = finish_numba_build(numba_build)

    # 6. Run IBSI compliance
    ibsi_results: list[ComplianceResult] = []
    if not args.skip_compliance:
        for label, run in (
            ("IBSI 1 Compliance", run_ibsi_compliance),
            ("IBSI 2 Phase 1", run_ibsi2_compliance),
            ("IBSI 2 Phase 2", run_ibsi2_phase2_compliance),
            ("IBSI 2 Phase 3", run_ibsi2_phase3_compliance),
        ):
            result, _, _ = run(
                update_baseline=args.update_baseline,
                accept_changes=args.accept_changes,
                allow_missing_data=args.allow_missing_data,
            )
            ibsi_results.append(result)
            results[label] = result.passed

        # Print aggregated IBSI summary
        print_ibsi_summary(ibsi_results)
    else:
        print_warning("IBSI compliance checks skipped")

    # 7. Run benchmarks
    if tools and not args.skip_benchmarks:
        bench_passed, _ = run_benchmarks(update_baseline=args.update_baseline)
        results["Benchmarks"] = bench_passed
    elif tools:
        print_warning("Benchmarks skipped")

    # Print summary
    print_summary(results)

    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main() if os.environ.get("PICTOLOGICS_GATE_LOGGED") else run_with_log())
