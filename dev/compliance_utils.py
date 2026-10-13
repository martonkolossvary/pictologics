"""
Shared utilities for compliance reporting and console output.
Used by pre_push.py and generate_docs.py to ensure consistent behavior.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dev.pipeline_config import (
    GATE_CACHE_DIR,
    PHASE1_ERROR_MARGIN_PCT,
    PHASE3_ALLOWED_DROP,
    REGRESSION_THRESHOLD,
)

# The process pools of the IBSI checks start their workers with spawn on every system, as on
# a Mac. On Linux, a forked worker of a process that ran numba kernels in threads stops.
WORKERS = multiprocessing.get_context("spawn")


def redirect_caches(clean: bool) -> None:
    """Keep the tool caches outside OneDrive, in GATE_CACHE_DIR. With clean=True they start
    empty, so the run proves that everything builds from the source. Call it before numba is
    imported; child processes inherit the settings."""
    if clean:
        shutil.rmtree(GATE_CACHE_DIR, ignore_errors=True)
    GATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, sub in (
        ("NUMBA_CACHE_DIR", "numba"),
        ("PYTHONPYCACHEPREFIX", "pycache"),
        ("MYPY_CACHE_DIR", "mypy"),
        ("RUFF_CACHE_DIR", "ruff"),
        ("COVERAGE_FILE", ".coverage"),
    ):
        os.environ[name] = str(GATE_CACHE_DIR / sub)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]


def get_rounding_decimals(value_str: str) -> int:
    """Get number of decimals from string representation."""
    value_str = str(value_str).lower()
    if "e" in value_str:
        parts = value_str.split("e")
        mantissa = parts[0]
        exponent = int(parts[1])
        mantissa_decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return mantissa_decimals - exponent

    if "." in value_str:
        return len(value_str.split(".")[1])
    return 0


def environment() -> dict[str, str]:
    """The platform and the numeric packages that our IBSI values depend on. A baseline
    saves it, so the gate knows whether a value change can come from our code alone."""
    import platform

    import llvmlite
    import numba
    import numpy
    import scipy

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
    }


def other_environment(baseline: dict[str, Any]) -> str:
    """An empty text when the baseline comes from this platform and these packages; else
    what differs.
    In another environment, moved values are a report for review, not a failure: the IBSI
    pass counts are the requirement, and other packages round otherwise."""
    saved = baseline.get("environment")
    if not saved:
        return "the baseline has no environment record"
    now = environment()
    differ = [f"{key} {saved.get(key, '?')} -> {now[key]}" for key in now if saved.get(key) != now[key]]
    return "; ".join(differ)


# IBSI 2 phases 2 and 3: Pictologics feature name -> IBSI tag
FEATURE_MAP = {
    "mean_intensity_Q4LE": "stat_mean",
    "intensity_variance_ECT3": "stat_var",
    "intensity_skewness_KE2A": "stat_skew",
    "intensity_kurtosis_IPH6": "stat_kurt",
    "median_intensity_Y12H": "stat_median",
    "minimum_intensity_1GSF": "stat_min",
    "10th_intensity_percentile_QG58": "stat_p10",
    "90th_intensity_percentile_8DWT": "stat_p90",
    "maximum_intensity_84IY": "stat_max",
    "intensity_interquartile_range_SALO": "stat_iqr",
    "intensity_range_2OJQ": "stat_range",
    "intensity_mean_absolute_deviation_4FUA": "stat_mad",
    "intensity_robust_mean_absolute_deviation_1128": "stat_rmad",
    "intensity_median_absolute_deviation_N72L": "stat_medad",
    "intensity_coefficient_of_variation_7TET": "stat_cov",
    "intensity_quartile_coefficient_of_dispersion_9S40": "stat_qcod",
    "intensity_energy_N8CA": "stat_energy",
    "root_mean_square_intensity_5ZWQ": "stat_rms",
}

# IBSI 2 phase 3: the participating teams
PHASE3_TEAMS = [
    "CERR",
    "Cardiff University",
    "King's College London",
    "NCT Dresden",
    "Qurit SERA",
    "UCSF",
    "USZ",
    "UdeS",
    "Veneto Institute of Oncology",
]

# Items that are not expected to pass. Any other item that does not pass fails the gate,
# whatever the baseline says. IBSI 1: IBSI publishes no reference value (area under the IVH
# curve, and the bounding-box and enclosing-ellipsoid morphology features). IBSI 2 phase 1:
# 2D filters (SKIP), and tests without a published reference map (MISSING).
_IBSI1_NO_REFERENCE = {
    "Digital_Phantom": ("9CMM", "BRI8", "IQYR", "SWZ1", "ZH1A"),
    "Config_C": ("9CMM", "BRI8", "IQYR", "SKGS", "SWZ1", "ZH1A"),
    "Config_D": ("9CMM", "BRI8", "IQYR", "SWZ1", "ZH1A"),
    "Config_E": ("9CMM", "BRI8", "IQYR", "SWZ1", "ZH1A"),
}
KNOWN_GAPS: dict[str, dict[str, str]] = {
    "IBSI 1": {
        f"{config}:{'Intensity-Volume Histogram' if code == '9CMM' else 'Morphology'}:{code}": "SKIP"
        for config, codes in _IBSI1_NO_REFERENCE.items()
        for code in codes
    },
    "IBSI 2 Phase 1": {
        **dict.fromkeys(("1.b.1", "2.c", "3.c.1", "3.c.2", "3.c.3"), "SKIP"),
        **dict.fromkeys(("9.b.2", "10.a", "10.b.2"), "MISSING"),
    },
    "IBSI 2 Phase 2": {},
}


# Color codes
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


@dataclass
class ComplianceResult:
    """Structured result from a compliance check for aggregation."""

    name: str
    passed: bool
    total: int = 0
    passed_count: int = 0
    new_failures: int = 0
    missing: int = 0
    known_failures: int = 0
    status_improvements: int = 0
    time_seconds: float = 0.0
    baseline_time_seconds: float = 0.0
    memory_mb: float = 0.0
    baseline_memory_mb: float = 0.0
    perf_regressions: int = 0
    perf_improvements: int = 0
    mem_regressions: int = 0
    mem_improvements: int = 0
    known_gaps: int = 0
    value_changes: int = 0


# Platform agnostic clear methods
def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_step(text: str) -> None:
    """Print a step indicator."""
    print(f"{Colors.CYAN}→ {text}{Colors.ENDC}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def run_command(
    cmd: list[str], cwd: Path | str | None = None, capture: bool = False
) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=True,
    )
    stdout = result.stdout if capture else ""
    stderr = result.stderr if capture else ""
    return result.returncode, stdout, stderr


def load_json_baseline(path: Path | str) -> dict[str, Any]:
    """Load a JSON baseline file."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def backup_baseline(path: Path | str) -> None:
    """Copy an existing baseline to a dated folder outside the project first: dev/ has no git
    history."""
    from dev.pipeline_config import BASELINE_BACKUP_DIR

    p = Path(path)
    if p.exists():
        folder = BASELINE_BACKUP_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, folder / p.name)


def save_json_baseline(path: Path | str, data: dict[str, Any]) -> None:
    """Save a JSON baseline file: a copy of the old file first, then an atomic write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    backup_baseline(p)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, p)


def _count(stats: dict[str, float] | None, n: int, same: int, rel: float) -> None:
    """Add compared values to `stats`: how many, how many are the same, the largest relative
    difference."""
    if stats is not None:
        stats["n"] = stats.get("n", 0) + n
        stats["same"] = stats.get("same", 0) + same
        stats["max_rel"] = max(stats.get("max_rel", 0.0), rel)


def compare_values(
    current: Any,
    baseline: Any,
    rel: float = 1e-9,
    path: str = "",
    stats: dict[str, float] | None = None,
) -> list[str]:
    """Differences between two nested dicts of numbers: values that differ by more than `rel`
    times the larger magnitude, and keys that are lost or new. NaN equals NaN. `stats`, if
    given, counts the compared values (see _count)."""
    if isinstance(current, dict) or isinstance(baseline, dict):
        if not (isinstance(current, dict) and isinstance(baseline, dict)):
            return [f"{path}: {baseline!r} -> {current!r}"]
        changes = []
        for key in sorted(baseline.keys() | current.keys(), key=str):
            sub = f"{path}/{key}" if path else str(key)
            if key not in current:
                changes.append(f"{sub}: lost")
            elif key not in baseline:
                changes.append(f"{sub}: new")
            else:
                changes.extend(compare_values(current[key], baseline[key], rel, sub, stats))
        return changes
    a, b = float(current), float(baseline)
    if a == b or (math.isnan(a) and math.isnan(b)):
        _count(stats, 1, 1, 0.0)
        return []
    difference = abs(a - b) / max(abs(a), abs(b))
    _count(stats, 1, 0, difference)
    if difference <= rel:
        return []
    return [f"{path}: {b!r} -> {a!r}"]


def compare_maps(
    current: dict[str, Any], path: Path | str, rel: float = 1e-9, stats: dict[str, float] | None = None
) -> list[str] | None:
    """Differences between response maps and the saved ones (None: no saved maps). A map
    changes when a value moves by more than `rel` times the largest magnitude of the saved map,
    or when its shape or its NaN voxels change."""
    import numpy as np

    p = Path(path)
    if not p.exists():
        return None
    with np.load(p) as saved:
        baseline = {key: saved[key] for key in saved.files}
    changes = []
    for key in sorted(baseline.keys() | current.keys()):
        if key not in current:
            changes.append(f"{key}: lost")
        elif key not in baseline:
            changes.append(f"{key}: new")
        elif current[key].shape != baseline[key].shape:
            changes.append(f"{key}: shape {baseline[key].shape} -> {current[key].shape}")
        elif not np.array_equal(np.isnan(current[key]), np.isnan(baseline[key])):
            changes.append(f"{key}: NaN voxels changed")
        elif baseline[key].size:
            scale = float(np.nanmax(np.abs(baseline[key])))
            diff = float(np.nanmax(np.abs(current[key] - baseline[key])))
            same = np.count_nonzero((current[key] == baseline[key]) | np.isnan(baseline[key]))
            _count(stats, baseline[key].size, int(same), diff / scale if scale else 0.0)
            if diff > rel * scale:
                changes.append(f"{key}: largest difference {diff:.3g} (map scale {scale:.3g})")
    return changes


def save_npz_baseline(path: Path | str, arrays: dict[str, Any]) -> None:
    """Save maps: a copy of the old file first, then an atomic write."""
    import numpy as np

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    backup_baseline(p)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)


def print_value_report(
    title: str,
    changes: list[str] | None,
    stats: dict[str, float] | None = None,
    other: str = "",
) -> int:
    """Print the value check of a compliance step; return the number of problems. None means
    that the value baseline is missing. With `other` (see other_environment), the baseline
    comes from another platform or other packages: the changes are listed for review and
    are no problem."""
    print()
    if changes is None:
        if other:
            print(f"{Colors.WARNING}{title}: no value baseline here ({other}).{Colors.ENDC}")
            return 0
        print(f"{Colors.FAIL}{title}: no value baseline. Create it with --update-baseline.{Colors.ENDC}")
        return 1
    if stats:
        print(
            f"{title} values: {int(stats['same']):,} of {int(stats['n']):,} the same; "
            f"largest relative difference {stats['max_rel']:.1e}"
        )
    if not changes:
        print(f"{title} values: {Colors.GREEN}unchanged (relative difference <= 1e-9){Colors.ENDC}")
        return 0
    if other:
        print(
            f"{Colors.WARNING}{title} values: {len(changes)} moved (relative difference > 1e-9), "
            f"for review: the baseline comes from another environment ({other}).{Colors.ENDC}"
        )
    else:
        print(f"{Colors.FAIL}{title} values: {len(changes)} changed (relative difference > 1e-9):{Colors.ENDC}")
    for change in changes[:20]:
        print(f"    {change}")
    if len(changes) > 20:
        print(f"    ... and {len(changes) - 20} more")
    return 0 if other else len(changes)


def phase1_margin_problems(current_results: dict[str, Any]) -> list[str]:
    """The passing IBSI 2 phase 1 maps whose error is above PHASE1_ERROR_MARGIN_PCT (half the
    IBSI limit of 1 % of the reference range): a drift shows before a map fails IBSI."""
    return [
        f"{key}: error {test.get('error_pct', 0.0):.3f} % (margin {PHASE1_ERROR_MARGIN_PCT} %)"
        for key, test in sorted(current_results.get("tests", {}).items())
        if test.get("status") == "PASS" and test.get("error_pct", 0.0) > PHASE1_ERROR_MARGIN_PCT
    ]


def print_compliance_report(
    title: str,
    current_results: dict[str, Any],
    baseline_results: dict[str, Any],
    key_label: str = "Test ID",
    perf_threshold: float = REGRESSION_THRESHOLD,
    value_changes: list[str] | None = None,
    value_stats: dict[str, float] | None = None,
    other: str = "",
) -> ComplianceResult:
    """
    Print a standardized compliance report comparing current results to baseline.

    Args:
        title: Report title (e.g., "IBSI 2 Phase 1")
        current_results: Current run results dict
        baseline_results: Baseline results dict
        key_label: Label for the item identifier (e.g., "Test ID", "Feature Code")
        perf_threshold: Threshold for performance regression reporting (default 0.20)
        value_changes: Result of compare_values for this step (None: no value baseline).
        other: What differs from the environment of the baseline (see other_environment):
            then value changes are for review only.

    An item fails when it does not pass and is not a known gap (KNOWN_GAPS), and when a
    baseline item is missing from the current run.

    Returns:
        ComplianceResult: Structured result object.
    """
    print()
    print("-" * 60)
    print(f"{title} Comparison (vs Baseline)")
    print("-" * 60)

    # 1. Status Analysis
    # ------------------
    # Flatten items for comparison: {key: {"status": "PASS"|"FAIL", "time": float}}
    current_items = {}
    baseline_items = {}

    # Helper to extract items
    def extract_items(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
        items = {}
        # Try different known structures
        if "features" in source and isinstance(source["features"], dict):
            # IBSI 1: features dict with explicit status string or dict
            for k, v in source["features"].items():
                if isinstance(v, dict):
                    items[k] = {
                        "status": v.get("status", "UNKNOWN"),
                        "time": v.get("time_seconds", 0),
                        "memory": v.get("memory_mb", 0),
                    }
                else:
                    items[k] = {
                        "status": v,
                        "time": 0,
                        "memory": 0,
                    }  # Legacy
        elif "tests" in source:
            # IBSI 2: tests dict
            for k, v in source["tests"].items():
                # Handle time units: standardize to seconds
                t = v.get("time_seconds", v.get("time_ms", 0) / 1000.0)
                items[k] = {
                    "status": v.get("status", "UNKNOWN"),
                    "time": t,
                    "memory": v.get("memory_mb", 0),
                }
        elif "team_results" in source:
            # IBSI 2 Phase 3: team comparisons (special case, mostly for perf)
            pass
        return items

    current_items = extract_items(current_results)
    baseline_items = extract_items(baseline_results)

    passed = current_results.get("passed", 0)
    failed = current_results.get("failed", 0)
    skipped = current_results.get("skipped", 0)

    # Recalculate if not provided (e.g., IBSI 1 structure)
    if "passed" not in current_results:
        passed = sum(1 for v in current_items.values() if v["status"] == "PASS")
        failed = sum(1 for v in current_items.values() if v["status"] == "FAIL")
        skipped = sum(1 for v in current_items.values() if v["status"] == "SKIP")
        missing_count = sum(
            1 for v in current_items.values() if v["status"] in ("MISSING", "REF")
        )
    else:
        missing_count = current_results.get("missing", 0)

    total = passed + failed + skipped + missing_count  # Approximate total denominator

    new_failures = []
    known_failures = []
    status_improvements = []
    perf_improvements = []
    perf_regressions = []
    mem_regressions = []
    mem_improvements = []

    # Compare Status: every item passes or is a known gap; no baseline item may disappear
    known_gaps = KNOWN_GAPS.get(title, {})
    for key, cur in current_items.items():
        base = baseline_items.get(key, {})
        cur_status = cur["status"]
        base_status = base.get("status", "N/A")

        if cur_status == "PASS":
            if base_status not in ("PASS", "N/A"):
                status_improvements.append(key)
        elif known_gaps.get(key) == cur_status:
            known_failures.append(key)
        else:
            new_failures.append({"key": key, "base": base_status, "cur": cur_status})
    for key in sorted(baseline_items.keys() - current_items.keys()):
        new_failures.append({"key": key, "base": baseline_items[key]["status"], "cur": "LOST"})

    # Compare Performance & Memory
    cur_total_time = current_results.get(
        "total_time_seconds", sum(i["time"] for i in current_items.values())
    )
    base_total_time = baseline_results.get(
        "total_time_seconds", sum(i["time"] for i in baseline_items.values())
    )

    # Calculate total memory
    cur_total_mem = current_results.get(
        "memory_mb", sum(i.get("memory", 0) for i in current_items.values())
    )

    for key, cur in current_items.items():
        base = baseline_items.get(key)
        if base and base["time"] > 0 and cur["time"] > 0:
            change = (cur["time"] - base["time"]) / base["time"]
            entry = {
                "key": key,
                "old": base["time"],
                "new": cur["time"],
                "change": change,
            }
            if change > perf_threshold:  # Slower
                perf_regressions.append(entry)
            elif change < -perf_threshold:  # Faster (improvement)
                perf_improvements.append(entry)

            # Compare Memory
            if base.get("memory", 0) > 0 and cur.get("memory", 0) > 0:
                mem_change = (cur["memory"] - base["memory"]) / base["memory"]
                mem_entry = {
                    "key": key,
                    "old": base["memory"],
                    "new": cur["memory"],
                    "change": mem_change,
                }
                if mem_change > perf_threshold:
                    mem_regressions.append(mem_entry)
                elif mem_change < -perf_threshold:
                    mem_improvements.append(mem_entry)

    # 2. Print Summary
    # ----------------
    print("Summary:")
    pass_pct = (passed / total * 100) if total > 0 else 0
    print(f"  Passed: {passed}/{total} ({pass_pct:.1f}%)")
    if skipped:
        print(f"  Skipped: {skipped}")
    if missing_count:
        print(f"  Missing References: {missing_count}")
    print(f"  New Failures: {len(new_failures)}")
    print(f"  Known Gaps: {len(known_failures)}")
    print(f"  Status Improvements: {len(status_improvements)}")

    # 3. Print Performance
    # --------------------
    print()
    print("Performance & Memory:")

    time_diff = cur_total_time - base_total_time
    time_change_pct = (
        (time_diff / base_total_time * 100) if base_total_time > 0 else 0.0
    )
    color = (
        Colors.FAIL
        if time_change_pct > 10
        else (Colors.GREEN if time_change_pct < -10 else Colors.ENDC)
    )

    print(
        f"  Time (sum of the timed runs): {cur_total_time:.2f}s  (vs {base_total_time:.2f}s, {color}{time_change_pct:+.1f}%{Colors.ENDC})"
    )

    avg_msg = ""
    # Only show avg if item count is reasonable approximation of work units
    if len(current_items) > 0:
        avg_time = cur_total_time / len(current_items) * 1000
        avg_msg = f" (~{avg_time:.1f}ms/check)"

    elif total > 0:
        # Fallback if parsing failed but we have counts
        avg_time = cur_total_time / total * 1000
        avg_msg = f" (~{avg_time:.1f}ms/check)"

    print(f"  Throughput: {len(current_items) or total} checks{avg_msg}")

    # Memory Reporting
    if cur_total_mem > 0:
        print(f"  Memory Usage: {cur_total_mem:.1f} MB")

    # Time Regressions
    print()
    print(f"Time Regressions (> {perf_threshold*100:.0f}% slower):")
    if perf_regressions:
        print(f"    {'Test ID':<40} {'Old (ms)':<10} {'New (ms)':<10} {'Change':<10}")
        print(f"    {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        for r in sorted(perf_regressions, key=lambda x: x["change"], reverse=True):
            print(
                f"    {r['key']:<40} {r['old']*1000:<10.1f} {r['new']*1000:<10.1f} {Colors.FAIL}{r['change']:+.0%}{Colors.ENDC}"
            )
    else:
        print(f"  {Colors.GREEN}None.{Colors.ENDC}")

    # Time Improvements
    print()
    print(f"Significant Time Improvements (> {perf_threshold*100:.0f}% faster):")
    if perf_improvements:
        print(f"    {'Test ID':<40} {'Old (ms)':<10} {'New (ms)':<10} {'Change':<10}")
        print(f"    {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        for r in sorted(perf_improvements, key=lambda x: x["change"]):
            print(
                f"    {r['key']:<40} {r['old']*1000:<10.1f} {r['new']*1000:<10.1f} {Colors.GREEN}{r['change']:+.0%}{Colors.ENDC}"
            )
    else:
        print(f"  {Colors.GREEN}None.{Colors.ENDC}")

    # Memory Regressions
    print()
    print(f"Memory Regressions (> {perf_threshold*100:.0f}% increase):")
    if mem_regressions:
        print(f"    {'Test ID':<40} {'Old (MB)':<10} {'New (MB)':<10} {'Change':<10}")
        print(f"    {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        for r in sorted(mem_regressions, key=lambda x: x["change"], reverse=True):
            print(
                f"    {r['key']:<40} {r['old']:<10.1f} {r['new']:<10.1f} {Colors.FAIL}{r['change']:+.0%}{Colors.ENDC}"
            )
    else:
        print(f"  {Colors.GREEN}None.{Colors.ENDC}")

    # Memory Improvements
    print()
    print(f"Significant Memory Improvements (> {perf_threshold*100:.0f}% decrease):")
    if mem_improvements:
        print(f"    {'Test ID':<40} {'Old (MB)':<10} {'New (MB)':<10} {'Change':<10}")
        print(f"    {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        for r in sorted(mem_improvements, key=lambda x: x["change"]):
            print(
                f"    {r['key']:<40} {r['old']:<10.1f} {r['new']:<10.1f} {Colors.GREEN}{r['change']:+.0%}{Colors.ENDC}"
            )
    else:
        print(f"  {Colors.GREEN}None.{Colors.ENDC}")

    # 4. Print Feature Value Details
    # ------------------------------
    if new_failures:
        print()
        print(
            f"{Colors.FAIL}NEW FAILURES (REGRESSIONS) - ACTION REQUIRED:{Colors.ENDC}"
        )
        print(f"  {key_label:<50} {'Baseline':<10} {'Current':<10}")
        print(f"  {'-'*50} {'-'*10} {'-'*10}")
        for item in new_failures:
            print(
                f"  {item['key']:<50} {item['base']:<10} {Colors.FAIL}{item['cur']}{Colors.ENDC}"
            )

    if known_failures:
        print()
        print(f"{Colors.CYAN}KNOWN GAPS (expected, see KNOWN_GAPS): {len(known_failures)} items{Colors.ENDC}")

    if status_improvements:
        print()
        print(f"{Colors.GREEN}STATUS IMPROVEMENTS (FAIL -> PASS):{Colors.ENDC}")
        print(f"  {key_label:<50}")
        print(f"  {'-'*50}")
        for key in status_improvements:
            print(f"  {Colors.GREEN}{key}{Colors.ENDC}")

    value_problems = print_value_report(title, value_changes, value_stats, other)
    print("-" * 60)

    # Return structured result for aggregation
    return ComplianceResult(
        name=title,
        passed=len(new_failures) == 0 and value_problems == 0,
        total=total,
        passed_count=passed,
        new_failures=len(new_failures) + value_problems,
        missing=missing_count,
        known_failures=0,
        known_gaps=len(known_failures),
        value_changes=value_problems,
        status_improvements=len(status_improvements),
        time_seconds=cur_total_time,
        baseline_time_seconds=base_total_time,
        memory_mb=cur_total_mem,
        baseline_memory_mb=baseline_results.get("memory_mb", 0.0),
        perf_regressions=len(perf_regressions),
        perf_improvements=len(perf_improvements),
        mem_regressions=len(mem_regressions),
        mem_improvements=len(mem_improvements),
    )


def print_ibsi_summary(results: list[ComplianceResult]) -> bool:
    """
    Print aggregated IBSI compliance summary table.

    Returns:
        bool: True if all checks passed (only known failures), False if any new failures.
    """
    if not results:
        return True

    print_header("IBSI Compliance Summary")

    # Compliance Status Table
    print(
        f"{'Check':<22} {'Status':<12} {'Passed':>12} {'New Fails':>12} {'Known Gaps':>12} {'Improved':>12}"
    )
    print("-" * 90)

    all_passed = True
    for r in results:
        # Determine status: a step that crashed or found a problem fails
        if not r.passed or r.new_failures > 0:
            status_str = "FAIL"
            status_color = Colors.FAIL
            all_passed = False
        else:
            status_str = "PASS"
            status_color = Colors.GREEN

        passed_str = f"{r.passed_count}/{r.total}"
        print(
            f"{r.name:<22} {status_color}{status_str:<12}{Colors.ENDC} {passed_str:>12} "
            f"{r.new_failures:>12} {r.known_gaps:>12} {r.status_improvements:>12}"
        )

    print("-" * 90)

    # Performance Summary Table
    print()
    print("Performance Summary:")
    print(
        f"{'Check':<22} {'Time (s)':>12} {'Base (s)':>12} {'Time Δ':>12} {'Mem (MB)':>12} {'Mem Δ':>12}"
    )
    print("-" * 90)

    for r in results:
        # Calculate time delta string
        if r.baseline_time_seconds > 0:
            time_change = (
                (r.time_seconds - r.baseline_time_seconds)
                / r.baseline_time_seconds
                * 100
            )
            time_color = (
                Colors.GREEN
                if time_change < -10
                else (Colors.FAIL if time_change > 10 else "")
            )
            time_delta_str = f"{time_change:+.0f}%"
        else:
            time_color = ""
            time_delta_str = "-"

        # Calculate memory delta string
        if r.baseline_memory_mb > 0:
            mem_change = (
                (r.memory_mb - r.baseline_memory_mb) / r.baseline_memory_mb * 100
            )
            mem_color = (
                Colors.GREEN
                if mem_change < -10
                else (Colors.FAIL if mem_change > 10 else "")
            )
            mem_delta_str = f"{mem_change:+.0f}%"
        else:
            mem_color = ""
            mem_delta_str = "-"

        print(
            f"{r.name:<22} {r.time_seconds:>12.2f} {r.baseline_time_seconds:>12.2f} "
            f"{time_color}{time_delta_str:>12}{Colors.ENDC if time_color else ''} "
            f"{r.memory_mb:>12.1f} {mem_color}{mem_delta_str:>12}{Colors.ENDC if mem_color else ''}"
        )

    print("-" * 90)

    # Overall Status
    print()
    total_new_fails = sum(r.new_failures for r in results)
    failed_steps = [r.name for r in results if not r.passed]

    if failed_steps:
        print(
            f"{Colors.FAIL}Overall: ❌ FAIL - {', '.join(failed_steps)} "
            f"({total_new_fails} new failure(s)){Colors.ENDC}"
        )
    else:
        print(
            f"{Colors.GREEN}Overall: ✅ PASS - All compliance checks passed!{Colors.ENDC}"
        )

    return all_passed


def print_phase3_summary(
    current_results: dict[str, Any],
    baseline_results: dict[str, Any],
    value_changes: list[str] | None = None,
    value_stats: dict[str, float] | None = None,
    other: str = "",
) -> ComplianceResult:
    """
    Print the IBSI 2 Phase 3 (reproducibility) report and grade it.

    The step fails when fewer scans ran than in the baseline, when a scan failed, when a
    feature value is NaN or a configuration failed, when a team's comparisons fall below the
    baseline, when its matches (at 1, 5 or 10 % of the range) fall by more than
    PHASE3_ALLOWED_DROP of its comparisons under the same agreement rule (IBSI sets no pass
    mark for phase 3; a smaller drop is listed), and when our values changed in the
    environment of the baseline (value_changes; `other`: see print_value_report).
    """
    print()
    print("-" * 60)
    print("IBSI 2 Phase 3 Reproducibility Report")
    print("-" * 60)
    scans = current_results.get("scans_processed", 0)
    print(f"  Scans:    {scans} ({current_results.get('patients', 0)} patients)")
    print(f"  Teams:    {current_results.get('teams_compared', 0)}")
    print(f"  Rule:     tolerance = 1, 5 or 10 % of the {current_results.get('agreement_rule', '?')}")
    print()
    print(f"  {'Team':<30} {'Values':>7} {'Within 1%':>10} {'Within 5%':>10} {'Within 10%':>11}")
    print(f"  {'-' * 30} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 11}")
    for team, data in current_results.get("team_results", {}).items():
        total = data.get("total", 0)
        pct = [data.get(k, 0) / total * 100 if total else 0.0 for k in ("matches_1pct", "matches_5pct", "matches_10pct")]
        print(f"  {team:<30} {total:>7} {pct[0]:>9.1f}% {pct[1]:>9.1f}% {pct[2]:>10.1f}%")

    problems = []
    small_drops: list[str] = []
    expected = baseline_results.get("scans_processed")
    if not baseline_results:
        problems.append("no phase 3 baseline")
    elif not expected:
        problems.append("the baseline has the old format (no scans_processed); update it")
    elif scans < expected:
        problems.append(f"{scans} scans processed, {expected} in the baseline")
    for key, label in (
        ("scans_failed", "scans failed"),
        ("nan_values", "NaN feature values"),
        ("pipeline_errors", "configurations with an error or an empty ROI"),
    ):
        if current_results.get(key):
            problems.append(f"{current_results[key]} {label}")
    if expected and baseline_results.get("agreement_rule") != current_results.get("agreement_rule"):
        problems.append("the baseline uses another agreement rule")
    elif expected:
        current_teams = current_results.get("team_results", {})
        for team, base in baseline_results.get("team_results", {}).items():
            cur = current_teams.get(team)
            if cur is None:
                problems.append(f"{team}: no comparison")
                continue
            allowed = int(PHASE3_ALLOWED_DROP * base.get("total", 0))
            for level in ("total", "matches_1pct", "matches_5pct", "matches_10pct"):
                drop = base.get(level, 0) - cur.get(level, 0)
                change = f"{team} {level}: {base.get(level, 0)} -> {cur.get(level, 0)}"
                if drop > (0 if level == "total" else allowed):
                    problems.append(change)
                elif drop > 0:
                    small_drops.append(f"{change} (at most {allowed} allowed)")
    print()
    if problems:
        print(f"{Colors.FAIL}NEW FAILURES - ACTION REQUIRED:{Colors.ENDC}")
        for problem in problems:
            print(f"    {problem}")
    elif small_drops:
        print(f"  Scans and errors: {Colors.GREEN}as in the baseline{Colors.ENDC}")
        print(f"  {Colors.WARNING}Team agreement: {len(small_drops)} small drops, within {PHASE3_ALLOWED_DROP:.1%} of the comparisons:{Colors.ENDC}")
        for drop in small_drops:
            print(f"    {drop}")
    else:
        print(f"  Scans, errors and team agreement: {Colors.GREEN}as in the baseline{Colors.ENDC}")
    value_problems = print_value_report("IBSI 2 Phase 3", value_changes, value_stats, other)

    wall = current_results.get("wall_time_seconds", 0.0)
    base_wall = baseline_results.get("wall_time_seconds", 0.0)
    memory = current_results.get("memory_mb", 0.0)
    base_memory = baseline_results.get("memory_mb", 0.0) if baseline_results.get("memory_metric") == current_results.get("memory_metric") else 0.0
    print()
    print(f"  Wall time: {wall:.1f} s (baseline {base_wall:.1f} s); largest worker memory: {memory:.0f} MB")
    print("-" * 60)
    return ComplianceResult(
        name="IBSI 2 Phase 3",
        passed=not problems and value_problems == 0,
        total=expected or scans,
        passed_count=scans - current_results.get("scans_failed", 0),
        new_failures=len(problems) + value_problems,
        value_changes=value_problems,
        time_seconds=wall,
        baseline_time_seconds=base_wall,
        memory_mb=memory,
        baseline_memory_mb=base_memory,
    )
