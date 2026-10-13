"""
Shared configuration for pre_push.py and generate_docs.py.

Centralizes paths, thresholds, and constants used by the development pipeline scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DEV_DIR = PROJECT_ROOT / "dev"
RESULTS_DIR = DEV_DIR / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
IBSI2_DATA_DIR = DEV_DIR / "IBSI2"
TEMPLATES_DIR = DEV_DIR / "templates"

# --- Baselines (minimal data for regression checking) ---
BENCHMARK_BASELINE = RESULTS_DIR / "benchmark_baseline.json"
IBSI_BASELINE = RESULTS_DIR / "ibsi_baseline.json"
IBSI2_BASELINE = RESULTS_DIR / "ibsi2_baseline.json"
IBSI2_PHASE2_BASELINE = RESULTS_DIR / "ibsi2_phase2_baseline.json"
IBSI2_PHASE3_BASELINE = RESULTS_DIR / "ibsi2_phase3_baseline.json"

# --- Our own values (the gate compares them at a relative difference of 1e-9) ---
IBSI1_VALUES = RESULTS_DIR / "ibsi1_values.json"
IBSI2_PHASE1_MAPS = RESULTS_DIR / "ibsi2_phase1_maps.npz"
IBSI2_PHASE2_VALUES = RESULTS_DIR / "ibsi2_phase2_values.json"
IBSI2_PHASE3_VALUES = RESULTS_DIR / "ibsi2_phase3_values.json"

# --- Copies of replaced baselines (dev/ has no git history; outside OneDrive) ---
BASELINE_BACKUP_DIR = Path.home() / "Library" / "Caches" / "pictologics-baseline-backups"

# --- Tool caches and gate logs (outside OneDrive) ---
GATE_DIR = Path.home() / "Library" / "Caches" / "pictologics-gate"
GATE_CACHE_DIR = GATE_DIR / "cache"  # emptied at the start of each clean run
GATE_LOG_DIR = GATE_DIR / "logs"  # one log per gate run

# --- Raw Caches (full data for documentation generation) ---
IBSI1_CACHE = RESULTS_DIR / "ibsi1_cache.json"
IBSI2_PHASE1_CACHE = RESULTS_DIR / "ibsi2_phase1_cache.json"
IBSI2_PHASE2_CACHE = RESULTS_DIR / "ibsi2_phase2_cache.json"
IBSI2_PHASE3_CACHE = RESULTS_DIR / "ibsi2_phase3_cache.json"
BENCHMARK_CACHE = RESULTS_DIR / "benchmark_latest.json"

# --- Thresholds ---
REGRESSION_THRESHOLD = 0.20  # 20% performance regression threshold

# --- IBSI 2 phase 3 ---
PHASE3_WORKERS = min(8, os.cpu_count() or 1)  # 10 workers were not faster (2026-09-28)

# --- IBSI rules beyond the pass counts (see compliance_utils) ---
# A passing IBSI 2 phase 1 map keeps its error at most this (half the IBSI limit of 1 %)
PHASE1_ERROR_MARGIN_PCT = 0.5
# Each phase 3 team may lose this part of its comparisons at each level (1, 5 and 10 %)
PHASE3_ALLOWED_DROP = 0.001


def ensure_results_dir() -> None:
    """Ensure the results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
