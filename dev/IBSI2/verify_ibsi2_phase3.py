#!/usr/bin/env python3
"""
IBSI 2 Phase 3 Compliance Verification Script.

This script validates filter-based feature extraction against IBSI 2 Phase 3
team submissions using a 51-patient soft-tissue sarcoma dataset with CT, PET,
and T1-weighted MRI.

Comparison Strategy:
- No consensus reference values exist for Phase 3
- Compare our computed values against 9 participating teams' submissions
- Report overlap statistics per team and per feature/filter combination

Reference: IBSI 2 Reference Manual, Section 6.5 (Phase 3: Validation)

Usage:
    python verify_ibsi2_phase3.py [--output docs/ibsi2_phase3_compliance.md]
"""

import json
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from pictologics import RadiomicsPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev.compliance_utils import (  # noqa: E402
    FEATURE_MAP,
    PHASE3_TEAMS,
    WORKERS,
    get_rounding_decimals,
)

# =============================================================================
# Configuration: Preprocessing per Modality (IBSI Table 6.5)
# =============================================================================

MODALITY_CONFIG = {
    "CT": {
        "spacing": (1.0, 1.0, 1.0),
        "interpolation": "cubic",
        "round_intensities": True,
        "resegment_range": (-200, 200),
    },
    "PET": {
        "spacing": (3.0, 3.0, 3.0),
        "interpolation": "cubic",
        "round_intensities": False,
        "resegment_range": (0, float("inf")),
    },
    "MRI": {
        "spacing": (1.0, 1.0, 1.0),
        "interpolation": "cubic",
        "round_intensities": False,
        "resegment_range": (0, float("inf")),
    },
}

# =============================================================================
# Configuration: Filter Definitions (IBSI Table 6.6)
# =============================================================================


def get_filter_configs() -> dict[int, dict[str, Any]]:
    """
    Get all 9 filter configurations for IBSI 2 Phase 3.

    Returns:
        Dict mapping filter_id (1-9) to config dict with 'name', 'step', and 'notes'.
    """
    return {
        # ID 1: No filter (baseline)
        1: {
            "name": "None",
            "step": None,
            "notes": "Baseline (no filter)",
        },
        # ID 2: Mean filter, M=3
        2: {
            "name": "Mean",
            "step": {"step": "filter", "params": {"type": "mean", "support": 3}},
            "notes": "3D, M=3",
        },
        # ID 3: LoG, σ*=3.0mm, 4σ cutoff
        3: {
            "name": "LoG",
            "step": {
                "step": "filter",
                "params": {"type": "log", "sigma_mm": 3.0, "truncate": 4.0},
            },
            "notes": "σ=3mm, 4σ cutoff",
        },
        # ID 4: Laws S5E5L5, rotation-invariant max, energy δ=5
        4: {
            "name": "Laws",
            "step": {
                "step": "filter",
                "params": {
                    "type": "laws",
                    "kernel": "S5E5L5",
                    "rotation_invariant": True,
                    "pooling": "max",
                    "compute_energy": True,
                    "energy_distance": 5,
                },
            },
            "notes": "S5E5L5, rot-inv max, energy δ=5",
        },
        # ID 5: Gabor 2D, σ*=3mm, λ*=3mm, γ=1, θ=-5π/8
        5: {
            "name": "Gabor",
            "step": {
                "step": "filter",
                "params": {
                    "type": "gabor",
                    "sigma_mm": 3.0,
                    "lambda_mm": 3.0,
                    "gamma": 1.0,
                    "theta": -5 * np.pi / 8,
                },
            },
            "notes": "2D, σ=λ=3mm, γ=1, θ=-5π/8",
        },
        # ID 6: Coiflet 3, LHH level 1, rotation-invariant average
        6: {
            "name": "Coiflet 3",
            "step": {
                "step": "filter",
                "params": {
                    "type": "wavelet",
                    "wavelet": "coif3",
                    "level": 1,
                    "decomposition": "LHH",
                    "rotation_invariant": True,
                    "pooling": "average",
                },
            },
            "notes": "LHH level 1, rot-inv avg",
        },
        # ID 7: Coiflet 3, HHH level 2, rotation-invariant max
        7: {
            "name": "Coiflet 3",
            "step": {
                "step": "filter",
                "params": {
                    "type": "wavelet",
                    "wavelet": "coif3",
                    "level": 2,
                    "decomposition": "HHH",
                    "rotation_invariant": True,
                    "pooling": "max",
                },
            },
            "notes": "HHH level 2, rot-inv max",
        },
        # ID 8: Simoncelli B map level 1
        8: {
            "name": "Simoncelli",
            "step": {"step": "filter", "params": {"type": "simoncelli", "level": 1}},
            "notes": "B map level 1",
        },
        # ID 9: Simoncelli B map level 2
        9: {
            "name": "Simoncelli",
            "step": {"step": "filter", "params": {"type": "simoncelli", "level": 2}},
            "notes": "B map level 2",
        },
    }


def get_filter_name(filter_id: int) -> str:
    """Get human-readable name for a filter ID."""
    configs = get_filter_configs()
    if filter_id in configs:
        cfg = configs[filter_id]
        return f"{cfg['name']} ({cfg['notes']})"
    return f"Filter {filter_id}"


# Short names for heatmap x-axis labels
FILTER_SHORT_NAMES = {
    1: "None",
    2: "Mean",
    3: "LoG",
    4: "Laws",
    5: "Gabor",
    6: "Coif3 LHH L1",
    7: "Coif3 HHH L2",
    8: "Simon. L1",
    9: "Simon. L2",
}


# =============================================================================
# Feature Mapping
# =============================================================================

# Reverse mapping: IBSI tag -> pictologics name
TAG_TO_PICTO = {v: k for k, v in FEATURE_MAP.items()}

# A value matches a team's value when the difference is within 1, 5 or 10 % of this range.
AGREEMENT_RULE = "team's range for each modality, filter and feature (max - min over the patients)"
MEMORY_METRIC = "largest worker peak resident memory"



# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FeatureComparison:
    """Comparison result for a single feature/filter combination."""

    patient_id: str
    feature_tag: str
    filter_id: int
    our_value: float
    team_value: float
    matches: bool
    error: float
    feature_range: float = 0.0


@dataclass
class TeamComparison:
    """Comparison results against one team."""

    team_name: str
    total_comparisons: int
    matches_1pct: int = 0  # Matches within 1% of range
    matches_5pct: int = 0  # Matches within 5% of range
    matches_10pct: int = 0  # Matches within 10% of range
    mismatches: list[FeatureComparison] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TeamComparison":
        data["mismatches"] = [
            FeatureComparison(**m) for m in data.get("mismatches", [])
        ]
        return cls(**data)


@dataclass
class PatientResult:
    """Results for a single patient/modality combination."""

    patient_id: str
    modality: str
    features: dict[str, dict[int, float]]  # feature_tag -> {filter_id: value}
    time_seconds: float = 0.0
    memory_mb: float = 0.0  # peak resident memory of the worker so far
    errors: int = 0  # configurations with an error or an empty ROI
    nan_values: int = 0


class Cell(NamedTuple):
    """One value that both we and a team have."""

    result: "PatientResult"
    feature_tag: str
    filter_id: int
    our_value: float  # rounded to the team's number of decimals
    team_value: float
    feature_range: float  # the team's range for this modality, filter and feature
    decimals: int


def within(cell: Cell, fraction: float) -> bool:
    """The match rule: the difference is within `fraction` of the range; a zero range needs an
    exact match."""
    error = abs(cell.our_value - cell.team_value)
    return error <= cell.feature_range * fraction if cell.feature_range > 0 else error == 0


def worker_peak_mb() -> float:
    """Peak resident memory of this process so far, in MB (ru_maxrss is in bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


# =============================================================================
# Helper Functions
# =============================================================================


def build_pipeline_config(modality: str, filter_config: dict | None) -> list[dict]:
    """Build pipeline configuration for a modality and filter."""
    mod_cfg = MODALITY_CONFIG[modality]

    steps = [
        {
            "step": "resample",
            "params": {
                "new_spacing": mod_cfg["spacing"],
                "interpolation": mod_cfg["interpolation"],
                "mask_interpolation": "linear",
                "mask_threshold": 0.5,
            },
        },
    ]

    if mod_cfg["round_intensities"]:
        steps.append({"step": "round_intensities", "params": {}})

    # Resegmentation
    range_min, range_max = mod_cfg["resegment_range"]
    if range_max == float("inf"):
        steps.append({"step": "resegment", "params": {"range_min": range_min}})
    else:
        steps.append(
            {
                "step": "resegment",
                "params": {"range_min": range_min, "range_max": range_max},
            }
        )

    # Filter (if any)
    if filter_config is not None:
        steps.append(filter_config)

    # Feature extraction (intensity only)
    steps.append({"step": "extract_features", "params": {"families": ["intensity"]}})

    return steps


def run_single_patient(
    patient_id: str,
    modality: str,
    image_path: str,
    mask_path: str,
    filter_configs: dict[int, dict],
) -> PatientResult:
    """Run all filters for a single patient/modality: one pipeline with one configuration per
    filter, so the scan is loaded and preprocessed once and every filter continues from there."""
    features: dict[str, dict[int, float]] = {tag: {} for tag in FEATURE_MAP.values()}
    start_time = time.perf_counter()

    pipeline = RadiomicsPipeline()
    names = {}
    for filter_id, filter_cfg in filter_configs.items():
        names[filter_id] = f"phase3_{modality}_{filter_id}"
        pipeline.add_config(names[filter_id], build_pipeline_config(modality, filter_cfg.get("step")))
    results = pipeline.run(image_path, mask_path, config_names=list(names.values()))
    for filter_id, name in names.items():
        for picto_name, ibsi_tag in FEATURE_MAP.items():
            features[ibsi_tag][filter_id] = float(results[name].get(picto_name, np.nan))

    elapsed = time.perf_counter() - start_time
    errors = sum(entry.get("status") != "completed" for entry in pipeline._log)
    nan_values = sum(bool(np.isnan(v)) for values in features.values() for v in values.values())
    return PatientResult(patient_id, modality, features, elapsed, worker_peak_mb(), errors, nan_values)


SIMONCELLI_IDS = (8, 9)


def run_simoncelli_mirror(
    patient_id: str, modality: str, image_path: str, mask_path: str
) -> tuple[str, str, dict[str, dict[int, float]]]:
    """The Simoncelli filters of one scan with the mirror boundary of the manual's phase 3
    table (the check itself uses the periodic boundary)."""
    pipeline = RadiomicsPipeline()
    names = {}
    for filter_id in SIMONCELLI_IDS:
        params = {**get_filter_configs()[filter_id]["step"]["params"], "boundary": "mirror"}
        names[filter_id] = f"mirror_{modality}_{filter_id}"
        pipeline.add_config(
            names[filter_id], build_pipeline_config(modality, {"step": "filter", "params": params})
        )
    results = pipeline.run(image_path, mask_path, config_names=list(names.values()))
    features = {
        tag: {fid: float(results[name].get(picto, np.nan)) for fid, name in names.items()}
        for picto, tag in FEATURE_MAP.items()
    }
    return patient_id, modality, features


# =============================================================================
# Main Validator Class
# =============================================================================


class IBSIPhase3Validator:
    """
    Validator for IBSI 2 Phase 3 reproducibility benchmarks.

    Computes features from filtered response maps and compares against
    team submissions from the IBSI data analysis repository.
    """

    def __init__(self, data_dir: Path | str, n_workers: int = 8):
        """
        Initialize the validator.

        Args:
            data_dir: Path to IBSI2 data directory
            n_workers: Number of parallel workers (default: 8)
        """
        self.data_dir = Path(data_dir)
        self.n_workers = n_workers

        # Paths
        self.validation_dir = self.data_dir / "data" / "validation"
        self.submissions_dir = self.data_dir / "references" / "phase3_submissions"

        # Results
        self._patient_results: list[PatientResult] = []
        self._team_comparisons: dict[str, TeamComparison] = {}
        self._failed_scans: list[str] = []
        self._wall_time = 0.0
        self._team_cache: dict[str, dict[tuple[str, str], list[tuple[str, int, str, float]]]] = {}

    def get_patient_ids(self) -> list[str]:
        """Get list of patient IDs from validation data."""
        # CT images follow pattern: STS_XXX_image.nii.gz
        ct_dir = self.validation_dir / "ct"
        if not ct_dir.exists():
            return []
        # Extract patient ID from filename like STS_001_image.nii.gz -> STS_001
        patient_ids = []
        for f in ct_dir.glob("*_image.nii.gz"):
            patient_id = f.stem.replace("_image.nii", "").replace(".nii", "")
            if patient_id.startswith("STS_"):
                patient_ids.append(patient_id)
        return sorted(set(patient_ids))

    def get_paths(self, patient_id: str, modality: str) -> tuple[Path, Path]:
        """Get image and mask paths for a patient/modality.

        Files are named:
        - Image: {modality_dir}/{patient_id}_image.nii.gz
        - Mask: masks/{patient_id}_{modality}_mask.nii.gz
        """
        modality_dirs = {"CT": "ct", "PET": "pet", "MRI": "mri"}
        mask_modalities = {"CT": "CT", "PET": "PET", "MRI": "MRI"}

        mod_dir = modality_dirs.get(modality, modality.lower())
        mask_mod = mask_modalities.get(modality, modality)

        image_path = self.validation_dir / mod_dir / f"{patient_id}_image.nii.gz"
        mask_path = (
            self.validation_dir / "masks" / f"{patient_id}_{mask_mod}_mask.nii.gz"
        )

        return image_path, mask_path

    def load_team_submission(
        self, team: str, patient_id: str, modality: str
    ) -> pd.DataFrame | None:
        """Load a team's submission CSV for a patient/modality."""
        mod_suffix = {"CT": "CT", "PET": "PET", "MRI": "MR_T1"}.get(modality, modality)
        csv_path = self.submissions_dir / team / f"{patient_id}_{mod_suffix}.csv"

        if not csv_path.exists():
            return None

        try:
            # Use python engine with sep=None to auto-detect delimiter (handles ; vs ,)
            # Read as string to preserve formatting (decimals)
            df = pd.read_csv(csv_path, sep=None, engine="python", dtype=str)
            return df
        except Exception:
            return None

    def run_all_patients(
        self,
        patient_limit: int | None = None,
        modalities: list[str] | None = None,
        verbose: bool = True,
    ) -> list[PatientResult]:
        """
        Run validation for all patients.

        Args:
            patient_limit: Max number of patients to process (for testing)
            modalities: List of modalities to process (default: all)
        """
        patient_ids = self.get_patient_ids()
        if patient_limit:
            patient_ids = patient_ids[:patient_limit]

        if modalities is None:
            modalities = ["CT", "PET", "MRI"]

        filter_configs = get_filter_configs()
        self._patient_results = []
        self._failed_scans = []
        self._team_cache = {}
        wall_start = time.perf_counter()

        print(
            f"Processing {len(patient_ids)} patients × {len(modalities)} modalities..."
        )

        # Prepare tasks
        tasks = []
        for patient_id in patient_ids:
            for modality in modalities:
                image_path, mask_path = self.get_paths(patient_id, modality)
                if image_path.exists() and mask_path.exists():
                    tasks.append(
                        (patient_id, modality, str(image_path), str(mask_path))
                    )

        # Run in parallel
        from tqdm import tqdm

        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=WORKERS) as executor:
            futures = {
                executor.submit(
                    run_single_patient, pid, mod, img, msk, filter_configs
                ): (pid, mod)
                for pid, mod, img, msk in tasks
            }

            iterator = as_completed(futures)
            if not verbose:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc="Processing Patients",
                    unit="scan",
                )

            for future in iterator:
                pid, mod = futures[future]
                try:
                    result = future.result()
                    self._patient_results.append(result)
                    if verbose:
                        print(f"  ✓ {pid}/{mod} ({result.time_seconds:.1f}s)")
                except Exception as e:
                    self._failed_scans.append(f"{pid}/{mod}: {e}")
                    # If using tqdm, use write to avoid breaking bar
                    if not verbose:
                        tqdm.write(f"  ✗ {pid}/{mod}: {e}")
                    else:
                        print(f"  ✗ {pid}/{mod}: {e}")

        # Completion order differs between runs; the report and the figures must not.
        self._patient_results.sort(key=lambda r: (r.patient_id, r.modality))
        self._wall_time = time.perf_counter() - wall_start
        return self._patient_results

    def _team_values(self, team: str) -> dict[tuple[str, str], list[tuple[str, int, str, float]]]:
        """A team's submitted values per scan as (feature tag, filter ID, text, value). Each
        file is read once."""
        if team not in self._team_cache:
            per_scan: dict[tuple[str, str], list[tuple[str, int, str, float]]] = {}
            for result in self._patient_results:
                team_df = self.load_team_submission(team, result.patient_id, result.modality)
                if team_df is None:
                    continue
                cells = []
                for _, row in team_df.iterrows():
                    feature_tag = row.get("feature_tag", "")
                    for filter_id in range(1, 10):
                        col = f"ID_{filter_id}"
                        if col not in team_df.columns:
                            continue
                        text = str(row.get(col))
                        try:
                            value = float(text)
                        except (ValueError, TypeError):
                            continue
                        if not np.isnan(value):
                            cells.append((feature_tag, filter_id, text, value))
                per_scan[(result.patient_id, result.modality)] = cells
            self._team_cache[team] = per_scan
        return self._team_cache[team]

    def _score_team(self, team: str) -> list[Cell]:
        """Every value that both we and the team have, with the team's range for its modality,
        filter and feature over all patients. Our value is rounded to the team's number of
        decimals first, as in the IBSI tolerance tests."""
        team_values = self._team_values(team)
        spread: dict[tuple[str, int, str], list[float]] = defaultdict(list)
        for result in self._patient_results:
            for tag, filter_id, _text, value in team_values.get((result.patient_id, result.modality), []):
                spread[(result.modality, filter_id, tag)].append(value)
        ranges = {key: max(values) - min(values) for key, values in spread.items()}

        cells = []
        for result in self._patient_results:
            for tag, filter_id, text, value in team_values.get((result.patient_id, result.modality), []):
                our_val = result.features.get(tag, {}).get(filter_id)
                if our_val is None or np.isnan(our_val):
                    continue
                decimals = get_rounding_decimals(text)
                cells.append(
                    Cell(result, tag, filter_id, round(our_val, decimals), value,
                         ranges[(result.modality, filter_id, tag)], decimals)
                )
        return cells

    def compare_with_teams(self) -> dict[str, TeamComparison]:
        """
        Compare our results against all team submissions.

        A value matches when the difference is within 1, 5 or 10 % of the team's range for
        its modality, filter and feature (see AGREEMENT_RULE and within()).
        """
        self._team_comparisons = {}
        for team in PHASE3_TEAMS:
            cells = self._score_team(team)
            mismatches = [
                FeatureComparison(
                    patient_id=f"{c.result.patient_id}_{c.result.modality}",
                    feature_tag=c.feature_tag,
                    filter_id=c.filter_id,
                    our_value=c.our_value,
                    team_value=c.team_value,
                    matches=False,
                    error=abs(c.our_value - c.team_value),
                    feature_range=c.feature_range,
                )
                for c in cells
                if not within(c, 0.10)
            ]
            self._team_comparisons[team] = TeamComparison(
                team_name=team,
                total_comparisons=len(cells),
                matches_1pct=sum(within(c, 0.01) for c in cells),
                matches_5pct=sum(within(c, 0.05) for c in cells),
                matches_10pct=sum(within(c, 0.10) for c in cells),
                mismatches=mismatches,
            )
        return self._team_comparisons

    def simoncelli_boundary_agreement(self) -> dict[str, tuple[int, int, int]]:
        """Per team, the number of Simoncelli values (IDs 8, 9) within 1 % of the range with the
        periodic boundary that the check uses and with the mirror boundary, and the number of
        values compared."""
        tasks = [
            (r.patient_id, r.modality, *map(str, self.get_paths(r.patient_id, r.modality)))
            for r in self._patient_results
        ]
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=WORKERS) as executor:
            mirror = {
                (pid, mod): feats
                for pid, mod, feats in executor.map(run_simoncelli_mirror, *zip(*tasks, strict=True))
            }
        agreement = {}
        for team in PHASE3_TEAMS:
            cells = [c for c in self._score_team(team) if c.filter_id in SIMONCELLI_IDS]
            if not cells:
                continue
            periodic = sum(within(c, 0.01) for c in cells)
            mirrored = 0
            for c in cells:
                value = mirror[(c.result.patient_id, c.result.modality)][c.feature_tag][c.filter_id]
                mirrored += not np.isnan(value) and within(c._replace(our_value=round(value, c.decimals)), 0.01)
            agreement[team] = (periodic, mirrored, len(cells))
        return agreement

    def values(self) -> dict[str, dict[str, dict[str, float]]]:
        """Our feature values per scan, feature and filter, for the gate's value check."""
        return {
            f"{r.patient_id}|{r.modality}": {
                tag: {str(fid): value for fid, value in sorted(per_filter.items())}
                for tag, per_filter in sorted(r.features.items())
            }
            for r in self._patient_results
        }

    def load_values(self, values: dict[str, dict[str, dict[str, float]]]) -> None:
        """Restore the patient results from saved values (for the docs without a new run)."""
        self._patient_results = []
        for key, features in sorted(values.items()):
            patient_id, modality = key.split("|")
            self._patient_results.append(
                PatientResult(
                    patient_id,
                    modality,
                    {tag: {int(fid): v for fid, v in d.items()} for tag, d in features.items()},
                )
            )
        self._team_cache = {}

    def save_cache(self, path: Path | str) -> None:
        """Save comparison results to JSON cache."""
        path = Path(path)
        data = {
            "metadata": {
                "scans_processed": len(self._patient_results),
                "timestamp": datetime.now().isoformat(),
            },
            "comparisons": {
                team: comp.to_dict() for team, comp in self._team_comparisons.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_cache(self, path: Path | str) -> bool:
        """Load comparison results from JSON cache. Returns True if successful."""
        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Handle legacy cache format (direct dictionary of comparisons)
            if "comparisons" not in data and "metadata" not in data:
                comparisons_data = data
                self._cached_metadata = {}
            else:
                comparisons_data = data.get("comparisons", {})
                self._cached_metadata = data.get("metadata", {})

            self._team_comparisons = {
                team: TeamComparison.from_dict(comp_data)
                for team, comp_data in comparisons_data.items()
            }
            return True
        except Exception as e:
            print(f"Failed to load cache from {path}: {e}")
            return False

    def get_summary(self) -> dict:
        """Get structured summary dictionary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "agreement_rule": AGREEMENT_RULE,
            "scans_processed": len(self._patient_results),
            "scans_failed": len(self._failed_scans),
            "nan_values": sum(p.nan_values for p in self._patient_results),
            "pipeline_errors": sum(p.errors for p in self._patient_results),
            "patients": len({p.patient_id for p in self._patient_results}),
            "teams_compared": len(self._team_comparisons),
            "total_time_seconds": sum(p.time_seconds for p in self._patient_results),
            "wall_time_seconds": self._wall_time,
            "memory_metric": MEMORY_METRIC,
            "memory_mb": max((p.memory_mb for p in self._patient_results), default=0.0),
            "team_results": {
                team: {
                    "total": comp.total_comparisons,
                    "matches_1pct": comp.matches_1pct,
                    "matches_5pct": comp.matches_5pct,
                    "matches_10pct": comp.matches_10pct,
                    "mismatch_count": len(comp.mismatches),
                }
                for team, comp in self._team_comparisons.items()
            },
        }

    def _generate_heatmap(self, docs_dir: Path) -> dict[str, dict[str, float]]:
        """Generate a heatmap: configurations (rows) vs teams (columns).

        Each cell shows the percentage of features matching within 10% of feature
        range for that configuration/team combination. Uses dark theme matching
        benchmark plots.

        Returns:
            The scored cells per team, for reuse in the patient heatmap.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        teams = list(self._team_comparisons.keys())
        if not teams or not self._patient_results:
            print("    Skipping heatmap (no patient results in memory)")
            return {}

        config_ids = list(range(1, 10))  # Filters 1-9
        scored = {team: self._score_team(team) for team in teams}
        data = []  # rows = configs, cols = teams
        for filter_id in config_ids:
            row = []
            for team in teams:
                cells = [c for c in scored[team] if c.filter_id == filter_id]
                row.append(sum(within(c, 0.10) for c in cells) / len(cells) * 100 if cells else np.nan)
            data.append(row)

        data_array = np.array(data)

        # Dark theme styling (matching benchmark plots)
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        # Create custom colormap (red -> yellow -> green)
        colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
        cmap = mcolors.LinearSegmentedColormap.from_list("compliance", colors)
        cmap.set_bad(color="#333344")  # NaN values in dark gray

        # Plot heatmap
        im = ax.imshow(data_array, cmap=cmap, aspect="auto", vmin=0, vmax=100)

        # Team labels at top (use ha='left' for rotated labels at top)
        ax.set_xticks(range(len(teams)))
        ax.set_xticklabels(teams, rotation=45, ha="left", fontsize=9, color="white")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        # Configuration labels on y-axis
        config_labels = [FILTER_SHORT_NAMES.get(i, f"ID {i}") for i in config_ids]
        ax.set_yticks(range(len(config_ids)))
        ax.set_yticklabels(config_labels, fontsize=10, color="white")

        # Add text annotations for each cell
        for i in range(len(config_ids)):
            for j in range(len(teams)):
                val = data_array[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 50 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight="bold",
                    )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(
            "Match % (within 10% of feature range)",
            rotation=270,
            labelpad=20,
            color="white",
            fontsize=10,
        )
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # Title and labels with teal accent
        ax.set_title(
            "IBSI 2 Phase 3: Agreement by Configuration",
            fontsize=14,
            fontweight="bold",
            pad=20,
            color="#00d9ff",
        )
        ax.set_xlabel("Team", fontsize=12, color="white")
        ax.set_ylabel("Image Filter Configuration", fontsize=12, color="white")

        # Grid styling
        ax.set_xticks(np.arange(len(teams)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(config_ids)) - 0.5, minor=True)
        ax.grid(which="minor", color="#333344", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Save figure
        assets_dir = docs_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        output_path = assets_dir / "ibsi2_phase3_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()
        plt.style.use("default")  # Reset style
        print(f"Generated heatmap: {output_path}")
        return scored

    def _generate_patient_heatmap(self, docs_dir: Path, scored: dict[str, list[Cell]]) -> None:
        """Generate a heatmap: patients (rows) vs teams (columns).

        Each cell shows the percentage of features matching within 10% of feature
        range for that patient/team combination. Uses dark theme matching benchmark plots.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        teams = list(self._team_comparisons.keys())
        if not teams or not self._patient_results:
            print("    Skipping patient heatmap (no patient results in memory)")
            return

        patient_labels = [f"{r.patient_id}_{r.modality}" for r in self._patient_results]
        data = []  # rows = patients, cols = teams
        per_scan = {team: defaultdict(list) for team in teams}
        for team in teams:
            for c in scored.get(team) or self._score_team(team):
                per_scan[team][id(c.result)].append(c)
        for result in self._patient_results:
            row = []
            for team in teams:
                cells = per_scan[team].get(id(result), [])
                row.append(sum(within(c, 0.10) for c in cells) / len(cells) * 100 if cells else np.nan)
            data.append(row)

        data_array = np.array(data)
        n_patients = len(patient_labels)

        # Dark theme styling
        plt.style.use("dark_background")
        fig_height = max(12, n_patients * 0.12)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        # Create custom colormap (red -> yellow -> green)
        colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
        cmap = mcolors.LinearSegmentedColormap.from_list("compliance", colors)
        cmap.set_bad(color="#333344")

        # Plot heatmap
        im = ax.imshow(data_array, cmap=cmap, aspect="auto", vmin=0, vmax=100)

        # Team labels at top
        ax.set_xticks(range(len(teams)))
        ax.set_xticklabels(teams, rotation=45, ha="left", fontsize=9, color="white")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        # Patient labels on y-axis - show every Nth for readability
        if n_patients <= 60:
            ax.set_yticks(range(n_patients))
            ax.set_yticklabels(patient_labels, fontsize=5, color="white")
        else:
            step = max(1, n_patients // 40)
            ax.set_yticks(range(0, n_patients, step))
            ax.set_yticklabels(
                [patient_labels[i] for i in range(0, n_patients, step)],
                fontsize=5,
                color="white",
            )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label(
            "Match % (within 10% of feature range)",
            rotation=270,
            labelpad=20,
            color="white",
            fontsize=10,
        )
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # Title and labels
        ax.set_title(
            "IBSI 2 Phase 3: Per-Patient Agreement with Teams",
            fontsize=14,
            fontweight="bold",
            pad=60,
            color="#00d9ff",
        )
        ax.set_xlabel("Team", fontsize=12, color="white", labelpad=10)
        ax.set_ylabel("Patient / Modality", fontsize=12, color="white")

        # Grid styling
        ax.set_xticks(np.arange(len(teams)) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_patients) - 0.5, minor=True)
        ax.grid(which="minor", color="#333344", linestyle="-", linewidth=0.3)
        ax.tick_params(which="minor", size=0)

        # Save figure
        assets_dir = docs_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        output_path = assets_dir / "ibsi2_phase3_patient_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()
        plt.style.use("default")
        print(f"Generated patient heatmap: {output_path}")

    def generate_docs(self, output_path: Path | str) -> None:
        """Generate markdown documentation with comparison results."""
        output_path = Path(output_path)
        summary = self.get_summary()

        lines = []
        lines.append("# IBSI 2 Phase 3 Compliance: Reproducibility")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(
            "IBSI 2 Phase 3 focuses on **reproducibility across different software implemen"
            "tations**. Unlike Phases 1 and 2, there are no consensus reference values. Ins"
            "tead, participant results are compared against each other to measure overlap."
        )
        lines.append("")
        lines.append(
            "This page documents Pictologics' agreement with 9 other teams on a **multimodal"
            " (CT, MRI, PET) soft-tissue sarcoma dataset of 51 patients**."
        )
        lines.append("")
        lines.append("## How to Run the Benchmarks")
        lines.append("")
        lines.append("### 1. Download the Data")
        lines.append("")
        lines.append(
            "-   **IBSI 2 Phase 3 Data**: Please refer to the [IBSI GitHub repository](htt"
            "ps://github.com/theibsi/data_sets) for all data download instructions."
        )
        lines.append("")
        lines.append(
            "Organize the data as follows:"
            "\n- `data/ibsi2/data/validation/ct/*.nii.gz`"
            "\n- `data/ibsi2/data/validation/pet/*.nii.gz`"
            "\n- `data/ibsi2/data/validation/mri/*.nii.gz`"
            "\n- `data/ibsi2/data/validation/masks/*.nii.gz`"
        )
        lines.append("")
        lines.append("### 2. Run Validation Programmatically")
        lines.append("")
        lines.append("```python")
        lines.append("from pictologics import RadiomicsPipeline")
        lines.append("")
        lines.append("# Example: Run Mean filter (ID 2) on a specific patient/modality")
        lines.append(
            'image_path = "data/ibsi2/data/validation/ct/STS_001_image.nii.gz"'
        )
        lines.append(
            'mask_path = "data/ibsi2/data/validation/masks/STS_001_CT_mask.nii.gz"'
        )
        lines.append("")
        lines.append("pipeline = RadiomicsPipeline()")
        lines.append("")
        lines.append("# Define IBSI 2 Phase 3 CT Preprocessing")
        lines.append("preprocess_steps = [")
        lines.append('    {"step": "resample", "params": {')
        lines.append('        "new_spacing": (1.0, 1.0, 1.0),')
        lines.append('        "interpolation": "cubic",')
        lines.append('        "mask_interpolation": "linear",')
        lines.append('        "mask_threshold": 0.5')
        lines.append("    }},")
        lines.append('    {"step": "round_intensities", "params": {}},')
        lines.append(
            '    {"step": "resegment", "params": {"range_min": -200, "range_max": 200}}'
        )
        lines.append("]")
        lines.append("")
        lines.append("# Add Mean Filter (ID 2) and Feature Extraction")
        lines.append("config = preprocess_steps + [")
        lines.append(
            '    {"step": "filter", "params": {"type": "mean", "support": 3}},'
        )
        lines.append(
            '    {"step": "extract_features", "params": {"families": ["intensity"]}}'
        )
        lines.append("]")
        lines.append("")
        lines.append('pipeline.add_config("phase3_demo", config)')
        lines.append(
            'results = pipeline.run(image_path, mask_path, config_names=["phase3_demo"])'
        )
        lines.append('print(results["phase3_demo"])')
        lines.append("```")
        lines.append("")
        lines.append('!!! warning "Execution Time"')
        lines.append(
            "    **Processing 153 scans (51 patients × 3 modalities) with multiple filters "
            "takes significant time** (minutes to hours depending on CPU). The full validat"
            "ion suite is designed to run in parallel on multiple cores."
        )
        lines.append("")
        lines.append("## Phase 3 Results")
        lines.append("")
        lines.append(
            f"**Summary**: Processed {summary['scans_processed']} scans ({summary['patients']} "
            f"patients × 3 modalities), compared against {summary['teams_compared']} teams."
        )

        lines.extend(
            [
                "",
                "## Tolerance Breakdown (% of Feature Range)",
                "",
                "A value matches when its difference from the team's value is within 1, 5 or 10 % of the "
                "team's *range for that modality, filter and feature* (the maximum minus the minimum of "
                "the team's values over the 51 patients). Our value is first rounded to the team's number "
                "of decimals, as in the IBSI tolerance tests. A zero range needs an exact match.",
                "",
                '!!! note "A stricter rule than in version 0.5.1"',
                "    Up to version 0.5.1, this page used one range for each team and feature, over all "
                "patients, modalities and filters. That range spans the CT, PET and MR values of all "
                "filters, so it is wider, and more values matched. This page uses the range of one "
                "modality and one filter, so its shares are lower.",
                "",
                "    The values did not change. With the old rule, the 0.6.0 values give the same match "
                "counts as version 0.5.1 for all 9 teams: 91.97 % of the 205,597 values within 1 % "
                "(checked on 2026-10-04).",
                "",
                "| Team | Total Features | Within 1% | Within 5% | Within 10% | Status |",
                "|:-----|:--------------:|----------:|----------:|-----------:|:------:|",
            ]
        )

        for team, comp in self._team_comparisons.items():
            total = comp.total_comparisons
            if total == 0:
                continue

            pct_1 = comp.matches_1pct / total * 100
            pct_5 = comp.matches_5pct / total * 100
            pct_10 = comp.matches_10pct / total * 100

            # Visual feedback based on 10% match rate
            if pct_10 >= 95:
                status = "✅ 95%+"
            elif pct_10 >= 80:
                status = "🟢 80-95%"
            elif pct_10 >= 60:
                status = "🟡 60-80%"
            elif pct_10 >= 40:
                status = "🟠 40-60%"
            else:
                status = "🔴 <40%"

            lines.append(
                f"| {team} | {total} | "
                f"{comp.matches_1pct} ({pct_1:.1f}%) | "
                f"{comp.matches_5pct} ({pct_5:.1f}%) | "
                f"{comp.matches_10pct} ({pct_10:.1f}%) | {status} |"
            )

        # Generate heatmap visualizations
        scored = self._generate_heatmap(output_path.parent)
        self._generate_patient_heatmap(output_path.parent, scored)

        lines.extend(
            [
                "",
                "## Agreement by Configuration",
                "",
                "The heatmap below shows the percentage of values within 10 % of the team's range (for the "
                "modality, filter and feature) for each configuration/team combination.",
                "",
                "![Phase 3 Agreement Heatmap](assets/ibsi2_phase3_heatmap.png)",
                "",
                "## Per-Patient Agreement",
                "",
                "The heatmap below shows per-patient agreement with each team across all filter configurations.",
                "",
                "![Phase 3 Per-Patient Agreement Heatmap](assets/ibsi2_phase3_patient_heatmap.png)",
                "",
            ]
        )

        # Add mismatch details per team
        lines.extend(["", "## Mismatch Details", ""])
        lines.append(
            "Mismatches shown below are cases where the error exceeds **10%** of the team's range for "
            "the modality, filter and feature."
        )
        lines.append("")

        # Configuration legend table
        lines.extend(
            [
                "### Configuration Legend",
                "",
                "| Short Name | Full Description |",
                "|------------|------------------|",
                "| None | No filter (baseline) |",
                "| Mean | Mean filter (3×3×3 kernel) |",
                "| LoG | Laplacian of Gaussian (σ=3mm) |",
                "| Laws | Laws S5E5L5 texture filter |",
                "| Gabor | Gabor filter (2D, θ=-5π/8) |",
                "| Coif3 LHH L1 | Coiflet 3 wavelet, LHH decomposition, level 1 |",
                "| Coif3 HHH L2 | Coiflet 3 wavelet, HHH decomposition, level 2 |",
                "| Simon. L1 | Simoncelli steerable pyramid, level 1 (periodic boundary, see below) |",
                "| Simon. L2 | Simoncelli steerable pyramid, level 2 (periodic boundary, see below) |",
                "",
            ]
        )

        # Why the Simoncelli filters use the periodic boundary
        if self._patient_results:
            agreement = self.simoncelli_boundary_agreement()
            periodic_all = sum(a[0] for a in agreement.values())
            mirror_all = sum(a[1] for a in agreement.values())
            total_all = sum(a[2] for a in agreement.values()) or 1
            periodic_teams = sum(a[0] >= a[1] for a in agreement.values())
            lines.extend(
                [
                    "### Simoncelli Boundary",
                    "",
                    "The IBSI 2 manual's phase 3 table lists the mirror boundary for all filters. For the "
                    "Simoncelli filters (IDs 8 and 9), Pictologics uses the periodic boundary of its FFT "
                    "filters, as in IBSI 2 phases 1 and 2, because it gives the better match: over all "
                    f"teams, {periodic_all / total_all * 100:.1f} % of the Simoncelli values are within 1 % "
                    f"of the team's range with the periodic boundary and {mirror_all / total_all * 100:.1f} % "
                    f"with the mirror boundary; the periodic boundary matches {periodic_teams} of the "
                    f"{len(agreement)} teams at least as well. Share of values within 1 % of the range:",
                    "",
                    "| Team | Periodic (used) | Mirror |",
                    "|:-----|----------------:|-------:|",
                ]
            )
            for team, (periodic, mirrored, total) in agreement.items():
                lines.append(f"| {team} | {periodic / total * 100:.1f}% | {mirrored / total * 100:.1f}% |")
            lines.append("")

        for team, comp in self._team_comparisons.items():
            lines.append(f"### {team}")
            lines.append("")
            if not comp.mismatches:
                lines.append("No mismatches! ✅")
                lines.append("")
                continue

            # The largest errors relative to the range (a zero range first)
            largest = sorted(
                comp.mismatches,
                key=lambda mm: mm.error / mm.feature_range if mm.feature_range else float("inf"),
                reverse=True,
            )[:10]
            lines.append(
                f"{len(comp.mismatches)} of {comp.total_comparisons} values differ by more than 10 % "
                "of the range. The 10 largest, relative to the range:"
            )
            lines.append("")
            lines.append(
                "| Patient | Feature | Configuration | Pictologics Value | Team Value | Error | Range |"
            )
            lines.append(
                "|:--------|:--------|:-------------|----------:|-----------:|------:|------:|"
            )
            for mm in largest:
                config_name = FILTER_SHORT_NAMES.get(mm.filter_id, f"ID {mm.filter_id}")
                lines.append(
                    f"| {mm.patient_id} | {mm.feature_tag} | {config_name} | {mm.our_value:.4g} | "
                    f"{mm.team_value:.4g} | {mm.error:.4g} | {mm.feature_range:.4g} |"
                )
            lines.append("")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(lines))


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for Phase 3 validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="IBSI 2 Phase 3 Compliance Verification"
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--patient-limit", type=int, default=None)
    parser.add_argument("--workers", "-j", type=int, default=8)
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("IBSI 2 Phase 3 Compliance Verification")
    print("=" * 60)
    print()

    validator = IBSIPhase3Validator(args.data_dir, n_workers=args.workers)

    # Check for data
    patient_ids = validator.get_patient_ids()
    if not patient_ids:
        print("ERROR: No validation data found.")
        print(f"  Expected at: {validator.validation_dir}")
        print("  Run: python dev/IBSI2/download_ibsi2_data.py")
        return 1

    print(f"Found {len(patient_ids)} patients")
    print()

    # Run validation
    print("Running Phase 3 validation...")
    validator.run_all_patients(patient_limit=args.patient_limit, verbose=not args.quiet)
    print()

    # Compare with teams
    print("Comparing with team submissions...")
    validator.compare_with_teams()
    print()

    # Print summary
    summary = validator.get_summary()
    print("-" * 60)
    print("Summary:")
    for team, data in summary["team_results"].items():
        total = data["total"]
        overlap_pct = data["matches_1pct"] / total * 100 if total else 0.0
        print(
            f"  {team}: {overlap_pct:.1f}% within 1 % of the range "
            f"({data['matches_1pct']}/{total})"
        )
    print("-" * 60)

    # Save outputs
    if args.output:
        validator.generate_docs(args.output)
        print(f"\nDocumentation saved to: {args.output}")

    if args.baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Baseline saved to: {args.baseline}")

    problems = summary["scans_failed"] + summary["nan_values"] + summary["pipeline_errors"]
    return 1 if problems else 0


if __name__ == "__main__":
    exit(main())
