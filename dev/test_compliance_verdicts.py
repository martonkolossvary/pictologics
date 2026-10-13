"""Self-test of the IBSI gate verdicts and the dev tools. dev/ publishes only the IBSI
harness (see .gitignore), so CI runs this file too; the tests of private tools skip there.

Run: poetry run pytest dev/test_compliance_verdicts.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # as in tests/conftest.py: no compiled kernels needed

from dev import compliance_utils as cu  # noqa: E402
from dev import pre_push  # noqa: E402

GAP = next(iter(cu.KNOWN_GAPS["IBSI 1"]))


def report(current: dict, baseline: dict, values: tuple[str, ...] | None = ()) -> cu.ComplianceResult:
    changes = None if values is None else list(values)
    return cu.print_compliance_report("IBSI 1", {"features": current}, {"features": baseline}, value_changes=changes)


def test_pass_and_known_gaps_pass() -> None:
    result = report({"a": {"status": "PASS"}, GAP: {"status": "SKIP"}}, {"a": {"status": "PASS"}, GAP: {"status": "SKIP"}})
    assert result.passed and result.known_gaps == 1


@pytest.mark.parametrize("status", ["FAIL", "MISSING", "SKIP", "ERROR", "UNKNOWN"])
def test_any_change_away_from_pass_fails(status: str) -> None:
    assert not report({"a": {"status": status}}, {"a": {"status": "PASS"}}).passed


def test_a_lost_item_fails() -> None:
    assert not report({}, {"a": {"status": "PASS"}}).passed


def test_a_known_failure_in_the_baseline_still_fails() -> None:
    assert not report({"a": {"status": "FAIL"}}, {"a": {"status": "FAIL"}}).passed


def test_value_changes_and_a_missing_value_baseline_fail() -> None:
    same = {"a": {"status": "PASS"}}
    assert not report(same, same, values=("x: 1.0 -> 1.1",)).passed
    assert not report(same, same, values=None).passed


def test_compare_values() -> None:
    base = {"c": {"f": 1.0, "g": float("nan")}}
    assert cu.compare_values({"c": {"f": 1.0 + 1e-12, "g": float("nan")}}, base) == []
    assert cu.compare_values({"c": {"f": 1.1, "g": float("nan")}}, base) == ["c/f: 1.0 -> 1.1"]
    assert cu.compare_values({"c": {"f": 1.0}}, base) == ["c/g: lost"]
    assert cu.compare_values({"c": {"f": 1.0, "g": float("nan"), "h": 2.0}}, base) == ["c/h: new"]


def test_compare_maps(tmp_path: Path) -> None:
    path = tmp_path / "maps.npz"
    assert cu.compare_maps({"t": np.ones(3)}, path) is None
    cu.save_npz_baseline(path, {"t": np.ones(3, np.float32)})
    assert cu.compare_maps({"t": np.ones(3, np.float32)}, path) == []
    assert cu.compare_maps({"t": np.full(3, 1.001, np.float32)}, path)[0].startswith("t: largest difference")
    assert cu.compare_maps({"t": np.ones(4, np.float32)}, path) == ["t: shape (3,) -> (4,)"]
    assert cu.compare_maps({}, path) == ["t: lost"]


def test_saving_copies_the_old_baseline(tmp_path: Path) -> None:
    path = tmp_path / "base.json"
    cu.save_json_baseline(path, {"v": 1})
    with patch("dev.pipeline_config.BASELINE_BACKUP_DIR", tmp_path / "backups"):
        cu.save_json_baseline(path, {"v": 2})
    assert cu.load_json_baseline(path) == {"v": 2}
    (copy,) = (tmp_path / "backups").glob("*/base.json")
    assert cu.load_json_baseline(copy) == {"v": 1}


PHASE3 = {
    "agreement_rule": "rule",
    "scans_processed": 153,
    "scans_failed": 0,
    "nan_values": 0,
    "pipeline_errors": 0,
    "team_results": {"T": {"total": 10, "matches_1pct": 9, "matches_5pct": 10, "matches_10pct": 10}},
}


def phase3(**changes: object) -> dict:
    current = {**PHASE3, **changes}
    return current


def test_phase3_passes_as_in_the_baseline() -> None:
    assert cu.print_phase3_summary(phase3(), PHASE3, value_changes=[]).passed


@pytest.mark.parametrize(
    "changes",
    [
        {"scans_processed": 152},
        {"scans_failed": 1},
        {"nan_values": 3},
        {"pipeline_errors": 1},
        {"agreement_rule": "another rule"},
        {"team_results": {"T": {"total": 10, "matches_1pct": 8, "matches_5pct": 10, "matches_10pct": 10}}},
        {"team_results": {}},
    ],
)
def test_phase3_fails(changes: dict) -> None:
    assert not cu.print_phase3_summary(phase3(**changes), PHASE3, value_changes=[]).passed


def test_phase3_fails_without_a_baseline_or_on_value_changes() -> None:
    assert not cu.print_phase3_summary(phase3(), {}, value_changes=[]).passed
    assert not cu.print_phase3_summary(phase3(), PHASE3, value_changes=["x: lost"]).passed


def test_a_crash_fails_the_step() -> None:
    with (
        patch.object(pre_push, "_missing_data", return_value=None),
        patch("dev.verify_ibsi_compliance.TestIBSICompliance.get_detailed_results", side_effect=RuntimeError("boom")),
    ):
        result, _, checker = pre_push.run_ibsi_compliance()
    assert not result.passed and checker is None


def test_missing_data_fails_unless_allowed(tmp_path: Path) -> None:
    with patch.dict(pre_push.IBSI_DATA, {"IBSI 1": [tmp_path / "absent.nii.gz"]}):
        assert not pre_push.run_ibsi_compliance()[0].passed
        assert pre_push.run_ibsi_compliance(allow_missing_data=True)[0].passed


def test_no_update_after_a_failure() -> None:
    failed = cu.ComplianceResult(name="IBSI 1", passed=False)
    assert not pre_push._may_update(failed, accept_changes=False)
    assert pre_push._may_update(failed, accept_changes=True)


def test_redirect_caches_starts_empty_outside_the_project(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "numba").mkdir(parents=True)
    (cache / "numba" / "old.nbi").write_text("old")
    prefix = sys.pycache_prefix
    try:
        with patch.object(cu, "GATE_CACHE_DIR", cache), patch.dict(os.environ):
            cu.redirect_caches(clean=True)
            assert not (cache / "numba").exists()
            assert os.environ["NUMBA_CACHE_DIR"] == str(cache / "numba")
            assert os.environ["MYPY_CACHE_DIR"] == str(cache / "mypy")
            assert sys.pycache_prefix == str(cache / "pycache")
    finally:
        sys.pycache_prefix = prefix


def test_clear_caches_deletes_only_caches(tmp_path: Path) -> None:
    for rel in ("pictologics/__pycache__/a.pyc", "tests/sub/__pycache__/b.pyc", ".mypy_cache/x", "coverage.xml",
                ".coverage.host", "pictologics/keep.py", "dev/results/keep.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    with patch.object(pre_push, "PROJECT_ROOT", tmp_path):
        assert pre_push.clear_caches()
    left = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert left == ["dev/results/keep.json", "pictologics/keep.py"]


def test_changelog_links_follow_the_release_headings() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    tpb = pytest.importorskip("towncrier_post_build")  # private: not in CI

    content = "## [0.6.0] - x\n## [0.5.1] - x\n## [0.4.10] - x\n## [0.4.9] - x\n"
    assert tpb.comparison_links(content) == [
        f"[0.6.0]: {tpb.REPO_URL}/compare/v0.5.1...v0.6.0",
        f"[0.5.1]: {tpb.REPO_URL}/compare/v0.4.10...v0.5.1",
        f"[0.4.10]: {tpb.REPO_URL}/compare/v0.4.9...v0.4.10",
        f"[0.4.9]: {tpb.REPO_URL}/releases/tag/v0.4.9",
    ]


def test_skipped_benchmarks_keep_the_readme_block() -> None:
    generate_docs = pytest.importorskip("dev.generate_docs")  # private: not in CI

    block = generate_docs._readme_benchmark_block()
    readme = (generate_docs.PROJECT_ROOT / "README.md").read_text()
    assert block.lstrip("\n").startswith("## Performance Benchmarks")
    assert "## Quality & Compliance" not in block
    assert "```\n\n" + block + "\n\n## Quality & Compliance" in readme



def test_download_maps_the_upstream_layout(tmp_path: Path) -> None:
    from dev.IBSI2 import download_ibsi2_data as d

    ds = tmp_path / "data_sets"
    upstream = [
        "ibsi_2_digital_phantom/nifti/impulse/image/impulse.nii.gz",
        "ibsi_2_ct_radiomics_phantom/nifti/image/phantom.nii.gz",
        "ibsi_validation/nifti/STS_001/MR_T1_mask.nii.gz",
    ]
    for rel in upstream:
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        (ds / rel).write_text(rel)
    files = d.plan({"data_sets": ds})
    assert files["data/phantoms/impulse_response.nii.gz"] == ds / upstream[0]
    assert files["data/ct_phantom/image.nii.gz"] == ds / upstream[1]
    assert files["data/validation/masks/STS_001_MRI_mask.nii.gz"] == ds / upstream[2]


def test_download_verify_finds_missing_and_changed_files(tmp_path: Path) -> None:
    from dev.IBSI2 import download_ibsi2_data as d

    (tmp_path / "a.csv").write_text("a")
    (tmp_path / "b.csv").write_text("b")
    manifest = tmp_path / "manifest.json"
    with patch.object(d, "MANIFEST", manifest), patch.object(d, "BACKUP_DIR", tmp_path / "backups"):
        d.write_manifest({"a.csv": tmp_path / "a.csv", "b.csv": tmp_path / "b.csv"}, tmp_path)
        assert d.verify(tmp_path) == []
        (tmp_path / "a.csv").write_text("changed")
        (tmp_path / "b.csv").unlink()
        assert d.verify(tmp_path) == ["a.csv: differs from the manifest", "b.csv: missing"]


def test_the_numba_build_result_decides_its_step() -> None:
    class Build:
        def __init__(self, code: int) -> None:
            self.returncode = code

        def communicate(self) -> tuple[str, None]:
            return "RuntimeWarning: Pictologics JIT warmup failed", None

    assert pre_push.finish_numba_build(Build(0))
    assert not pre_push.finish_numba_build(Build(1))


def test_value_counts_show_what_stayed_the_same(tmp_path: Path) -> None:
    stats: dict[str, float] = {}
    current = {"c": {"f": 1.0, "g": 2.0 + 1e-15, "h": float("nan")}}
    assert cu.compare_values(current, {"c": {"f": 1.0, "g": 2.0, "h": float("nan")}}, stats=stats) == []
    assert stats["n"] == 3 and stats["same"] == 2 and 0 < stats["max_rel"] < 1e-14
    path = tmp_path / "maps.npz"
    cu.save_npz_baseline(path, {"t": np.array([1.0, 2.0, np.nan])})
    maps: dict[str, float] = {}
    assert cu.compare_maps({"t": np.array([1.0, 2.5, np.nan])}, path, stats=maps)
    assert maps["n"] == 3 and maps["same"] == 2 and maps["max_rel"] == 0.25


def test_values_of_another_environment_are_for_review_only() -> None:
    # The pass counts are the requirement: in another environment (other packages or another
    # platform, as in CI) moved values and a missing value baseline are listed, not failed
    same = {"a": {"status": "PASS"}}
    other = "scipy 1.17.0 -> 1.18.1"
    moved = cu.print_compliance_report("IBSI 1", {"features": same}, {"features": same}, value_changes=["x: 1.0 -> 1.1"], other=other)
    assert moved.passed and moved.new_failures == 0
    assert cu.print_compliance_report("IBSI 1", {"features": same}, {"features": same}, value_changes=None, other=other).passed
    failed = cu.print_compliance_report("IBSI 1", {"features": {"a": {"status": "FAIL"}}}, {"features": same}, value_changes=[], other=other)
    assert not failed.passed


def test_other_environment_names_what_differs() -> None:
    now = cu.environment()
    assert cu.other_environment({"environment": now}) == ""
    assert cu.other_environment({}) == "the baseline has no environment record"
    older = {**now, "scipy": "0.1"}
    assert cu.other_environment({"environment": older}) == f"scipy 0.1 -> {now['scipy']}"


def test_phase1_margin_problems() -> None:
    tests = {
        "a": {"status": "PASS", "error_pct": 0.6},
        "b": {"status": "PASS", "error_pct": 0.5},
        "c": {"status": "FAIL", "error_pct": 3.0},
    }
    assert cu.phase1_margin_problems({"tests": tests}) == ["a: error 0.600 % (margin 0.5 %)"]


@pytest.mark.parametrize(("error", "passes"), [(0.4, True), (0.7, False)])
def test_a_phase1_map_above_the_margin_fails_the_step(error: float, passes: bool) -> None:
    class Validator:
        def __init__(self, *args: object) -> None:
            pass

        def run_all_tests(self, verbose: bool) -> None:
            pass

        def get_results_dict(self) -> dict:
            return {"tests": {"x": {"status": "PASS", "error_pct": error, "time_ms": 1.0}}, "passed": 1}

        def maps(self) -> dict:
            return {}

    baseline = {"tests": {"x": {"status": "PASS"}}, "environment": cu.environment()}
    with (
        patch("dev.IBSI2.verify_ibsi2_compliance.IBSIPhase1Validator", Validator),
        patch.object(pre_push, "_missing_data", return_value=None),
        patch.object(pre_push, "load_json_baseline", return_value=baseline),
        patch.object(pre_push, "compare_maps", return_value=[]),
    ):
        result = pre_push.run_ibsi2_compliance()[0]
    assert result.passed is passes and result.new_failures == (0 if passes else 1)


def test_phase3_allows_a_small_drop_of_the_matches() -> None:
    # 0.1 % of 2,000 comparisons: 2 matches may go at each level; a lost comparison fails
    base = {**PHASE3, "team_results": {"T": {"total": 2000, "matches_1pct": 1900, "matches_5pct": 1990, "matches_10pct": 2000}}}

    def team(**levels: int) -> dict:
        return phase3(team_results={"T": {**base["team_results"]["T"], **levels}})

    assert cu.print_phase3_summary(team(matches_1pct=1898, matches_10pct=1998), base, value_changes=[]).passed
    assert not cu.print_phase3_summary(team(matches_5pct=1987), base, value_changes=[]).passed
    assert not cu.print_phase3_summary(team(total=1999), base, value_changes=[]).passed


def test_phase3_values_of_another_environment_are_for_review_only() -> None:
    assert cu.print_phase3_summary(phase3(), PHASE3, value_changes=["x: 1.0 -> 1.1"], other="another platform").passed


def test_download_puts_the_ibsi1_phantoms_next_to_the_output(tmp_path: Path) -> None:
    from dev.IBSI2 import download_ibsi2_data as d

    ds = tmp_path / "data_sets"
    for upstream in d.IBSI1_FILES.values():
        (ds / upstream).parent.mkdir(parents=True, exist_ok=True)
        (ds / upstream).write_text(upstream)
    files = d.plan({"data_sets": ds})
    assert files["IBSI1/data/CT_mask.nii.gz"] == ds / "ibsi_1_ct_radiomics_phantom/nifti/mask/mask.nii.gz"
    out = tmp_path / "dev" / "IBSI2"
    assert d.target(out, "IBSI1/data/mask.nii.gz") == tmp_path / "dev" / "IBSI1" / "data" / "mask.nii.gz"
    assert d.target(out, "data/phantoms/noise.nii.gz") == out / "data" / "phantoms" / "noise.nii.gz"


def test_compliance_only_runs_the_numba_build_and_the_ibsi_checks() -> None:
    with patch.object(sys, "argv", ["pre_push.py", "--compliance-only", "--skip-compliance"]):
        args = pre_push.parse_args()
        assert args.compliance_only
        build = type("Build", (), {"returncode": 0, "communicate": lambda self: ("", None)})()
        with (
            patch.object(pre_push, "redirect_caches"),
            patch.object(pre_push, "start_numba_build", return_value=build),
            patch.object(pre_push, "run_ruff") as ruff,
            patch.object(pre_push, "run_pytest") as pytest_run,
            patch.object(pre_push, "clear_caches") as clear,
            patch.object(cu, "SHOW_TIMES", True),
        ):
            assert pre_push.main() == 0
            assert not cu.SHOW_TIMES
    assert not (ruff.called or pytest_run.called or clear.called)


def test_compliance_only_hides_the_times(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(cu, "SHOW_TIMES", False):
        result = report({"a": {"status": "PASS"}}, {"a": {"status": "PASS"}})
        cu.print_phase3_summary(phase3(), PHASE3, value_changes=[])
        cu.print_ibsi_summary([result])
    out = capsys.readouterr().out
    assert "Overall" in out
    assert not any(text in out for text in ("Time Regressions", "Wall time", "Performance Summary"))

