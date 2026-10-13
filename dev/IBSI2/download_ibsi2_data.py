#!/usr/bin/env python3
"""
Download IBSI 2 Data and Documentation

This script downloads all necessary data for IBSI 2 compliance testing:
1. Reference manual PDF from arXiv (version 9)
2. Digital phantoms for Phase 1 validation
3. CT radiomics phantom for Phase 2 validation
4. Reference response maps and feature values
5. Phase 3 validation dataset (51 patients, CT/PET/MRI)
6. Phase 3 team submissions for comparison
7. The IBSI 1 digital phantom and CT phantom (into dev/IBSI1/data)

Each repository is fetched at a pinned commit into a temporary folder outside OneDrive.
Each installed data file is checked against the SHA-256 manifest (ibsi2_data_manifest.json);
the manual PDF is not in it, as arXiv may rebuild its PDF files.
A file that is already correct is not copied again. A file that differs is moved to a
dated folder outside OneDrive before the new copy. Any failure gives exit code 1.

Usage:
    python dev/IBSI2/download_ibsi2_data.py [--output-dir PATH] [--keep-temp] [--no-manual]
    python dev/IBSI2/download_ibsi2_data.py --verify-only     # check the files, no download
    python dev/IBSI2/download_ibsi2_data.py --write-manifest  # after a new pin: save the hashes

Sources:
    - Reference Manual: https://arxiv.org/abs/2006.05470v9
    - Data Sets: https://github.com/theibsi/data_sets
    - Reference Data: https://github.com/theibsi/ibsi_2_reference_data
    - Data Analysis: https://github.com/theibsi/ibsi_2_data_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dev.compliance_utils import PHASE3_TEAMS  # noqa: E402

ARXIV_PDF_URL = "https://arxiv.org/pdf/2006.05470v9"
# The heads on 2026-09-27 (the last pushes were in 2023 and 2024).
REPOSITORIES = {
    "data_sets": (
        "https://github.com/theibsi/data_sets.git",
        "6da96021bc91faf4c0cb7fd7fa56a4225d2064a8",
    ),
    "ibsi_2_reference_data": (
        "https://github.com/theibsi/ibsi_2_reference_data.git",
        "5404579fb3e0d17e8db421f0e82d64ce2432ec03",
    ),
    "ibsi_2_data_analysis": (
        "https://github.com/theibsi/ibsi_2_data_analysis.git",
        "fde70ca61819799f673c07dca776deea065d2a7d",
    ),
}
MANIFEST = Path(__file__).with_name("ibsi2_data_manifest.json")
CACHES = Path.home() / "Library" / "Caches"
BACKUP_DIR = CACHES / "pictologics-baseline-backups"
# The IBSI 1 phantoms: local name in dev/IBSI1/data -> path in data_sets
IBSI1_FILES = {
    "phantom.nii.gz": "ibsi_1_digital_phantom/nifti/image/phantom.nii.gz",
    "mask.nii.gz": "ibsi_1_digital_phantom/nifti/mask/mask.nii.gz",
    "CT_phantom.nii.gz": "ibsi_1_ct_radiomics_phantom/nifti/image/phantom.nii.gz",
    "CT_mask.nii.gz": "ibsi_1_ct_radiomics_phantom/nifti/mask/mask.nii.gz",
}

# Digital phantoms (Phase 1): local name -> name in data_sets/ibsi_2_digital_phantom/nifti
PHASE1_PHANTOMS = {
    "empty": "empty",
    "impulse_response": "impulse",
    "checkerboard": "checkerboard",
    "noise": "noise",
    "sphere": "sphere",
    "pattern_1": "pattern_1",
    "pattern_2": "pattern_2",
    "pattern_3": "pattern_3",
    "orientation": "orientation",
}
# Phase 3 file prefix -> (image folder, mask name)
MODALITIES = {"CT": ("ct", "CT"), "PET": ("pet", "PET"), "MR_T1": ("mri", "MRI")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo: str, url: str, commit: str, temp: Path) -> Path | None:
    """Fetch one commit of a repository, without its history."""
    dest = temp / repo
    print(f"  Fetching {repo} at {commit[:12]}...")
    try:
        for cmd in (
            ["git", "init", "-q", str(dest)],
            ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", url, commit],
            ["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to fetch {repo}: {e.stderr.strip()}")
        return None
    return dest


def target(out: Path, rel: str) -> Path:
    """Where a planned file goes: IBSI1/... next to the output folder (dev/IBSI1), the
    others into it."""
    return out.parent / rel if rel.startswith("IBSI1/") else out / rel


def plan(sources: dict[str, Path]) -> dict[str, Path]:
    """The files to install (paths relative to the output folder, see target) and their
    sources."""
    files: dict[str, Path] = {}
    if "data_sets" in sources:
        data_sets = sources["data_sets"]
        for local, upstream in IBSI1_FILES.items():
            files[f"IBSI1/data/{local}"] = data_sets / upstream
        phantoms = data_sets / "ibsi_2_digital_phantom" / "nifti"
        for local, upstream in PHASE1_PHANTOMS.items():
            files[f"data/phantoms/{local}.nii.gz"] = phantoms / upstream / "image" / f"{upstream}.nii.gz"
        ct_phantom = data_sets / "ibsi_2_ct_radiomics_phantom" / "nifti"
        files["data/ct_phantom/image.nii.gz"] = ct_phantom / "image" / "phantom.nii.gz"
        files["data/ct_phantom/mask.nii.gz"] = ct_phantom / "mask" / "mask.nii.gz"
        for patient in sorted((data_sets / "ibsi_validation" / "nifti").glob("STS_*")):
            for prefix, (folder, mask_name) in MODALITIES.items():
                image = f"data/validation/{folder}/{patient.name}_image.nii.gz"
                mask = f"data/validation/masks/{patient.name}_{mask_name}_mask.nii.gz"
                files[image] = patient / f"{prefix}_image.nii.gz"
                files[mask] = patient / f"{prefix}_mask.nii.gz"
    if "ibsi_2_reference_data" in sources:
        reference = sources["ibsi_2_reference_data"]
        maps = reference / "reference_response_maps"
        for path in sorted(p for p in maps.rglob("*") if p.is_file()):
            files[f"references/response_maps/{path.relative_to(maps)}"] = path
        for path in sorted(p for p in (reference / "reference_feature_values").iterdir() if p.is_file()):
            files[f"references/feature_values/{path.name}"] = path
    if "ibsi_2_data_analysis" in sources:
        phase3 = sources["ibsi_2_data_analysis"] / "results" / "phase_3"
        for team in PHASE3_TEAMS:
            for path in sorted(p for p in (phase3 / team).rglob("*") if p.is_file()):
                files[f"references/phase3_submissions/{team}/{path.relative_to(phase3 / team)}"] = path
    return files


def install(files: dict[str, Path], out: Path) -> list[str]:
    """Copy the files. Skip a file that is already the same; move a different one aside."""
    problems: list[str] = []
    copied = 0
    backup = BACKUP_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-ibsi2-data"
    for rel, src in files.items():
        dst = target(out, rel)
        if not src.is_file():
            problems.append(f"{rel}: not in the download")
            continue
        if dst.is_file() and sha256(dst) == sha256(src):
            continue
        if dst.exists():
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(dst, backup / rel)
            print(f"  Moved the old {rel} to {backup}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"  ✓ {copied} files copied, {len(files) - copied - len(problems)} already up to date")
    return problems


def verify(out: Path) -> list[str]:
    """Compare the installed files with the manifest."""
    if not MANIFEST.exists():
        return [f"no manifest ({MANIFEST.name}); create it with --write-manifest"]
    problems = []
    for rel, digest in json.loads(MANIFEST.read_text())["files"].items():
        path = target(out, rel)
        if not path.is_file():
            problems.append(f"{rel}: missing")
        elif sha256(path) != digest:
            problems.append(f"{rel}: differs from the manifest")
    return problems


def write_manifest(files: dict[str, Path], out: Path) -> None:
    if MANIFEST.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MANIFEST, BACKUP_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{MANIFEST.name}")
    manifest = {
        "arxiv": ARXIV_PDF_URL,
        "commits": {repo: commit for repo, (_, commit) in REPOSITORIES.items()},
        "files": {rel: sha256(target(out, rel)) for rel in sorted(files) if not rel.startswith("documentation/")},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"  ✓ Wrote {MANIFEST.name} ({len(manifest['files'])} files)")


def main(args: argparse.Namespace) -> int:
    out = (args.output_dir or Path(__file__).parent).resolve()
    print("=" * 60)
    print("IBSI 2 Data Download Script")
    print("=" * 60)
    print(f"\nOutput directory: {out}\n")

    problems: list[str] = []
    if not args.verify_only:
        # outside OneDrive on a Mac; the system's temporary folder elsewhere (CI)
        temp = Path(tempfile.mkdtemp(dir=CACHES if CACHES.is_dir() else None, prefix="ibsi2-download-"))
        try:
            sources = {}
            for repo, (url, commit) in REPOSITORIES.items():
                path = fetch(repo, url, commit, temp)
                if path is None:
                    problems.append(f"{repo}: fetch failed")
                else:
                    sources[repo] = path
            files = plan(sources)
            if not args.no_manual:
                print("  Downloading the reference manual...")
                # certifi's certificates: the python.org Python has no certificate store
                context = ssl.create_default_context(cafile=certifi.where())
                try:
                    with urllib.request.urlopen(ARXIV_PDF_URL, context=context) as response:
                        (temp / "manual.pdf").write_bytes(response.read())
                    files["documentation/ibsi_2_reference_manual.pdf"] = temp / "manual.pdf"
                except OSError as e:
                    problems.append(f"reference manual: {e}")
            problems += install(files, out)
            if args.write_manifest and not problems:
                write_manifest(files, out)
        finally:
            if args.keep_temp:
                print(f"  Keeping the temporary files at {temp}")
            else:
                shutil.rmtree(temp, ignore_errors=True)

    problems += verify(out)
    print()
    if problems:
        print(f"✗ {len(problems)} problems:")
        for problem in problems[:20]:
            print(f"  {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1
    print("✓ All files match the manifest.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IBSI 2 data and documentation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as this script)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary download folder (outside OneDrive)",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Check the files against the manifest only"
    )
    parser.add_argument(
        "--no-manual",
        action="store_true",
        help="Do not download the reference manual PDF (CI needs the data only)",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Save the SHA-256 of the installed files as the manifest (after a new pin)",
    )
    sys.exit(main(parser.parse_args()))
