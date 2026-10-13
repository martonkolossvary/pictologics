import json
import os
import pprint
import sys
import textwrap
import time
import unittest
from typing import Any, Dict, cast

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dev.compliance_utils import WORKERS, get_rounding_decimals  # noqa: E402
from pictologics.pipeline import RadiomicsPipeline

# Feature code to name mapping
CODE_TO_NAME = {
    # Morphological features -----
    "RNU0": "Volume",
    "YEKZ": "Volume voxel counting",
    "C0JK": "Surface area",
    "2PR5": "Surface to volume ratio",
    "SKGS": "Compactness 1",
    "BQWJ": "Compactness 2",
    "KRCK": "Spherical disproportion",
    "QCFX": "Sphericity",
    "25C7": "Asphericity",
    "KLMA": "Center of mass shift",
    "L0JK": "Maximum 3D diameter",
    "TDIC": "Major axis length",
    "P9VJ": "Minor axis length",
    "7J51": "Least axis length",
    "Q3CK": "Elongation",
    "N17B": "Flatness",
    "PBX1": "Volume density (AABB)",
    "R59B": "Area density (AABB)",
    "ZH1A": "Volume density (OMBB)",
    "IQYR": "Area density (OMBB)",
    "6BDE": "Volume density (AEE)",
    "RDD2": "Area density (AEE)",
    "SWZ1": "Volume density (MVEE)",
    "BRI8": "Area density (MVEE)",
    "R3ER": "Volume density (convex hull)",
    "7T7F": "Area density (convex hull)",
    "99N0": "Integrated intensity",
    "N365": "Moran's I index",
    "NPT7": "Geary's C measure",
    # Local intensity features -----
    "VJGA": "Local intensity peak ",
    "0F91": "Global intensity peak",
    # Intensity-based statistical features -----
    "Q4LE": "Mean intensity",
    "ECT3": "Intensity variance",
    "KE2A": "Intensity skewness",
    "IPH6": "Intensity kurtosis",
    "Y12H": "Median intensity",
    "1GSF": "Minimum intensity",
    "QG58": "10th intensity percentile",
    "8DWT": "90th intensity percentile",
    "84IY": "Maximum intensity",
    "SALO": "Intensity interquartile range",
    "2OJQ": "Intensity range",
    "4FUA": "Intensity Mean absolute deviation",
    "1128": "Intensity Robust mean absolute deviation",
    "N72L": "Intensity Median absolute deviation",
    "7TET": "Intensity Coefficient of variation",
    "9S40": "Intensity Quartile coefficient of dispersion",
    "N8CA": "Intensity energy",
    "5ZWQ": "Root mean square intensity",
    # Intensity histogram features -----
    "X6K6": "Mean discretised intensity",
    "CH89": "Discretised intensity variance",
    "88K1": "Discretised intensity skewness",
    "C3I7": "Discretised intensity kurtosis",
    "WIFQ": "Median discretised intensity",
    "1PR8": "Minimum discretised intensity",
    "1PR": "10th discretised intensity percentile",
    "GPMT": "90th discretised intensity percentile",
    "3NCY": "Maximum discretised intensity",
    "AMMC": "Intensity histogram mode",
    "WR0O": "Discretised intensity interquartile range",
    "5Z3W": "Discretised intensity range",
    "D2ZX": "Intensity histogram mean absolute deviation",
    "WRZB": "Intensity histogram robust mean absolute deviation",
    "4RNL": "Intensity histogram median absolute deviation",
    "CWYJ": "Intensity histogram coefficient of variation",
    "SLWD": "Intensity histogram quartile coefficient of dispersion",
    "TLU2": "Discretised intensity entropy",
    "BJ5W": "Discretised intensity uniformity",
    "12CE": "Maximum histogram gradient",
    "8E6O": "Maximum histogram gradient intensity",
    "VQB3": "Minimum histogram gradient",
    "RHQZ": "Minimum histogram gradient intensity",
    # Intensity-volume histogram features -----
    "BC2M_10": "Volume at intensity fraction 0.10",
    "BC2M_90": "Volume at intensity fraction 0.90",
    "GBPN_10": "Intensity at volume fraction 0.10",
    "GBPN_90": "Intensity at volume fraction 0.90",
    "DDTU": "Volume fraction difference between intensity 0.10 and 0.90 fractions",
    "CNV2": "Intensity fraction difference between volume 0.10 and 0.90 fractions",
    "9CMM": "Area under the IVH curve",
    # Grey level co-occurrence based features -----
    "GYBY": "Joint maximum",
    "60VM": "Joint average",
    "UR99": "Joint variance",
    "TU9B": "Joint entropy",
    "TF7R": "Difference average",
    "D3YU": "Difference variance",
    "NTRS": "Difference entropy",
    "ZGXS": "Sum average",
    "OEEB": "Sum variance",
    "P6QZ": "Sum entropy",
    "8ZQL": "Angular second moment",
    "ACUI": "Contrast",
    "8S9J": "Dissimilarity",
    "IB1Z": "Inverse difference",
    "NDRX": "Normalised inverse difference",
    "WF0Z": "Inverse difference moment",
    "1QCO": "Normalised inverse difference moment",
    "E8JP": "Inverse variance",
    "NI2N": "Correlation",
    "QWB0": "Autocorrelation",
    "DG8W": "Cluster tendency",
    "7NFM": "Cluster shade",
    "AE86": "Cluster prominence",
    "R8DG": "Information correlation 1",
    "JN9H": "Information correlation 2",
    # Grey level run length based features -----
    "22OV": "Short runs emphasis",
    "W4KF": "Long runs emphasis",
    "V3SW": "Low grey level run emphasis",
    "G3QZ": "High grey level run emphasis",
    "HTZT": "Short run low grey level emphasis",
    "GD3A": "Short run high grey level emphasis",
    "IVPO": "Long run low grey level emphasis",
    "3KUM": "Long run high grey level emphasis",
    "R5YN": "Grey level non-uniformity",
    "OVBL": "Normalised grey level non-uniformity",
    "W92Y": "Run length non-uniformity",
    "IC23": "Normalised run length non-uniformity",
    "9ZK5": "Run percentage",
    "8CE5": "Grey level variance",
    "SXLW": "Run length variance",
    "HJ9O": "Run entropy",
    # Grey level size zone based features -----
    "P001": "Small zone emphasis",
    "48P8": "Large zone emphasis",
    "XMSY": "Low grey level zone emphasis",
    "5GN9": "High grey level zone emphasis",
    "5RAI": "Small zone low grey level emphasis",
    "HW1V": "Small zone high grey level emphasis",
    "YH51": "Large zone low grey level emphasis",
    "J17V": "Large zone high grey level emphasis",
    "JNSA": "Grey level non-uniformity",
    "Y1RO": "Normalised grey level non-uniformity",
    "4JP3": "Zone size non-uniformity",
    "VB3A": "Normalised zone size non-uniformity",
    "P30P": "Zone percentage",
    "BYLV": "Grey level variance",
    "3NSA": "Zone size variance",
    "GU8N": "Zone size entropy",
    # Grey level distance zone based features -----
    "0GBI": "Small distance emphasis",
    "MB4I": "Large distance emphasis",
    "S1RA": "Low grey level zone emphasis",
    "K26C": "High grey level zone emphasis",
    "RUVG": "Small distance low grey level emphasis",
    "DKNJ": "Small distance high grey level emphasis",
    "A7WM": "Large distance low grey level emphasis",
    "KLTH": "Large distance high grey level emphasis",
    "VFT7": "Grey level non-uniformity",
    "7HP3": "Normalised grey level non-uniformity",
    "V294": "Zone distance non-uniformity",
    "IATH": "Normalised zone distance non-uniformity",
    "VIWW": "Zone percentage",
    "QK93": "Grey level variance",
    "7WT1": "Zone distance variance",
    "GBDU": "Zone distance entropy",
    # Neighbourhood grey tone difference based features ----
    "QCDE": "Coarseness",
    "65HE": "Contrast",
    "NQ30": "Busyness",
    "HDEZ": "Complexity",
    "1X9X": "Strength",
    # Neighbouring grey level dependence based features ----
    "SODN": "Low dependence emphasis",
    "IMOQ": "High dependence emphasis",
    "TL9H": "Low grey level count emphasis",
    "OAE7": "High grey level count emphasis",
    "EQ3F": "Low dependence low grey level emphasis",
    "JA6D": "Low dependence high grey level emphasis",
    "NBZI": "High dependence low grey level emphasis",
    "9QMG": "High dependence high grey level emphasis",
    "FP8K": "Grey level non-uniformity",
    "5SPA": "Normalised grey level non-uniformity",
    "Z87G": "Dependence count non-uniformity",
    "OKJI": "Normalised dependence count non-uniformity",
    "6XV8": "Dependence count percentage",
    "1PFV": "Grey level variance",
    "DNX2": "Dependence count variance",
    "FCBV": "Dependence count entropy",
    "CAS9": "Dependence count energy",
}

FEATURE_FAMILIES = {
    "Morphology": [
        "RNU0",
        "YEKZ",
        "C0JK",
        "2PR5",
        "SKGS",
        "BQWJ",
        "KRCK",
        "QCFX",
        "25C7",
        "KLMA",
        "L0JK",
        "TDIC",
        "P9VJ",
        "7J51",
        "Q3CK",
        "N17B",
        "PBX1",
        "R59B",
        "ZH1A",
        "IQYR",
        "6BDE",
        "RDD2",
        "SWZ1",
        "BRI8",
        "R3ER",
        "7T7F",
    ],
    "Intensity": [
        "99N0",
        "N365",
        "NPT7",
        "VJGA",
        "0F91",
        "Q4LE",
        "ECT3",
        "KE2A",
        "IPH6",
        "Y12H",
        "1GSF",
        "QG58",
        "8DWT",
        "84IY",
        "SALO",
        "2OJQ",
        "4FUA",
        "1128",
        "N72L",
        "7TET",
        "9S40",
        "N8CA",
        "5ZWQ",
    ],
    "Intensity Histogram": [
        "X6K6",
        "CH89",
        "88K1",
        "C3I7",
        "WIFQ",
        "1PR8",
        "1PR",
        "GPMT",
        "3NCY",
        "AMMC",
        "WR0O",
        "5Z3W",
        "D2ZX",
        "WRZB",
        "4RNL",
        "CWYJ",
        "SLWD",
        "TLU2",
        "BJ5W",
        "12CE",
        "8E6O",
        "VQB3",
        "RHQZ",
    ],
    "Intensity-Volume Histogram": [
        "BC2M_10",
        "BC2M_90",
        "GBPN_10",
        "GBPN_90",
        "DDTU",
        "CNV2",
        "9CMM",
    ],
    "GLCM": [
        "GYBY",
        "60VM",
        "UR99",
        "TU9B",
        "TF7R",
        "D3YU",
        "NTRS",
        "ZGXS",
        "OEEB",
        "P6QZ",
        "8ZQL",
        "ACUI",
        "8S9J",
        "IB1Z",
        "NDRX",
        "WF0Z",
        "1QCO",
        "E8JP",
        "NI2N",
        "QWB0",
        "DG8W",
        "7NFM",
        "AE86",
        "R8DG",
        "JN9H",
    ],
    "GLRLM": [
        "22OV",
        "W4KF",
        "V3SW",
        "G3QZ",
        "HTZT",
        "GD3A",
        "IVPO",
        "3KUM",
        "R5YN",
        "OVBL",
        "W92Y",
        "IC23",
        "9ZK5",
        "8CE5",
        "SXLW",
        "HJ9O",
    ],
    "GLSZM": [
        "P001",
        "48P8",
        "XMSY",
        "5GN9",
        "5RAI",
        "HW1V",
        "YH51",
        "J17V",
        "JNSA",
        "Y1RO",
        "4JP3",
        "VB3A",
        "P30P",
        "BYLV",
        "3NSA",
        "GU8N",
    ],
    "GLDZM": [
        "0GBI",
        "MB4I",
        "S1RA",
        "K26C",
        "RUVG",
        "DKNJ",
        "A7WM",
        "KLTH",
        "VFT7",
        "7HP3",
        "V294",
        "IATH",
        "VIWW",
        "QK93",
        "7WT1",
        "GBDU",
    ],
    "NGTDM": ["QCDE", "65HE", "NQ30", "HDEZ", "1X9X"],
    "NGLDM": [
        "SODN",
        "IMOQ",
        "TL9H",
        "OAE7",
        "EQ3F",
        "JA6D",
        "NBZI",
        "9QMG",
        "FP8K",
        "5SPA",
        "Z87G",
        "OKJI",
        "6XV8",
        "1PFV",
        "DNX2",
        "FCBV",
        "CAS9",
    ],
}

# =============================================================================
# IBSI Pipeline Configurations
# =============================================================================
# Each IBSI configuration is defined as a sequence of pipeline steps.
# The pipeline executes steps linearly in order, allowing complex workflows
# including repeated steps if needed.

IBSI_DIGITAL_PHANTOM_CONFIG = [
    # Digital Phantom: FBS 1.0 discretisation from 1 (its grey levels are its values),
    # no resampling
    {"step": "discretise", "params": {"method": "FBS", "bin_width": 1.0, "min_val": 1.0}},
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True,
            "include_local_intensity": True,
        },
    },
]

IBSI_CONFIG_C = [
    # Config C: 2mm isotropic, linear + round, resegment [-1000, 400], FBS 25 HU
    {"step": "binarize_mask", "params": {"threshold": 0.5}},
    {
        "step": "resample",
        "params": {
            "new_spacing": (2.0, 2.0, 2.0),
            "interpolation": "linear",
            "round_intensities": True,
            "mask_interpolation": "linear",
            "mask_threshold": 0.5,
        },
    },
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400, "apply_to": "intensity"}},
    {
        "step": "discretise",
        "params": {"method": "FBS", "bin_width": 25.0, "min_val": -1000},
    },
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True,
            "include_local_intensity": True,
            # IVH uses different discretisation (FBS 2.5 HU)
            "ivh_discretisation": {"method": "FBS", "bin_width": 2.5, "min_val": -1000},
            "ivh_params": {"target_range_max": 400},
        },
    },
]

IBSI_CONFIG_D = [
    # Config D: 2mm isotropic, linear + round, 3σ outlier filter, FBN 32 bins, continuous IVH
    {"step": "binarize_mask", "params": {"threshold": 0.5}},
    {
        "step": "resample",
        "params": {
            "new_spacing": (2.0, 2.0, 2.0),
            "interpolation": "linear",
            "round_intensities": True,
            "mask_interpolation": "linear",
            "mask_threshold": 0.5,
        },
    },
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "filter_outliers", "params": {"sigma": 3.0, "apply_to": "intensity"}},
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True,
            "include_local_intensity": True,
            # Continuous IVH (no discretisation for IVH)
            "ivh_use_continuous": True,
        },
    },
]

IBSI_CONFIG_E = [
    # Config E: 2mm isotropic cubic, resegment + outlier, round after, FBN 32 for texture, FBN 1000 for IVH
    {"step": "binarize_mask", "params": {"threshold": 0.5}},
    {
        "step": "resample",
        "params": {
            "new_spacing": (2.0, 2.0, 2.0),
            "interpolation": "cubic",
            "round_intensities": False,  # Round later
            "mask_interpolation": "linear",
            "mask_threshold": 0.5,
        },
    },
    {"step": "keep_largest_component", "params": {"apply_to": "morph"}},
    {"step": "resegment", "params": {"range_min": -1000, "range_max": 400, "apply_to": "intensity"}},
    {"step": "filter_outliers", "params": {"sigma": 3.0, "apply_to": "intensity"}},
    {"step": "round_intensities", "params": {}},  # Round after all preprocessing
    {"step": "discretise", "params": {"method": "FBN", "n_bins": 32}},
    {
        "step": "extract_features",
        "params": {
            "families": ["intensity", "morphology", "texture", "histogram", "ivh"],
            "include_spatial_intensity": True,
            "include_local_intensity": True,
            # IVH uses FBN 1000 bins
            "ivh_discretisation": {"method": "FBN", "n_bins": 1000},
            "ivh_params": {"bin_width": 1.0, "min_val": 0.5, "max_val": 1000.5},
        },
    },
]


# =============================================================================
# Helper Functions (Standalone for Parallel Execution)
# =============================================================================


def run_single_config(
    config_name: str,
    image_path: str,
    mask_path: str,
    steps: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], float, float]:
    """Run a single IBSI configuration with JIT warmup excluded from timing.

    To ensure JIT compilation is not included in timing (the import already runs the
    package warm-up):
    1. Run the pipeline ONCE (untimed, traced for the peak memory) to trigger any
       remaining JIT compilation
    2. Run the pipeline AGAIN (timed, not traced) to measure actual execution time
    """
    import tracemalloc

    # Step 1: Create pipeline and run once (untimed) to trigger any JIT compilation
    pipeline = RadiomicsPipeline()
    pipeline.add_config(config_name, steps)

    # Warmup run - triggers JIT compilation for this specific pipeline configuration
    tracemalloc.start()
    _ = pipeline.run(image_path, mask_path, config_names=[config_name])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Step 2: Timed run (JIT functions now compiled)
    start_time = time.perf_counter()
    results = pipeline.run(image_path, mask_path, config_names=[config_name])
    elapsed = time.perf_counter() - start_time
    return config_name, dict(results[config_name].to_dict()), elapsed, peak / (1024 * 1024)


class TestIBSICompliance(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.base_dir, "IBSI1", "data")
        self.refs_dir = os.path.join(self.base_dir, "IBSI1", "references")

        self.digital_phantom_path = os.path.join(self.data_dir, "phantom.nii.gz")
        self.digital_mask_path = os.path.join(self.data_dir, "mask.nii.gz")
        self.ct_phantom_path = os.path.join(self.data_dir, "CT_phantom.nii.gz")
        self.ct_mask_path = os.path.join(self.data_dir, "CT_mask.nii.gz")

        # Configs are defined globally
        pass

    def load_refs(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.refs_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return cast(Dict[str, Any], json.load(f))
        return {}

    def run_all_configs(self) -> dict:
        """Run all configurations in parallel and return results."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [
            (
                "digital_phantom",
                self.digital_phantom_path,
                self.digital_mask_path,
                IBSI_DIGITAL_PHANTOM_CONFIG,
            ),
            ("config_c", self.ct_phantom_path, self.ct_mask_path, IBSI_CONFIG_C),
            ("config_d", self.ct_phantom_path, self.ct_mask_path, IBSI_CONFIG_D),
            ("config_e", self.ct_phantom_path, self.ct_mask_path, IBSI_CONFIG_E),
        ]

        self._cached_results = {}

        # Use 4 workers since we have 4 independent configs
        n_workers = min(4, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=WORKERS) as executor:
            futures = [
                executor.submit(run_single_config, name, img, msk, steps)
                for name, img, msk, steps in tasks
            ]

            for future in as_completed(futures):
                try:
                    name, res, t, m = future.result()
                    self._cached_results[name] = {
                        "results": res,
                        "time": t,
                        "memory": m,
                    }
                except Exception as e:
                    print(f"Error executing config: {e}")
                    # Re-raise or handle gracefully?
                    # If one fails, the whole compliance check fails basically.
                    raise e

        return self._cached_results

    def save_cache(self, path: str):
        """Save results to JSON cache."""
        if not hasattr(self, "_cached_results"):
            return

        with open(path, "w") as f:
            json.dump(self._cached_results, f, indent=2)

    def values(self) -> Dict[str, Dict[str, float]]:
        """Our feature values per configuration, for the gate's value check."""
        if not hasattr(self, "_cached_results"):
            self.run_all_configs()
        return {name: data["results"] for name, data in sorted(self._cached_results.items())}

    def load_cache(self, path: str) -> bool:
        """Load results from JSON cache."""
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r") as f:
                self._cached_results = json.load(f)
            return True
        except Exception:
            return False

    def generate_report(self) -> str:
        # Load all refs
        refs_digital = self.load_refs("refs_digital_phantom.json")
        refs_c = self.load_refs("refs_config_c.json")
        refs_d = self.load_refs("refs_config_d.json")
        refs_e = self.load_refs("refs_config_e.json")

        # Get results (either computed or cached)
        if not hasattr(self, "_cached_results"):
            self.run_all_configs()

        data = self._cached_results

        # Helper to extract parts
        def _get(key):
            d = data.get(key, {})
            return d.get("results", {}), d.get("time", 0), d.get("memory", 0)

        results_digital, _, _ = _get("digital_phantom")
        results_c, _, _ = _get("config_c")
        results_d, _, _ = _get("config_d")
        results_e, _, _ = _get("config_e")

        # Build Tables
        report_lines = []
        report_lines.append("# Comprehensive IBSI Benchmark Results")
        report_lines.append("")
        report_lines.append("## How to Run the Benchmarks")
        report_lines.append("")
        report_lines.append(
            "To reproduce these results or run the IBSI compliance benchmarks on your own system:"
        )
        report_lines.append("")
        report_lines.append("### 1. Download the Data")
        report_lines.append(
            "The IBSI reference datasets (Digital Phantom and Lung Cancer CT) are available on the [IBSI GitHub repository](https://github.com/theibsi/data_sets)."
        )
        report_lines.append("")
        report_lines.append(
            "- **Digital Phantom**: Download `phantom.nii.gz` and `mask.nii.gz` from the `digital_phantom` folder."
        )
        report_lines.append(
            "- **Lung Cancer CT**: Download the `PAT1` NIfTI files. Note that IBSI recommends converting these to at least 32-bit floating point and rounding to the nearest integer before processing."
        )
        report_lines.append("")
        report_lines.append(
            "Place these files in `dev/IBSI1/data/` or provide their paths to the script."
        )
        report_lines.append("")
        report_lines.append(
            "### 2. Run Configurations Programmatically using `RadiomicsPipeline`"
        )
        report_lines.append(
            "You can run any of the IBSI configurations programmatically using the `RadiomicsPipeline` class."
        )
        report_lines.append("")
        report_lines.append("```python")
        report_lines.append("from pictologics.pipeline import RadiomicsPipeline")
        report_lines.append("from pictologics.loader import load_image")
        report_lines.append("")
        report_lines.append("# 1. Initialize Pipeline")
        report_lines.append("pipeline = RadiomicsPipeline()")
        report_lines.append("")
        report_lines.append("# --- DEFINE CONFIGURATIONS ---")
        report_lines.append("")
        for label, name, config in (
            ("A. Digital Phantom Config (FBS 1.0, no resampling)", "config_digital_phantom", IBSI_DIGITAL_PHANTOM_CONFIG),
            ("B. Config C (2mm isotropic, resegment [-1000, 400], FBS 25 HU)", "config_c", IBSI_CONFIG_C),
            ("C. Config D (2mm isotropic, 3-sigma outlier, FBN 32, Continuous IVH)", "config_d", IBSI_CONFIG_D),
            ("D. Config E (Cubic resamp, 3-sigma, round last, FBN 32 for tex, FBN 1000 for IVH)", "config_e", IBSI_CONFIG_E),
        ):
            report_lines.append(f"# {label}")
            steps = ",\n".join(textwrap.indent(pprint.pformat(step, width=84, sort_dicts=False), "    ") for step in config)
            report_lines.append(f"{name} = [\n{steps},\n]")
            report_lines.append("")
        report_lines.append("# --- RUN PIPELINE ---")
        report_lines.append("# Choose which config to run:")
        report_lines.append('pipeline.add_config("ibsi_config_c", config_c)')
        report_lines.append("")
        report_lines.append("# 3. Load image and mask")
        report_lines.append("# For Config C/D/E (Lung Cancer CT)")
        report_lines.append('image = load_image("path/to/CT_image.nii.gz")')
        report_lines.append('mask = load_image("path/to/CT_mask.nii.gz")')
        report_lines.append("")
        report_lines.append("# 4. Run extraction")
        report_lines.append('results = pipeline.run(image, mask, config_names=["ibsi_config_c"])')
        report_lines.append("")
        report_lines.append("# 5. Access results")
        report_lines.append('print(results["ibsi_config_c"])')
        report_lines.append("```")
        report_lines.append("")
        report_lines.append(
            "Pictologics considers all data as 3D volumes (even 2D slices are converted to 3D volumes) and does not support 2D radiomic extraction mechanisms. Therefore, benchmark comparisons to configuration A and B are not possible."
        )
        report_lines.append("")

        configs = [
            ("Digital Phantom", results_digital, refs_digital),
            ("Config C", results_c, refs_c),
            ("Config D", results_d, refs_d),
            ("Config E", results_e, refs_e),
        ]

        for config_name, results, refs in configs:
            report_lines.append(f"## {config_name}")

            for family, codes in FEATURE_FAMILIES.items():
                report_lines.append(f"### {family}")
                report_lines.append("| Feature | Code | Calc | Ref | Tol | Status |")
                report_lines.append("|---|---|---|---|---|---|")

                for code in codes:
                    if code not in CODE_TO_NAME:
                        continue
                    feature_name = CODE_TO_NAME[code]

                    # Find calculated value
                    calc_val = None
                    for k, v in results.items():
                        if k.endswith(code):
                            calc_val = v
                            break

                    # Find reference value
                    ref_entry = refs.get(code)

                    if calc_val is None and not ref_entry:
                        continue

                    if ref_entry and ref_entry.get("value") not in ["—", "N/A"]:
                        decimals = get_rounding_decimals(str(ref_entry["value"]))
                        if decimals >= 0:
                            calc_str = (
                                f"{calc_val:.{decimals}f}"
                                if calc_val is not None
                                else "MISSING"
                            )
                        else:
                            # For negative decimals (e.g. 1.2e3 -> decimals=-2), use .4g fallback or int
                            calc_str = (
                                f"{calc_val:.4g}" if calc_val is not None else "MISSING"
                            )
                    else:
                        calc_str = (
                            f"{calc_val:.4g}" if calc_val is not None else "MISSING"
                        )

                    if ref_entry:
                        ref_val_raw = str(ref_entry["value"])
                        tol_raw = str(ref_entry["tol"])

                        if ref_val_raw == "—" or ref_val_raw == "N/A":
                            ref_str = "—"
                            tol_str = "—"
                            status = "REF."
                            icon = "❗"
                        else:
                            ref_str = ref_val_raw
                            tol_str = tol_raw

                            try:
                                ref_val = float(ref_val_raw)
                                tol = (
                                    float(tol_raw)
                                    if tol_raw != "—" and tol_raw != "N/A"
                                    else None
                                )

                                if calc_val is not None:
                                    decimals = get_rounding_decimals(ref_val_raw)
                                    calc_rounded = round(calc_val, decimals)

                                    if tol is not None:
                                        diff = abs(calc_rounded - ref_val)
                                        if round(diff, 10) <= tol:
                                            status = "PASS"
                                        else:
                                            status = "FAIL"
                                    else:
                                        if calc_rounded == ref_val:
                                            status = "PASS"
                                        else:
                                            status = "FAIL"
                                else:
                                    status = "FAIL"
                            except ValueError:
                                status = "ERROR"
                                icon = "⚠️"

                            if status == "PASS":
                                icon = "✅"
                            elif status == "FAIL":
                                icon = "❌"
                    else:
                        ref_str = "N/A"
                        tol_str = "N/A"
                        status = "EXTRA"
                        icon = "❓"

                    report_lines.append(
                        f"| {feature_name} | {code} | {calc_str} | {ref_str} | {tol_str} | {icon} {status} |"
                    )

                report_lines.append("")

        return "\n".join(report_lines)

    def get_detailed_results(self) -> tuple[Dict[str, Any], float, float]:
        """
        Get individual feature results with timing and memory.
        Returns: (results_dict, total_time_seconds, max_memory_mb)
        """
        # Load all refs
        refs_digital = self.load_refs("refs_digital_phantom.json")
        refs_c = self.load_refs("refs_config_c.json")
        refs_d = self.load_refs("refs_config_d.json")
        refs_e = self.load_refs("refs_config_e.json")

        # Reuse or run all configs (populates self._cached_results)
        if not hasattr(self, "_cached_results"):
            self.run_all_configs()

        data = self._cached_results

        # Extract results
        results_digital = data["digital_phantom"]["results"]
        t_dig = data["digital_phantom"]["time"]
        m_dig = data["digital_phantom"]["memory"]

        results_c = data["config_c"]["results"]
        t_c = data["config_c"]["time"]
        m_c = data["config_c"]["memory"]

        results_d = data["config_d"]["results"]
        t_d = data["config_d"]["time"]
        m_d = data["config_d"]["memory"]

        results_e = data["config_e"]["results"]
        t_e = data["config_e"]["time"]
        m_e = data["config_e"]["memory"]

        total_time = t_dig + t_c + t_d + t_e
        max_memory = max(m_dig, m_c, m_d, m_e)

        configs = [
            ("Digital_Phantom", results_digital, refs_digital),
            ("Config_C", results_c, refs_c),
            ("Config_D", results_d, refs_d),
            ("Config_E", results_e, refs_e),
        ]

        feature_results: Dict[str, Dict[str, Any]] = {}

        for config_name, results, refs in configs:
            for family, codes in FEATURE_FAMILIES.items():
                for code in codes:
                    if code not in CODE_TO_NAME:
                        continue

                    key = f"{config_name}:{family}:{code}"

                    # Assign a status
                    status = "UNKNOWN"

                    # Find calculated value
                    calc_val = None
                    for k, v in results.items():
                        if k.endswith(code):
                            calc_val = v
                            break

                    # Find reference value
                    ref_entry = refs.get(code)

                    if calc_val is None and not ref_entry:
                        continue

                    if not ref_entry:
                        status = "SKIP"
                    else:
                        ref_val_raw = str(ref_entry["value"])
                        tol_raw = str(ref_entry["tol"])

                        if ref_val_raw == "—" or ref_val_raw == "N/A":
                            status = "SKIP"
                        else:
                            try:
                                ref_val = float(ref_val_raw)
                                tol = (
                                    float(tol_raw)
                                    if tol_raw != "—" and tol_raw != "N/A"
                                    else None
                                )

                                if calc_val is not None:
                                    # Get rounding decimals
                                    decimals = get_rounding_decimals(ref_val_raw)
                                    calc_rounded = round(calc_val, decimals)

                                    if tol is not None:
                                        diff = abs(calc_rounded - ref_val)
                                        if round(diff, 10) <= tol:
                                            status = "PASS"
                                        else:
                                            status = "FAIL"
                                    else:
                                        if calc_rounded == ref_val:
                                            status = "PASS"
                                        else:
                                            status = "FAIL"
                                else:
                                    status = "MISSING"
                            except (ValueError, TypeError):
                                status = "ERROR"

                    feature_results[key] = {"status": status}

        return feature_results, total_time, max_memory


def main() -> int:
    """Run IBSI 1 and print the items that are neither PASS nor a known gap."""
    from dev.compliance_utils import KNOWN_GAPS

    test = TestIBSICompliance()
    test.setUp()
    results, total_time, max_memory = test.get_detailed_results()
    gaps = KNOWN_GAPS["IBSI 1"]
    problems = {k: v["status"] for k, v in results.items() if v["status"] != "PASS" and gaps.get(k) != v["status"]}
    passed = sum(v["status"] == "PASS" for v in results.values())
    print(f"IBSI 1: {passed}/{len(results)} PASS, {len(results) - passed - len(problems)} known gaps, {len(problems)} problems")
    for key, status in sorted(problems.items()):
        print(f"  {status}: {key}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
