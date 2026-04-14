from .intensity import (
    calculate_intensity_features,
    calculate_intensity_histogram_features,
    calculate_ivh_features,
    calculate_local_intensity_features,
    calculate_spatial_intensity_features,
)
from .morphology import calculate_morphology_features
from .texture import (
    calculate_all_texture_features,
    calculate_all_texture_matrices,
    calculate_glcm_features,
    calculate_gldzm_features,
    calculate_glrlm_features,
    calculate_glszm_features,
    calculate_ngldm_features,
    calculate_ngtdm_features,
)

# ---------------------------------------------------------------------------
# Feature-name registry
# ---------------------------------------------------------------------------
# Canonical ordered tuples of every feature key each family can produce.
# Used by the pipeline to build NaN-filled Series when a configuration fails
# (e.g. empty ROI after preprocessing).
#
# **Maintenance rule**: when a feature key is added or removed in a
# ``calculate_*`` function, the corresponding tuple below MUST be updated.
# The ``test_feature_name_registry_sync`` test enforces this automatically.
# ---------------------------------------------------------------------------

INTENSITY_FEATURE_NAMES: tuple[str, ...] = (
    "mean_intensity_Q4LE",
    "intensity_variance_ECT3",
    "intensity_skewness_KE2A",
    "intensity_kurtosis_IPH6",
    "median_intensity_Y12H",
    "minimum_intensity_1GSF",
    "10th_intensity_percentile_QG58",
    "90th_intensity_percentile_8DWT",
    "maximum_intensity_84IY",
    "intensity_interquartile_range_SALO",
    "intensity_range_2OJQ",
    "intensity_mean_absolute_deviation_4FUA",
    "intensity_robust_mean_absolute_deviation_1128",
    "intensity_median_absolute_deviation_N72L",
    "intensity_coefficient_of_variation_7TET",
    "intensity_quartile_coefficient_of_dispersion_9S40",
    "intensity_energy_N8CA",
    "root_mean_square_intensity_5ZWQ",
)

HISTOGRAM_FEATURE_NAMES: tuple[str, ...] = (
    "mean_discretised_intensity_X6K6",
    "discretised_intensity_variance_CH89",
    "discretised_intensity_skewness_88K1",
    "discretised_intensity_kurtosis_C3I7",
    "median_discretised_intensity_WIFQ",
    "minimum_discretised_intensity_1PR8",
    "10th_discretised_intensity_percentile_1PR",
    "90th_discretised_intensity_percentile_GPMT",
    "maximum_discretised_intensity_3NCY",
    "intensity_histogram_mode_AMMC",
    "discretised_intensity_interquartile_range_WR0O",
    "discretised_intensity_range_5Z3W",
    "intensity_histogram_mean_absolute_deviation_D2ZX",
    "intensity_histogram_robust_mean_absolute_deviation_WRZB",
    "intensity_histogram_median_absolute_deviation_4RNL",
    "intensity_histogram_coefficient_of_variation_CWYJ",
    "intensity_histogram_quartile_coefficient_of_dispersion_SLWD",
    "discretised_intensity_entropy_TLU2",
    "discretised_intensity_uniformity_BJ5W",
    "maximum_histogram_gradient_12CE",
    "maximum_histogram_gradient_intensity_8E6O",
    "minimum_histogram_gradient_VQB3",
    "minimum_histogram_gradient_intensity_RHQZ",
)

IVH_FEATURE_NAMES: tuple[str, ...] = (
    "volume_at_intensity_fraction_0.10_BC2M_10",
    "volume_at_intensity_fraction_0.90_BC2M_90",
    "volume_fraction_difference_between_intensity_0.10_and_0.90_fractions_DDTU",
    "intensity_at_volume_fraction_0.10_GBPN_10",
    "intensity_at_volume_fraction_0.90_GBPN_90",
    "intensity_fraction_difference_between_volume_0.10_and_0.90_fractions_CNV2",
    "area_under_the_ivh_curve_9CMM",
)

SPATIAL_INTENSITY_FEATURE_NAMES: tuple[str, ...] = (
    "morans_i_index_N365",
    "gearys_c_measure_NPT7",
)

LOCAL_INTENSITY_FEATURE_NAMES: tuple[str, ...] = (
    "global_intensity_peak_0F91",
    "local_intensity_peak_VJGA",
)

MORPHOLOGY_FEATURE_NAMES: tuple[str, ...] = (
    "volume_voxel_counting_YEKZ",
    "surface_area_C0JK",
    "volume_RNU0",
    "surface_to_volume_ratio_2PR5",
    "compactness_1_SKGS",
    "compactness_2_BQWJ",
    "spherical_disproportion_KRCK",
    "sphericity_QCFX",
    "asphericity_25C7",
    "major_axis_length_TDIC",
    "minor_axis_length_P9VJ",
    "least_axis_length_7J51",
    "elongation_Q3CK",
    "flatness_N17B",
    "volume_density_aee_6BDE",
    "area_density_aee_RDD2",
    "volume_density_convex_hull_R3ER",
    "area_density_convex_hull_7T7F",
    "maximum_3d_diameter_L0JK",
    "volume_density_aabb_PBX1",
    "area_density_aabb_R59B",
    "volume_density_ombb_ZH1A",
    "area_density_ombb_IQYR",
    "volume_density_mvee_SWZ1",
    "area_density_mvee_BRI8",
    "integrated_intensity_99N0",
    "center_of_mass_shift_KLMA",
)

GLCM_FEATURE_NAMES: tuple[str, ...] = (
    "joint_maximum_GYBY",
    "joint_average_60VM",
    "joint_variance_UR99",
    "joint_entropy_TU9B",
    "difference_average_TF7R",
    "difference_variance_D3YU",
    "difference_entropy_NTRS",
    "sum_average_ZGXS",
    "sum_variance_OEEB",
    "sum_entropy_P6QZ",
    "angular_second_moment_8ZQL",
    "contrast_ACUI",
    "dissimilarity_8S9J",
    "inverse_difference_IB1Z",
    "normalised_inverse_difference_NDRX",
    "inverse_difference_moment_WF0Z",
    "normalised_inverse_difference_moment_1QCO",
    "inverse_variance_E8JP",
    "correlation_NI2N",
    "autocorrelation_QWB0",
    "cluster_tendency_DG8W",
    "cluster_shade_7NFM",
    "cluster_prominence_AE86",
    "information_correlation_1_R8DG",
    "information_correlation_2_JN9H",
)

GLRLM_FEATURE_NAMES: tuple[str, ...] = (
    "short_runs_emphasis_22OV",
    "long_runs_emphasis_W4KF",
    "grey_level_non_uniformity_R5YN",
    "normalised_grey_level_non_uniformity_OVBL",
    "run_length_non_uniformity_W92Y",
    "normalised_run_length_non_uniformity_IC23",
    "run_percentage_9ZK5",
    "grey_level_variance_8CE5",
    "run_length_variance_SXLW",
    "run_entropy_HJ9O",
    "low_grey_level_run_emphasis_V3SW",
    "high_grey_level_run_emphasis_G3QZ",
    "short_run_low_grey_level_emphasis_HTZT",
    "short_run_high_grey_level_emphasis_GD3A",
    "long_run_low_grey_level_emphasis_IVPO",
    "long_run_high_grey_level_emphasis_3KUM",
)

GLSZM_FEATURE_NAMES: tuple[str, ...] = (
    "small_zone_emphasis_P001",
    "large_zone_emphasis_48P8",
    "grey_level_non_uniformity_JNSA",
    "normalised_grey_level_non_uniformity_Y1RO",
    "zone_size_non_uniformity_4JP3",
    "normalised_zone_size_non_uniformity_VB3A",
    "zone_percentage_P30P",
    "grey_level_variance_BYLV",
    "zone_size_variance_3NSA",
    "zone_size_entropy_GU8N",
    "low_grey_level_zone_emphasis_XMSY",
    "high_grey_level_zone_emphasis_5GN9",
    "small_zone_low_grey_level_emphasis_5RAI",
    "small_zone_high_grey_level_emphasis_HW1V",
    "large_zone_low_grey_level_emphasis_YH51",
    "large_zone_high_grey_level_emphasis_J17V",
)

GLDZM_FEATURE_NAMES: tuple[str, ...] = (
    "small_distance_emphasis_0GBI",
    "large_distance_emphasis_MB4I",
    "grey_level_non_uniformity_VFT7",
    "normalised_grey_level_non_uniformity_7HP3",
    "zone_distance_non_uniformity_V294",
    "normalised_zone_distance_non_uniformity_IATH",
    "zone_percentage_VIWW",
    "grey_level_variance_QK93",
    "zone_distance_variance_7WT1",
    "zone_distance_entropy_GBDU",
    "low_grey_level_zone_emphasis_S1RA",
    "high_grey_level_zone_emphasis_K26C",
    "small_distance_low_grey_level_emphasis_RUVG",
    "small_distance_high_grey_level_emphasis_DKNJ",
    "large_distance_low_grey_level_emphasis_A7WM",
    "large_distance_high_grey_level_emphasis_KLTH",
)

NGTDM_FEATURE_NAMES: tuple[str, ...] = (
    "coarseness_QCDE",
    "contrast_65HE",
    "busyness_NQ30",
    "complexity_HDEZ",
    "strength_1X9X",
)

NGLDM_FEATURE_NAMES: tuple[str, ...] = (
    "low_dependence_emphasis_SODN",
    "high_dependence_emphasis_IMOQ",
    "low_grey_level_count_emphasis_TL9H",
    "high_grey_level_count_emphasis_OAE7",
    "low_dependence_low_grey_level_emphasis_EQ3F",
    "low_dependence_high_grey_level_emphasis_JA6D",
    "high_dependence_low_grey_level_emphasis_NBZI",
    "high_dependence_high_grey_level_emphasis_9QMG",
    "grey_level_non_uniformity_FP8K",
    "normalised_grey_level_non_uniformity_5SPA",
    "dependence_count_non_uniformity_Z87G",
    "normalised_dependence_count_non_uniformity_OKJI",
    "dependence_count_percentage_6XV8",
    "grey_level_variance_1PFV",
    "dependence_count_variance_DNX2",
    "dependence_count_entropy_FCBV",
    "dependence_count_energy_CAS9",
)

# Mapping from family name → feature-name tuple, used by the pipeline.
FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "intensity": INTENSITY_FEATURE_NAMES,
    "histogram": HISTOGRAM_FEATURE_NAMES,
    "ivh": IVH_FEATURE_NAMES,
    "spatial_intensity": SPATIAL_INTENSITY_FEATURE_NAMES,
    "local_intensity": LOCAL_INTENSITY_FEATURE_NAMES,
    "morphology": MORPHOLOGY_FEATURE_NAMES,
    "glcm": GLCM_FEATURE_NAMES,
    "glrlm": GLRLM_FEATURE_NAMES,
    "glszm": GLSZM_FEATURE_NAMES,
    "gldzm": GLDZM_FEATURE_NAMES,
    "ngtdm": NGTDM_FEATURE_NAMES,
    "ngldm": NGLDM_FEATURE_NAMES,
}

__all__ = [
    "calculate_intensity_features",
    "calculate_intensity_histogram_features",
    "calculate_ivh_features",
    "calculate_spatial_intensity_features",
    "calculate_local_intensity_features",
    "calculate_morphology_features",
    "calculate_all_texture_features",
    "calculate_all_texture_matrices",
    "calculate_glcm_features",
    "calculate_glrlm_features",
    "calculate_glszm_features",
    "calculate_gldzm_features",
    "calculate_ngtdm_features",
    "calculate_ngldm_features",
    "FEATURE_NAMES",
    "INTENSITY_FEATURE_NAMES",
    "HISTOGRAM_FEATURE_NAMES",
    "IVH_FEATURE_NAMES",
    "SPATIAL_INTENSITY_FEATURE_NAMES",
    "LOCAL_INTENSITY_FEATURE_NAMES",
    "MORPHOLOGY_FEATURE_NAMES",
    "GLCM_FEATURE_NAMES",
    "GLRLM_FEATURE_NAMES",
    "GLSZM_FEATURE_NAMES",
    "GLDZM_FEATURE_NAMES",
    "NGTDM_FEATURE_NAMES",
    "NGLDM_FEATURE_NAMES",
]
