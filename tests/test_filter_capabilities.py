"""Tests for pictologics.filters.capabilities module."""

import inspect
from types import MappingProxyType

import pytest

from pictologics.filters import (
    CAPABILITIES_SCHEMA_VERSION,
    FILTER_CAPABILITIES,
    FilterCapability,
    gabor_filter,
    get_filter_capabilities,
    laplacian_of_gaussian,
    laws_filter,
    mean_filter,
    riesz_log,
    riesz_simoncelli,
    riesz_transform,
    simoncelli_wavelet,
    wavelet_transform,
)

# The public callable each FILTER_CAPABILITIES key describes. FilterCapability
# intentionally has no "callable" field (it's declarative metadata, not a
# registry), so this link is asserted explicitly by the tests below instead.
_FILTER_CALLABLES = {
    "mean": mean_filter,
    "log": laplacian_of_gaussian,
    "laws": laws_filter,
    "gabor": gabor_filter,
    "wavelet": wavelet_transform,
    "simoncelli": simoncelli_wavelet,
    "riesz": riesz_transform,
    "riesz_log": riesz_log,
    "riesz_simoncelli": riesz_simoncelli,
}

# FFT-domain filters honour a requested boundary via the shared pad-filter-crop helper
# in filters/base.py; every filter that declares supported_boundaries now genuinely
# exposes a `boundary` parameter, so no forward-declaration exception set is needed.
_FFT_PADDED_BOUNDARY = {"simoncelli", "riesz", "riesz_log", "riesz_simoncelli"}


def _params(name: str) -> MappingProxyType[str, inspect.Parameter]:
    return inspect.signature(_FILTER_CALLABLES[name]).parameters


class TestSchemaVersion:
    """CAPABILITIES_SCHEMA_VERSION is a dotted major.minor.patch string."""

    def test_is_a_three_part_numeric_version_string(self):
        assert isinstance(CAPABILITIES_SCHEMA_VERSION, str)
        parts = CAPABILITIES_SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestFilterCapabilityDataclass:
    """Structural checks on the FilterCapability record type."""

    def test_is_frozen(self):
        capability = FILTER_CAPABILITIES["mean"]
        with pytest.raises(AttributeError):
            capability.kernel_dimensionality = 99  # type: ignore[misc]

    def test_all_registered_values_are_filter_capability_instances(self):
        for capability in FILTER_CAPABILITIES.values():
            assert isinstance(capability, FilterCapability)


class TestFilterCapabilitiesAgainstRealAPI:
    """Verify FILTER_CAPABILITIES against the real filter callables via inspect."""

    @pytest.mark.parametrize("name", sorted(FILTER_CAPABILITIES))
    def test_key_maps_to_a_real_importable_callable(self, name):
        assert name in _FILTER_CALLABLES
        assert inspect.isfunction(_FILTER_CALLABLES[name])

    @pytest.mark.parametrize("name", sorted(FILTER_CAPABILITIES))
    def test_rotation_pooling_matches_pooling_parameter(self, name):
        has_pooling = "pooling" in _params(name)
        assert bool(FILTER_CAPABILITIES[name].rotation_pooling) == has_pooling

    @pytest.mark.parametrize("name", sorted(FILTER_CAPABILITIES))
    def test_orthogonal_plane_averaging_matches_average_over_planes_parameter(self, name):
        has_param = "average_over_planes" in _params(name)
        assert FILTER_CAPABILITIES[name].orthogonal_plane_averaging == has_param

    @pytest.mark.parametrize("name", sorted(FILTER_CAPABILITIES))
    def test_supported_boundaries_matches_boundary_parameter(self, name):
        has_boundary = "boundary" in _params(name)
        assert bool(FILTER_CAPABILITIES[name].supported_boundaries) == has_boundary

    @pytest.mark.parametrize("name", sorted(FILTER_CAPABILITIES))
    def test_effective_boundary_distinguishes_padded_fft_filters(self, name):
        # The FFT filters honour a boundary only approximately (pad-filter-crop on a
        # domain the transform still treats as periodic); spatial-convolution filters
        # apply it directly. The declared value must not overstate either case.
        effective = FILTER_CAPABILITIES[name].effective_boundary
        if name in _FFT_PADDED_BOUNDARY:
            assert effective == "as_specified_via_padding"
        else:
            assert effective == "as_specified"

    @pytest.mark.parametrize("name", sorted(FILTER_CAPABILITIES))
    def test_structure_tensor_steering_is_false(self, name):
        # Guard: fails the day steering (e.g. tensor_sigma) is actually implemented,
        # signaling that this field needs a real per-filter value.
        assert FILTER_CAPABILITIES[name].structure_tensor_steering is False


class TestGetFilterCapabilities:
    """Tests for the get_filter_capabilities public accessor."""

    @pytest.mark.parametrize("name", sorted(FILTER_CAPABILITIES))
    def test_returns_the_registered_capability(self, name):
        assert get_filter_capabilities(name) is FILTER_CAPABILITIES[name]

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown_filter_xyz"):
            get_filter_capabilities("unknown_filter_xyz")
