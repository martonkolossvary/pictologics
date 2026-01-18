import numpy as np
import pytest
from pictologics.filters.gabor import gabor_filter
from pictologics.filters.laws import laws_filter
from pictologics.filters.wavelets import wavelet_transform


class TestFiltersParallelCoverage:
    """specific tests to force coverage of parallel execution blocks."""

    def test_gabor_parallel_coverage(self):
        # Force use_parallel=True on a small image
        image = np.random.rand(10, 10, 10).astype(np.float32)

        # Test basic parallel execution
        gabor_filter(
            image,
            sigma_mm=1.0,
            lambda_mm=1.0,
            gamma=1.0,
            theta=0.0,
            rotation_invariant=True,
            delta_theta=np.pi / 4,
            use_parallel=True,
        )

        # Test parallel with average_over_planes=True
        gabor_filter(
            image,
            sigma_mm=1.0,
            lambda_mm=1.0,
            rotation_invariant=True,
            delta_theta=np.pi / 4,
            average_over_planes=True,
            use_parallel=True,
        )

    def test_laws_parallel_coverage(self):
        # Force use_parallel=True
        image = np.random.rand(10, 10, 10).astype(np.float32)

        # Test parallel execution
        laws_filter(image, "E5L5S5", rotation_invariant=True, use_parallel=True)

        # Test energy computation coverage
        laws_filter(
            image,
            "E5L5S5",
            rotation_invariant=True,
            compute_energy=True,
            energy_distance=1,
            use_parallel=True,
        )

        # Test pooling modes in parallel (Laws default is "max")
        laws_filter(
            image,
            "E3L3S3",
            rotation_invariant=True,
            pooling="average",
            use_parallel=True,
        )
        laws_filter(
            image, "E3L3S3", rotation_invariant=True, pooling="min", use_parallel=True
        )

        # Test boundary as string
        laws_filter(image, "E3L3S3", boundary="mirror")

        # Test invalid pooling (raises ValueError at API level)
        with pytest.raises(ValueError, match="Unknown pooling"):
            laws_filter(image, "E3L3S3", rotation_invariant=True, pooling="invalid")

    def test_wavelet_parallel_coverage(self):
        # Force use_parallel=True
        image = np.random.rand(16, 16, 16).astype(np.float32)

        # Test parallel execution
        wavelet_transform(
            image,
            wavelet="db2",
            level=1,
            decomposition="LHL",
            rotation_invariant=True,
            use_parallel=True,
        )

        # Test pooling modes in parallel (Wavelet default is "average")
        wavelet_transform(
            image,
            wavelet="db2",
            level=1,
            decomposition="LHL",
            rotation_invariant=True,
            pooling="max",
            use_parallel=True,
        )
        wavelet_transform(
            image,
            wavelet="db2",
            level=1,
            decomposition="LHL",
            rotation_invariant=True,
            pooling="min",
            use_parallel=True,
        )

        # Test invalid pooling (raises ValueError at API level)
        with pytest.raises(ValueError, match="Unknown pooling"):
            wavelet_transform(
                image,
                wavelet="db2",
                level=1,
                rotation_invariant=True,
                pooling="invalid",
            )
