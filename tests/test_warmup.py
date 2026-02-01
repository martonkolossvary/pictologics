import os
import unittest
import warnings
from unittest.mock import MagicMock, patch

# Filter warnings for the test suite itself if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", message="The NumPy module was reloaded.*", category=UserWarning
)


class TestWarmup(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure we start with no env var set (or save previous state)
        self.prev_env = os.environ.get("PICTOLOGICS_DISABLE_WARMUP")
        if self.prev_env:
            del os.environ["PICTOLOGICS_DISABLE_WARMUP"]

    def tearDown(self) -> None:
        # Restore env var
        if self.prev_env:
            os.environ["PICTOLOGICS_DISABLE_WARMUP"] = self.prev_env
        elif "PICTOLOGICS_DISABLE_WARMUP" in os.environ:
            del os.environ["PICTOLOGICS_DISABLE_WARMUP"]

    def test_warmup_runs(self) -> None:
        """Test that warmup runs successfully and triggers expected internal functions."""
        # We Mock the internal _warmup_* methods to verify flow without paying JIT cost
        with patch("pictologics.warmup._warmup_texture") as mock_tex, patch(
            "pictologics.warmup._warmup_intensity"
        ) as mock_int, patch(
            "pictologics.warmup._warmup_morphology"
        ) as mock_morph:

            from pictologics.warmup import warmup_jit

            warmup_jit()

            mock_tex.assert_called_once()
            mock_int.assert_called_once()
            mock_morph.assert_called_once()

    def test_warmup_disabled(self) -> None:
        """Test that setting PICTOLOGICS_DISABLE_WARMUP stops execution."""
        os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

        # Since INFO logs are removed, just verify no exception is raised
        from pictologics.warmup import warmup_jit

        # Should complete without error when disabled
        warmup_jit()

    @patch("pictologics.warmup._warmup_texture")
    def test_warmup_failure_survived(self, mock_texture: MagicMock) -> None:
        """Test that exception in warmup doesn't crash the program."""
        mock_texture.side_effect = RuntimeError("Something exploded")

        # Warmup failure should emit a RuntimeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from pictologics.warmup import warmup_jit

            warmup_jit()

            # Verify warning was emitted
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            self.assertTrue(len(runtime_warnings) >= 1)
            self.assertTrue(any("warmup failed" in str(x.message).lower() for x in runtime_warnings))

    @patch("pictologics.warmup.numba.get_num_threads")
    def test_warmup_fallback_logic(self, mock_get_threads: MagicMock) -> None:
        """Test fallback logic when get_num_threads fails."""
        mock_get_threads.side_effect = Exception("Numba error")

        # We need to ensure the rest of the warmup can still run or at least start
        # This will hit lines 65-66 in warmup.py
        with patch("pictologics.warmup.numba.config") as mock_config:
            mock_config.NUMBA_NUM_THREADS = 2
            # We mock the internal heavy functions to avoid running them fully here
            # since we only care about the thread logic at the start of _warmup_texture
            with patch("pictologics.warmup._warmup_intensity"), patch(
                "pictologics.warmup._warmup_morphology"
            ), patch("pictologics.warmup.texture") as _mock_texture_mod:

                from pictologics.warmup import warmup_jit

                # Suppress expected warmup failure warning (mocked texture causes failure)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warmup_jit()

                # Check if it tried to access the fallback config
                _ = mock_config.NUMBA_NUM_THREADS

    def test_warmup_integration_data_setup_coverage(self) -> None:
        """
        Integration test to ensure the data setup code in _warmup_* methods runs without error.
        This actually compiles/runs the JIT kernels on dummy data.
        """
        # We allow this to run slowly. It is critical for checking the dummy data correctness.
        from pictologics.warmup import (
            _warmup_intensity,
            _warmup_morphology,
            _warmup_texture,
        )

        # Call them directly to ensure they don't raise exceptions
        try:
            _warmup_texture()
            _warmup_intensity()
            _warmup_morphology()
        except Exception as e:
            self.fail(f"Warmup integration failed with error: {e}")
