"""Unit tests for hardware platform detection module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from effgen.hardware.platform import (
    HardwarePlatform,
    detect_platform,
    get_best_local_backend,
    get_unified_memory_gb,
    is_apple_silicon,
    is_cuda_available,
    is_mlx_available,
    is_mlx_vlm_available,
)


def _clear_all_caches():
    """Clear lru_cache on all cached platform functions."""
    is_apple_silicon.cache_clear()
    is_cuda_available.cache_clear()
    is_mlx_available.cache_clear()
    is_mlx_vlm_available.cache_clear()


class TestIsAppleSilicon:
    """Tests for is_apple_silicon()."""

    def setup_method(self):
        _clear_all_caches()

    def teardown_method(self):
        _clear_all_caches()

    @patch("effgen.hardware.platform.platform")
    @patch("effgen.hardware.platform.sys")
    def test_is_apple_silicon_on_arm64_mac(self, mock_sys, mock_platform):
        mock_sys.platform = "darwin"
        mock_platform.machine.return_value = "arm64"
        assert is_apple_silicon() is True

    @patch("effgen.hardware.platform.platform")
    @patch("effgen.hardware.platform.sys")
    def test_is_apple_silicon_on_linux(self, mock_sys, mock_platform):
        mock_sys.platform = "linux"
        mock_platform.machine.return_value = "x86_64"
        assert is_apple_silicon() is False

    @patch("effgen.hardware.platform.platform")
    @patch("effgen.hardware.platform.sys")
    def test_is_apple_silicon_on_intel_mac(self, mock_sys, mock_platform):
        mock_sys.platform = "darwin"
        mock_platform.machine.return_value = "x86_64"
        assert is_apple_silicon() is False


class TestIsCudaAvailable:
    """Tests for is_cuda_available()."""

    def setup_method(self):
        _clear_all_caches()

    def teardown_method(self):
        _clear_all_caches()

    def test_is_cuda_available_with_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert is_cuda_available() is True

    def test_is_cuda_available_no_torch(self):
        """When torch is not installed, should return False."""
        with patch.dict("sys.modules", {"torch": None}):
            assert is_cuda_available() is False


class TestIsMlxAvailable:
    """Tests for is_mlx_available()."""

    def setup_method(self):
        _clear_all_caches()

    def teardown_method(self):
        _clear_all_caches()

    @patch("effgen.hardware.platform.platform")
    @patch("effgen.hardware.platform.sys")
    def test_is_mlx_available_on_apple_silicon(self, mock_sys, mock_platform):
        mock_sys.platform = "darwin"
        mock_platform.machine.return_value = "arm64"
        mock_mlx_core = MagicMock()
        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mlx_core}):
            assert is_mlx_available() is True

    @patch("effgen.hardware.platform.platform")
    @patch("effgen.hardware.platform.sys")
    def test_is_mlx_available_not_apple_silicon(self, mock_sys, mock_platform):
        """On non-Apple Silicon, should short-circuit to False without trying import."""
        mock_sys.platform = "linux"
        mock_platform.machine.return_value = "x86_64"
        assert is_mlx_available() is False

    @patch("effgen.hardware.platform.platform")
    @patch("effgen.hardware.platform.sys")
    def test_is_mlx_available_import_fails(self, mock_sys, mock_platform):
        mock_sys.platform = "darwin"
        mock_platform.machine.return_value = "arm64"
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            assert is_mlx_available() is False


class TestGetUnifiedMemoryGb:
    """Tests for get_unified_memory_gb()."""

    @patch("effgen.hardware.platform.sys")
    def test_non_macos_returns_zero(self, mock_sys):
        mock_sys.platform = "linux"
        assert get_unified_memory_gb() == 0.0

    @patch("effgen.hardware.platform.subprocess")
    @patch("effgen.hardware.platform.sys")
    def test_get_unified_memory_gb_macos(self, mock_sys, mock_subprocess):
        mock_sys.platform = "darwin"
        # 16 GB in bytes = 17179869184
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "17179869184\n"
        mock_subprocess.run.return_value = mock_result
        mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

        result = get_unified_memory_gb()
        assert abs(result - 16.0) < 0.01

    @patch("effgen.hardware.platform.subprocess")
    @patch("effgen.hardware.platform.sys")
    def test_get_unified_memory_gb_failure(self, mock_sys, mock_subprocess):
        mock_sys.platform = "darwin"
        mock_subprocess.run.side_effect = OSError("command not found")
        mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

        result = get_unified_memory_gb()
        assert result == 0.0


class TestGetBestLocalBackend:
    """Tests for get_best_local_backend()."""

    def setup_method(self):
        _clear_all_caches()

    def teardown_method(self):
        _clear_all_caches()

    @patch("effgen.hardware.platform.is_cuda_available", return_value=False)
    @patch("effgen.hardware.platform.is_mlx_available", return_value=True)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=True)
    def test_get_best_local_backend_apple_silicon_mlx(
        self, mock_as, mock_mlx, mock_cuda
    ):
        assert get_best_local_backend() == "mlx"

    @patch("effgen.hardware.platform.is_mlx_available", return_value=False)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=False)
    @patch("effgen.hardware.platform.is_cuda_available", return_value=True)
    def test_get_best_local_backend_cuda_vllm(self, mock_cuda, mock_as, mock_mlx):
        mock_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            assert get_best_local_backend() == "vllm"

    @patch("effgen.hardware.platform.is_mlx_available", return_value=False)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=False)
    @patch("effgen.hardware.platform.is_cuda_available", return_value=True)
    def test_get_best_local_backend_cuda_no_vllm(self, mock_cuda, mock_as, mock_mlx):
        with patch.dict("sys.modules", {"vllm": None}):
            assert get_best_local_backend() == "transformers"

    @patch("effgen.hardware.platform.is_mlx_available", return_value=False)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=False)
    @patch("effgen.hardware.platform.is_cuda_available", return_value=False)
    def test_get_best_local_backend_cpu_only(self, mock_cuda, mock_as, mock_mlx):
        assert get_best_local_backend() == "transformers"


class TestDetectPlatform:
    """Tests for detect_platform()."""

    def setup_method(self):
        _clear_all_caches()

    def teardown_method(self):
        _clear_all_caches()

    @patch("effgen.hardware.platform.is_cuda_available", return_value=False)
    @patch("effgen.hardware.platform.is_mlx_available", return_value=True)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=True)
    def test_detect_apple_silicon(self, mock_as, mock_mlx, mock_cuda):
        assert detect_platform() == HardwarePlatform.APPLE_SILICON

    @patch("effgen.hardware.platform.is_mlx_available", return_value=False)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=False)
    @patch("effgen.hardware.platform.is_cuda_available", return_value=True)
    def test_detect_cuda(self, mock_cuda, mock_as, mock_mlx):
        assert detect_platform() == HardwarePlatform.CUDA

    @patch("effgen.hardware.platform.is_mlx_available", return_value=False)
    @patch("effgen.hardware.platform.is_apple_silicon", return_value=False)
    @patch("effgen.hardware.platform.is_cuda_available", return_value=False)
    def test_detect_cpu_only(self, mock_cuda, mock_as, mock_mlx):
        assert detect_platform() == HardwarePlatform.CPU_ONLY
