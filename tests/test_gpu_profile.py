"""Tests for GPUProfile dataclass and GPU detection."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from ragicamp.models.providers.gpu_profile import GPUProfile


class TestGPUProfileDetect:
    """Test GPUProfile.detect() method with different GPU scenarios."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_no_gpu(self, _):
        """Test detect() when no GPU is available."""
        profile = GPUProfile.detect()

        assert profile.tier == "cpu"
        assert profile.gpu_mem_gb == 0.0
        assert profile.max_num_seqs is None
        assert profile.max_num_batched_tokens is None

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_low_gpu(self, _, mock_props):
        """Test detect() with low-end GPU (< 80 GB)."""
        mock_props.return_value = MagicMock(total_memory=8e9)
        profile = GPUProfile.detect()
        assert profile.tier == "low"
        assert profile.gpu_mem_gb == 8.0
        assert profile.max_num_seqs is None

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_mid_gpu(self, _, mock_props):
        """Test detect() with mid-tier GPU (>= 80 GB, < 160 GB)."""
        mock_props.return_value = MagicMock(total_memory=80e9)
        profile = GPUProfile.detect()
        assert profile.tier == "mid"
        assert profile.max_num_seqs == 256
        assert profile.max_num_batched_tokens == 16384

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_high_gpu(self, _, mock_props):
        """Test detect() with high-tier GPU (>= 160 GB)."""
        mock_props.return_value = MagicMock(total_memory=192e9)
        profile = GPUProfile.detect()
        assert profile.tier == "high"
        assert profile.max_num_seqs == 512
        assert profile.max_num_batched_tokens == 32768

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_boundary_low_to_mid(self, _, mock_props):
        """Test detect() just below mid threshold."""
        mock_props.return_value = MagicMock(total_memory=79e9)
        profile = GPUProfile.detect()
        assert profile.tier == "low"

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_boundary_mid_to_high(self, _, mock_props):
        """Test detect() just below high threshold."""
        mock_props.return_value = MagicMock(total_memory=159e9)
        profile = GPUProfile.detect()
        assert profile.tier == "mid"


class TestGPUProfileFrozenDataclass:
    """Test frozen dataclass behavior of GPUProfile."""

    def test_immutable(self):
        """Test that frozen dataclass prevents field modification."""
        profile = GPUProfile(
            tier="high", gpu_mem_gb=200.0, max_num_seqs=512, max_num_batched_tokens=32768
        )
        with pytest.raises(FrozenInstanceError):
            profile.tier = "low"

    def test_hashable_and_equal(self):
        """Test that frozen dataclass is hashable and supports equality."""
        p1 = GPUProfile(
            tier="mid", gpu_mem_gb=100.0, max_num_seqs=256, max_num_batched_tokens=16384
        )
        p2 = GPUProfile(
            tier="mid", gpu_mem_gb=100.0, max_num_seqs=256, max_num_batched_tokens=16384
        )
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_manual_creation(self):
        """Test manual creation for all tiers."""
        cpu = GPUProfile(tier="cpu", gpu_mem_gb=0, max_num_seqs=None, max_num_batched_tokens=None)
        assert cpu.tier == "cpu"
        low = GPUProfile(tier="low", gpu_mem_gb=8, max_num_seqs=None, max_num_batched_tokens=None)
        assert low.tier == "low"
