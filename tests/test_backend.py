"""Tests for :mod:`laker.backend`."""

import os

import pytest
import torch

from laker.backend import Backend


@pytest.fixture(autouse=True)
def reset_backend():
    yield
    Backend.load()


class TestDefaults:
    def test_initial_device(self):
        if "LAKER_DEVICE" not in os.environ:
            assert Backend.device == torch.device("cpu")

    def test_initial_dtype(self):
        assert Backend.dtype in (torch.float32, torch.float64)


class TestDeviceSet:
    def test_explicit_string(self):
        Backend.device_set("cpu")
        assert Backend.device == torch.device("cpu")

    def test_explicit_device(self):
        Backend.device_set(torch.device("cpu"))
        assert Backend.device == torch.device("cpu")

    def test_none_auto_select(self):
        Backend.device_set(None)
        assert isinstance(Backend.device, torch.device)


class TestDtypeSet:
    def test_set_float32(self):
        Backend.dtype_set(torch.float32)
        assert Backend.dtype == torch.float32

    def test_set_float64(self):
        Backend.dtype_set(torch.float64)
        assert Backend.dtype == torch.float64

    def test_rejects_int(self):
        with pytest.raises(ValueError, match="floating"):
            Backend.dtype_set(torch.int32)


class TestChunkSet:
    def test_set_positive(self):
        Backend.chunk_set(64)
        assert Backend.chunk == 64 * 1024 * 1024

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):
            Backend.chunk_set(0)


class TestTF32:
    def test_tf32_returns_bool(self):
        result = Backend.tf32()
        assert isinstance(result, bool)


class TestTensor:
    def test_from_list(self):
        t = Backend.tensor([1.0, 2.0, 3.0])
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)

    def test_dtype_device_override(self):
        t = Backend.tensor([1.0, 2.0], dtype=torch.float64, device="cpu")
        assert t.dtype == torch.float64
        assert t.device == torch.device("cpu")

    def test_passthrough_tensor(self):
        x = torch.randn(3, dtype=torch.float32)
        t = Backend.tensor(x, dtype=torch.float64)
        assert t.dtype == torch.float64


class TestCompile:
    def test_no_op_when_disabled(self):
        def f(x):
            return x + 1

        out = Backend.compile(f)
        assert out is f

    def test_with_explicit_mode(self):
        def f(x):
            return x + 1

        out = Backend.compile(f, mode="reduce-overhead")
        assert out is not None


class TestAutocast:
    def test_returns_context(self):
        ctx = Backend.autocast()
        with ctx:
            x = torch.randn(3)
            assert x.shape == (3,)


class TestSeed:
    def test_seed_sets_env(self):
        Backend.seed(99)
        assert os.environ["LAKER_SEED"] == "99"
        assert torch.initial_seed() == 99


class TestSummary:
    def test_summary_runs(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="laker.backend"):
            Backend.summary()
        assert any("backend:" in r.message for r in caplog.records)


class TestLoad:
    def test_load_refreshes(self):
        Backend.dtype_set(torch.float32)
        os.environ["LAKER_DTYPE"] = "float64"
        Backend.load()
        assert Backend.dtype == torch.float64


class TestEnvLoad:
    def test_env_chunk_budget(self):
        os.environ["LAKER_CHUNK_MEMORY_BUDGET"] = "32"
        Backend.load()
        assert Backend.chunk == 32 * 1024 * 1024
        del os.environ["LAKER_CHUNK_MEMORY_BUDGET"]
        Backend.load()