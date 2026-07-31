"""Tests for :mod:`laker.check`."""

import pytest
import torch

from laker.check import Check


class TestX:
    def test_valid_2d(self):
        x = torch.randn(5, 3, dtype=torch.float64)
        out = Check.x(x)
        torch.testing.assert_close(out, x)

    def test_wrong_dim(self):
        x = torch.randn(5, dtype=torch.float64)
        with pytest.raises(ValueError, match="2-D"):
            Check.x(x)

    def test_empty(self):
        x = torch.zeros(0, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="at least one row"):
            Check.x(x)

    def test_nan(self):
        x = torch.randn(5, 3, dtype=torch.float64)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            Check.x(x)

    def test_inf(self):
        x = torch.randn(5, 3, dtype=torch.float64)
        x[0, 0] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            Check.x(x)


class TestY:
    def test_1d(self):
        y = torch.randn(5, dtype=torch.float64)
        torch.testing.assert_close(Check.y(y), y)

    def test_2d_squeeze(self):
        y = torch.randn(5, 1, dtype=torch.float64)
        out = Check.y(y)
        assert out.dim() == 1
        assert out.shape == (5,)

    def test_scalar_rejected(self):
        y = torch.tensor(1.0, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            Check.y(y)

    def test_wrong_last_dim(self):
        y = torch.randn(5, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="\\(n, 1\\)"):
            Check.y(y)

    def test_nan(self):
        y = torch.randn(5, dtype=torch.float64)
        y[0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            Check.y(y)


class TestEmbed:
    def test_valid(self):
        out = torch.randn(4, 8, dtype=torch.float64)
        torch.testing.assert_close(Check.embed(out, 8), out)

    def test_wrong_dim(self):
        out = torch.randn(8, dtype=torch.float64)
        with pytest.raises(ValueError, match="2-D"):
            Check.embed(out, 8)

    def test_wrong_feature_dim(self):
        out = torch.randn(4, 7, dtype=torch.float64)
        with pytest.raises(ValueError, match="features"):
            Check.embed(out, 8)

    def test_nan(self):
        out = torch.randn(4, 8, dtype=torch.float64)
        out[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            Check.embed(out, 8)


class TestSplit:
    def test_typical(self):
        n_tr, n_va = Check.split(100, 0.2)
        assert n_tr + n_va == 100
        assert n_va == 20

    def test_minimum(self):
        n_tr, n_va = Check.split(2, 0.5)
        assert n_tr == 1
        assert n_va == 1

    def test_reject_too_few(self):
        with pytest.raises(ValueError, match="at least 2"):
            Check.split(1, 0.5)

    def test_reject_bad_fraction(self):
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            Check.split(10, 0.0)
        with pytest.raises(ValueError, match="\\(0, 1\\)"):
            Check.split(10, 1.0)

    def test_clamp_to_n_minus_1(self):
        n_tr, n_va = Check.split(10, 0.99)
        assert n_va <= 9
        assert n_tr + n_va == 10


class TestTensor:
    def test_passthrough_tensor(self):
        x = torch.randn(3, dtype=torch.float64)
        out = Check.tensor(x)
        torch.testing.assert_close(out, x)

    def test_cast_dtype(self):
        x = torch.randn(3, dtype=torch.float32)
        out = Check.tensor(x, dtype=torch.float64)
        assert out.dtype == torch.float64

    def test_from_list(self):
        out = Check.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert out.shape == (3,)
        assert out.dtype == torch.float64


class TestDevice:
    def test_first_tensor_device(self):
        a = torch.zeros(2, device="cpu")
        assert Check.device(a) == torch.device("cpu")

    def test_no_tensor_returns_cpu(self):
        assert Check.device() == torch.device("cpu")