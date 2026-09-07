"""Tests for :mod:`laker.embed`."""

import torch

from laker.embed import Embed, Position, Visual


class TestEmbedABC:
    def test_base_raises(self):
        import pytest

        e = Embed()
        with pytest.raises(NotImplementedError):
            e(torch.randn(2, 2, dtype=torch.float64))

    def test_subclass_must_implement(self):
        # An Embed subclass with no forward is instantiable but raises on call.
        class Bad(Embed):
            pass

        b = Bad()
        with __import__("pytest").raises(NotImplementedError):
            b(torch.randn(2, 2, dtype=torch.float64))


class TestPosition:
    def test_shape(self):
        e = Position(input_dim=2, dim=8, dtype=torch.float64)
        x = torch.randn(10, 2, dtype=torch.float64)
        out = e(x)
        assert out.shape == (10, 8)

    def test_single_point_unsqueezed(self):
        e = Position(input_dim=2, dim=4, dtype=torch.float64)
        x = torch.randn(2, dtype=torch.float64)
        out = e(x)
        assert out.shape == (1, 4)

    def test_deterministic_with_seed(self):
        e1 = Position(input_dim=2, dim=8, seed=42, dtype=torch.float64)
        e2 = Position(input_dim=2, dim=8, seed=42, dtype=torch.float64)
        x = torch.randn(5, 2, dtype=torch.float64)
        torch.testing.assert_close(e1(x), e2(x))

    def test_different_seed_differs(self):
        e1 = Position(input_dim=2, dim=8, seed=0, dtype=torch.float64)
        e2 = Position(input_dim=2, dim=8, seed=1, dtype=torch.float64)
        x = torch.randn(5, 2, dtype=torch.float64)
        assert not torch.equal(e1(x), e2(x))

    def test_dtype_preserved(self):
        e = Position(input_dim=2, dim=4, dtype=torch.float32)
        x = torch.randn(3, 2, dtype=torch.float32)
        out = e(x)
        assert out.dtype == torch.float32

    def test_attrs(self):
        e = Position(input_dim=3, dim=10, num=20, sigma=5.0, dtype=torch.float64)
        assert e.input_dim == 3
        assert e.dim == 10
        assert e.num == 20
        assert e.sigma == 5.0

    def test_default_num(self):
        e = Position(input_dim=2, dim=8, dtype=torch.float64)
        assert e.num == 16

    def test_finite_output(self):
        e = Position(input_dim=2, dim=4, dtype=torch.float64)
        x = torch.randn(50, 2, dtype=torch.float64) * 100
        out = e(x)
        assert torch.isfinite(out).all()

    def test_repr_contains_params(self):
        e = Position(input_dim=2, dim=8, dtype=torch.float64)
        s = e.info()
        assert "input_dim=2" in s
        assert "dim=8" in s


class TestVisual:
    def test_shape(self):
        v = Visual(input_dim=3, dim=10, patch=4, dtype=torch.float64)
        x = torch.randn(5, 3, 32, 32, dtype=torch.float64)
        out = v(x)
        assert out.shape == (5, 10)

    def test_attrs(self):
        v = Visual(input_dim=4, dim=12, patch=8, dtype=torch.float64)
        assert v.input_dim == 4
        assert v.dim == 12
        assert v.patch == 8

    def test_dtype_preserved(self):
        v = Visual(input_dim=3, dim=10, dtype=torch.float32)
        x = torch.randn(2, 3, 16, 16, dtype=torch.float32)
        out = v(x)
        assert out.dtype == torch.float32

    def test_finite_output(self):
        v = Visual(input_dim=3, dim=10, dtype=torch.float64)
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        out = v(x)
        assert torch.isfinite(out).all()
