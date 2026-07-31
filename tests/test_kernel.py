"""Tests for :mod:`laker.kernel`."""

import pytest
import torch

from laker.kernel import (
    Exact,
    Fourier,
    Grid,
    Hybrid,
    Kernel,
    Neighbors,
    Nystrom,
    Shaper,
    Spectrum,
    exp_safe,
    exact_matvec,
    weights,
)


def _emb(n=20, d=4, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d, dtype=torch.float64)


class TestExpSafe:
    def test_no_overflow_at_cap(self):
        for dtype, cap in [(torch.float32, 80.0), (torch.float64, 700.0)]:
            x = torch.tensor([cap + 100.0], dtype=dtype)
            out = exp_safe(x)
            assert torch.isfinite(out).all()

    def test_inplace_out(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        out = exp_safe(x, out=x)
        assert out is x


class TestExactMatvec:
    def test_matches_dense(self):
        e = _emb()
        v = _emb(n=20, d=4, seed=1)
        out = exact_matvec(e, lam=0.1, x=v)
        expected = 0.1 * v + torch.exp(e @ e.T) @ v
        torch.testing.assert_close(out, expected, atol=1e-8, rtol=1e-8)


class TestExact:
    def test_shape_and_matvec(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        v = torch.randn(20, dtype=torch.float64)
        out = k.matvec(v)
        assert out.shape == (20,)
        assert torch.isfinite(out).all()

    def test_matvec_matches_dense(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        v = torch.randn(20, dtype=torch.float64)
        torch.testing.assert_close(k.matvec(v), k.dense() @ v, atol=1e-8, rtol=1e-8)

    def test_diag_matches_dense_diag(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        torch.testing.assert_close(k.diag(), k.dense().diagonal(), atol=1e-8, rtol=1e-8)

    def test_eval_shape(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        q = torch.randn(5, 4, dtype=torch.float64)
        out = k.eval(q)
        assert out.shape == (5, 20)

    def test_eval_matches_kernel_without_lambda(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        # eval returns the unregularised kernel; dense adds lambda I.
        ref = torch.exp(e @ e.T)
        torch.testing.assert_close(k.eval(e), ref, atol=1e-8, rtol=1e-8)

    def test_dense_adds_lambda_to_diagonal(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        d = k.dense()
        diag_no_lam = torch.exp((e @ e.T).diagonal())
        torch.testing.assert_close(d.diagonal(), diag_no_lam + 0.1, atol=1e-8, rtol=1e-8)

    def test_2d_matvec(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        v = torch.randn(20, 3, dtype=torch.float64)
        out = k.matvec(v)
        assert out.shape == (20, 3)

    def test_rejects_wrong_dim_input(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D or 2-D"):
            k.matvec(torch.randn(2, 2, 2, dtype=torch.float64))

    def test_rejects_size_mismatch(self):
        e = _emb(n=20)
        k = Exact(e, lam=0.1, dtype=torch.float64)
        with pytest.raises(ValueError, match="must have"):
            k.matvec(torch.randn(15, dtype=torch.float64))

    def test_dense_shape(self):
        e = _emb()
        k = Exact(e, lam=0.1, dtype=torch.float64)
        d = k.dense()
        assert d.shape == (20, 20)


class TestNystrom:
    def test_diag_positive(self):
        torch.manual_seed(0)
        e = torch.rand(30, 4, dtype=torch.float64)
        k = Nystrom(e, lam=1e-3, num=5, dtype=torch.float64)
        d = k.diag()
        assert (d > 0).all()

    def test_matvec_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Nystrom(e, lam=1e-3, num=8, dtype=torch.float64)
        v = torch.randn(20, dtype=torch.float64)
        torch.testing.assert_close(k.matvec(v), k.dense() @ v, atol=1e-6, rtol=1e-6)

    def test_landmarks_within_range(self):
        torch.manual_seed(0)
        e = torch.rand(30, 4, dtype=torch.float64)
        k = Nystrom(e, lam=1e-3, num=5, dtype=torch.float64)
        idx = k.landmark_index
        assert (idx >= 0).all() and (idx < 30).all()

    def test_eval_shape(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Nystrom(e, lam=1e-3, num=5, dtype=torch.float64)
        q = torch.rand(7, 4, dtype=torch.float64)
        assert k.eval(q).shape == (7, 20)

    def test_leverage_selection(self):
        torch.manual_seed(0)
        e = torch.rand(40, 4, dtype=torch.float64)
        k = Nystrom(e, lam=1e-3, num=5, method="leverage", dtype=torch.float64)
        v = torch.randn(40, dtype=torch.float64)
        assert torch.isfinite(k.matvec(v)).all()


class TestFourier:
    def test_matvec_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Fourier(e, lam=1e-3, num=64, dtype=torch.float64)
        v = torch.randn(20, dtype=torch.float64)
        torch.testing.assert_close(k.matvec(v), k.dense() @ v, atol=1e-6, rtol=1e-6)

    def test_diag_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Fourier(e, lam=1e-3, num=64, dtype=torch.float64)
        torch.testing.assert_close(k.diag(), k.dense().diagonal(), atol=1e-8, rtol=1e-8)

    def test_eval_shape(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Fourier(e, lam=1e-3, num=64, dtype=torch.float64)
        q = torch.rand(5, 4, dtype=torch.float64)
        assert k.eval(q).shape == (5, 20)


class TestNeighbors:
    def test_diag_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(15, 4, dtype=torch.float64)
        k = Neighbors(e, lam=1e-3, k=5, dtype=torch.float64)
        torch.testing.assert_close(k.diag(), k.dense().diagonal(), atol=1e-6, rtol=1e-6)

    def test_matvec_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(15, 4, dtype=torch.float64)
        k = Neighbors(e, lam=1e-3, k=5, dtype=torch.float64)
        v = torch.randn(15, dtype=torch.float64)
        torch.testing.assert_close(k.matvec(v), k.dense() @ v, atol=1e-6, rtol=1e-6)

    def test_eval_returns_sparse(self):
        torch.manual_seed(0)
        e = torch.rand(15, 4, dtype=torch.float64)
        k = Neighbors(e, lam=1e-3, k=5, dtype=torch.float64)
        q = torch.rand(5, 4, dtype=torch.float64)
        out = k.eval(q)
        assert out.is_sparse


class TestWeights:
    def test_weights_sum_to_one(self):
        g1 = torch.linspace(0, 1, 4, dtype=torch.float64)
        g2 = torch.linspace(0, 1, 3, dtype=torch.float64)
        x = torch.tensor(
            [[0.1, 0.5], [0.9, 0.1], [0.5, 0.5]], dtype=torch.float64
        )
        _, w = weights(x, [g1, g2])
        sums = w.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(3, dtype=torch.float64))


class TestGrid:
    def test_diag_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(15, 3, dtype=torch.float64)
        k = Grid(e, lam=1e-3, grid_size=64, dtype=torch.float64)
        torch.testing.assert_close(k.diag(), k.dense().diagonal(), atol=1e-6, rtol=1e-6)

    def test_matvec_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(15, 3, dtype=torch.float64)
        k = Grid(e, lam=1e-3, grid_size=64, dtype=torch.float64)
        v = torch.randn(15, dtype=torch.float64)
        torch.testing.assert_close(k.matvec(v), k.dense() @ v, atol=1e-6, rtol=1e-6)

    def test_grid_size_too_small(self):
        torch.manual_seed(0)
        e = torch.rand(15, 4, dtype=torch.float64)
        with pytest.raises(ValueError, match="at least 2"):
            Grid(e, lam=1e-3, grid_size=1, dtype=torch.float64)


class TestHybrid:
    def test_diag_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Hybrid(e, lam=1e-3, alpha=0.5, num=8, k=5, dtype=torch.float64)
        torch.testing.assert_close(k.diag(), k.dense().diagonal(), atol=1e-6, rtol=1e-6)

    def test_matvec_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Hybrid(e, lam=1e-3, alpha=0.5, num=8, k=5, dtype=torch.float64)
        v = torch.randn(20, dtype=torch.float64)
        torch.testing.assert_close(k.matvec(v), k.dense() @ v, atol=1e-6, rtol=1e-6)

    def test_alpha_extremes(self):
        torch.manual_seed(0)
        e = torch.rand(15, 4, dtype=torch.float64)
        k = Hybrid(e, lam=1e-3, alpha=0.0, num=8, k=5, dtype=torch.float64)
        v = torch.randn(15, dtype=torch.float64)
        assert torch.isfinite(k.matvec(v)).all()


class TestShaper:
    def test_set_and_forward(self):
        s = Shaper(knots=5)
        s.set(0.0, 1.0, device="cpu", dtype=torch.float64)
        x = torch.tensor([0.5], dtype=torch.float64)
        out = s(x)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()

    def test_set_required_first(self):
        s = Shaper(knots=3)
        with pytest.raises(RuntimeError, match="knots not set"):
            s(torch.tensor([0.5], dtype=torch.float64))


class TestSpectrum:
    def test_matvec_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Spectrum(e, lam=1e-3, knots=5, dtype=torch.float64)
        v = torch.randn(20, dtype=torch.float64)
        torch.testing.assert_close(k.matvec(v), k.dense() @ v, atol=1e-6, rtol=1e-6)

    def test_diag_matches_dense(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Spectrum(e, lam=1e-3, knots=5, dtype=torch.float64)
        torch.testing.assert_close(k.diag(), k.dense().diagonal(), atol=1e-6, rtol=1e-6)

    def test_eval_shape(self):
        torch.manual_seed(0)
        e = torch.rand(20, 4, dtype=torch.float64)
        k = Spectrum(e, lam=1e-3, knots=5, dtype=torch.float64)
        q = torch.rand(7, 4, dtype=torch.float64)
        assert k.eval(q).shape == (7, 20)