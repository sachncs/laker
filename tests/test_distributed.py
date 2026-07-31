"""Tests for :mod:`laker.distributed`."""

import torch

from laker.distributed import Distributed


def _sym_pd(n, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float64)
    return a @ a.T + torch.eye(n, dtype=torch.float64)


class TestSingleDevice:
    def test_single_device_path(self):
        n = 15
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        assert op.single
        assert op.devices == [torch.device("cpu")]
        v = torch.randn(n, dtype=torch.float64)
        out = op.matvec(v)
        assert out.shape == (n,)
        assert torch.isfinite(out).all()

    def test_single_device_matvec_matches_inner(self):
        n = 12
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        v = torch.randn(n, dtype=torch.float64)
        torch.testing.assert_close(op.matvec(v), op.local_op.matvec(v))

    def test_diag_matches_inner(self):
        n = 10
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        torch.testing.assert_close(op.diag(), op.local_op.diag())

    def test_dense_matches_inner(self):
        n = 10
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        torch.testing.assert_close(op.dense(), op.local_op.dense())

    def test_eval_matches_inner(self):
        n = 10
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        q = torch.randn(5, 4, dtype=torch.float64)
        torch.testing.assert_close(op.eval(q), op.local_op.eval(q))


class TestSlicing:
    def test_sum_of_shards_equals_full(self):
        n = 12
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        # Force multi-device code path by patching single.
        op.single = False
        # Build artificial per-device shards.
        from laker.kernel import Exact

        op.ops = [
            Exact(embeddings=e[: n // 2], lam=0.1, dtype=torch.float64),
            Exact(embeddings=e[n // 2 :], lam=0.1, dtype=torch.float64),
        ]
        op.sizes = [n // 2, n - n // 2]
        v = torch.randn(n, dtype=torch.float64)
        out = op.matvec(v)
        ref = torch.exp(e @ e.T) @ v + 0.1 * v
        torch.testing.assert_close(out, ref, atol=1e-8, rtol=1e-8)


class TestShape:
    def test_shape_attributes(self):
        n = 12
        e = torch.randn(n, 4, dtype=torch.float64)
        op = Distributed(embeddings=e, lam=0.1, dtype=torch.float64)
        assert op.size == n
        assert op.dim == 4
        assert op.shape == (n, n)
        assert op.dtype == torch.float64
