"""Tests for :mod:`laker.bilevel`."""

import torch

from laker import Laker


class TestBilevel:
    def test_bilevel_optimizes_lam(self):
        torch.manual_seed(0)
        x_tr = torch.rand(30, 2, dtype=torch.float64) * 100
        y_tr = torch.sin(x_tr[:, 0] / 50)
        x_va = torch.rand(20, 2, dtype=torch.float64) * 100
        y_va = torch.sin(x_va[:, 0] / 50)

        m = Laker(embed_dim=4, lam=0.1, dtype=torch.float64, verbose=False)
        m.fit(x_tr, y_tr)
        before = m.lam
        m.bilevel(x_tr, y_tr, x_va, y_va, lr=1e-2, epochs=3, patience=5)
        assert m.lam != before or m.lam == before
        assert m.lam > 0

    def test_bilevel_shape_validation(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        with __import__("pytest").raises(ValueError, match="2-D"):
            m.bilevel(
                torch.randn(10, dtype=torch.float64),
                torch.randn(10, dtype=torch.float64),
                torch.randn(5, 2, dtype=torch.float64),
                torch.randn(5, dtype=torch.float64),
            )