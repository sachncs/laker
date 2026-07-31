"""Tests for :mod:`laker.train`."""

import torch

from laker import Laker


class TestLearn:
    def test_learn_optimizes_embeddings(self):
        torch.manual_seed(0)
        x = torch.rand(40, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50) + 0.05 * torch.randn(40, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        m.learn(x, y, lr=1e-2, epochs=5, rebuild=1, patience=5)
        assert m.coef_.shape == (40,)
        assert torch.isfinite(m.coef_).all()

    def test_learn_rejects_unfitted(self):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        x = torch.rand(10, 2, dtype=torch.float64)
        y = torch.rand(10, dtype=torch.float64)
        with __import__("pytest").raises(RuntimeError, match="encoder"):
            m.learn(x, y)


class TestCorrect:
    def test_correct_attaches_corrector(self):
        torch.manual_seed(0)
        x = torch.rand(40, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50) + 0.1 * torch.randn(40, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        m.correct(x, y, val=0.2, epochs=3, patience=5, lr=1e-2, seed=0)
        assert m.corrector is not None

    def test_correct_changes_predictions(self):
        torch.manual_seed(0)
        x = torch.rand(40, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50) + 0.1 * torch.randn(40, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        before = m.predict(x).clone()
        m.correct(x, y, val=0.2, epochs=5, patience=5, lr=1e-2, seed=0)
        after = m.predict(x)
        assert not torch.equal(before, after)


class TestCalibrate:
    def test_calibrate_runs(self):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50) + 0.05 * torch.randn(30, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        m.calibrate(x, y, lr=1e-2, epochs=3, beta=0.1, subset=0.5, patience=5, seed=0)
        assert torch.isfinite(m.coef_).all()
