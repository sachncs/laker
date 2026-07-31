"""Tests for :mod:`laker.store`."""

import os

import pytest
import torch

from laker import Laker
from laker.store import Store


@pytest.fixture
def fitted_model():
    torch.manual_seed(0)
    x = torch.rand(30, 2, dtype=torch.float64) * 100
    y = torch.sin(x[:, 0] / 50)
    m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
    m.fit(x, y)
    return m, x, y


class TestSaveLoad:
    def test_save_load_predictions_match(self, fitted_model, tmp_path):
        m, x, y = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        assert os.path.exists(path)
        m2 = Laker.load(path)
        torch.testing.assert_close(m.predict(x), m2.predict(x))

    def test_save_load_score_matches(self, fitted_model, tmp_path):
        m, x, y = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        s1 = m.score(x, y)
        s2 = m2.score(x, y)
        assert abs(s1 - s2) < 1e-8

    def test_save_load_variance_matches(self, fitted_model, tmp_path):
        m, x, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        v1 = m.variance(x)
        v2 = m2.variance(x)
        torch.testing.assert_close(v1, v2, atol=1e-6, rtol=1e-6)

    def test_save_rejects_unfitted(self, tmp_path):
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.save(str(tmp_path / "x.pt"))

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Store.load("/nonexistent/path.pt")

    def test_save_includes_format_version(self, fitted_model, tmp_path):
        m, _, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        assert "format" in state
        assert state["format"] >= 2

    def test_save_includes_hyperparameters(self, fitted_model, tmp_path):
        m, _, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        for key in ("embed_dim", "lam", "gamma", "num", "kernel"):
            assert key in state

    def test_save_includes_fitted_tensors(self, fitted_model, tmp_path):
        m, _, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        state = torch.load(path, weights_only=True)
        assert "embed" in state
        assert "coef" in state


class TestKernelRoundTrip:
    @pytest.mark.parametrize(
        "kernel,kwargs",
        [
            ("exact", {}),
            ("nystrom", {"landmarks": 5}),
            ("fourier", {"features": 32}),
            ("neighbors", {"neighbors": 4}),
            ("grid", {"grid_size": 32}),
            ("spectrum", {"knots": 5}),
        ],
    )
    def test_kernel_persistence(self, tmp_path, kernel, kwargs):
        torch.manual_seed(0)
        x = torch.rand(15, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(embed_dim=4, kernel=kernel, dtype=torch.float64, verbose=False, **kwargs)
        m.fit(x, y)
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        torch.testing.assert_close(m.predict(x), m2.predict(x), atol=1e-6, rtol=1e-6)

    def test_hybrid_kernel_persistence(self, tmp_path):
        torch.manual_seed(0)
        x = torch.rand(15, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50)
        m = Laker(
            embed_dim=4,
            kernel="hybrid",
            landmarks=5,
            neighbors=4,
            dtype=torch.float64,
            verbose=False,
        )
        m.fit(x, y)
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        torch.testing.assert_close(m.predict(x), m2.predict(x), atol=1e-6, rtol=1e-6)


class TestCorrectorPersistence:
    def test_corrector_round_trip(self, tmp_path):
        torch.manual_seed(0)
        x = torch.rand(30, 2, dtype=torch.float64) * 100
        y = torch.sin(x[:, 0] / 50) + 0.1 * torch.randn(30, dtype=torch.float64)
        m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        m.correct(x, y, val=0.2, epochs=3, patience=5, lr=1e-2, seed=0)
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        assert m2.corrector is not None
        torch.testing.assert_close(m.predict(x), m2.predict(x), atol=1e-6, rtol=1e-6)


class TestDeviceDtype:
    def test_load_preserves_float64(self, fitted_model, tmp_path):
        m, x, _ = fitted_model
        path = str(tmp_path / "model.pt")
        m.save(path)
        m2 = Laker.load(path)
        assert m2.dtype == torch.float64
        assert m2.coef_.dtype == torch.float64