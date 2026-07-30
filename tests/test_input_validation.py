"""Input validation tests."""
from __future__ import annotations

import pytest
import torch

from laker import Laker


def _data(n=20, d=3):
    torch.manual_seed(0)
    return torch.rand(n, d), torch.randn(n)


def test_fit_validates_shape_match():
    x, y = _data(n=20, d=3)
    model = Laker()
    with pytest.raises(ValueError, match="matching"):
        model.fit(x[:10], y)


def test_fit_rejects_nan_in_x():
    x, y = _data()
    x[0, 0] = float("nan")
    model = Laker()
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, y)


def test_fit_rejects_nan_in_y():
    x, y = _data()
    y[5] = float("inf")
    model = Laker()
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, y)


def test_fit_rejects_scalar_y():
    x, _ = _data(n=20, d=3)
    model = Laker()
    with pytest.raises(ValueError, match="1-D|2-D|scalar"):
        model.fit(x, torch.tensor(1.0))


def test_fit_rejects_empty_x():
    x = torch.zeros(0, 3)
    y = torch.zeros(0)
    model = Laker()
    with pytest.raises(ValueError, match="at least one row"):
        model.fit(x, y)


def test_fit_accepts_2d_y_with_one_column():
    x, y = _data()
    y_2d = y.unsqueeze(-1)
    model = Laker()
    model.fit(x, y_2d)
    assert model.coef_ is not None


def test_predict_validates_shape_match():
    x, y = _data()
    model = Laker()
    model.fit(x, y)
    with pytest.raises(ValueError, match="matching"):
        model.score(x[:5], y)


def test_set_params_rejects_unknown():
    model = Laker()
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(alpha=0.5)


def test_set_params_validates_dtype_string():
    model = Laker()
    model.set_params(dtype="float64")
    assert model._legacy.core.dtype == torch.float64


def test_set_params_validates_device_string():
    model = Laker()
    model.set_params(device="cpu")
    assert model._legacy.core.device == torch.device("cpu")
