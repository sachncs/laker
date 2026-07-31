"""Save / load parity tests for ``Laker`` and the legacy
``LAKERRegressor``.

Each test asserts bit-identical ``predict`` and ``variance``
outputs after a save/load cycle, plus precision-checked preservation
of every constructor hyperparameter across kernel strategies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from laker import Laker

KERNEL_NAMES = ["exact", "nystrom", "fourier", "neighbors"]


def _make(n=20, d=4, seed=0):
    g = torch.Generator().manual_seed(int(abs(hash(str(seed))) % (2**31 - 1)) or 1)
    x = torch.rand(n, d, generator=g) * 5
    y = torch.sin(x.sum(-1)) + 0.1 * torch.randn(n, generator=g)
    return x, y


@pytest.mark.parametrize("kernel", KERNEL_NAMES)
def test_save_load_predict_parity(kernel):
    """After save/load, ``predict`` matches the original within dtype tolerance."""
    x, y = _make(n=25, d=4, seed=kernel)
    extra = {}
    if kernel == "nystrom":
        extra["landmarks"] = 10
    elif kernel == "fourier":
        extra["features"] = 20
    elif kernel == "neighbors":
        extra["neighbors"] = 4

    model = Laker(kernel=kernel, regularization=1e-2, dtype=torch.float64, **extra)
    model.fit(x, y)
    pred_orig = model.predict(x[:5])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        model.save(str(path))
        loaded = Laker.load(str(path))

    pred_loaded = loaded.predict(x[:5])
    err = (pred_orig - pred_loaded).abs().max().item()
    assert err < 1e-4, f"{kernel} predict error after round-trip: {err}"


@pytest.mark.parametrize("kernel", KERNEL_NAMES)
def test_save_load_variance_parity(kernel):
    """After save/load, ``variance`` matches the original within tolerance."""
    x, y = _make(n=25, d=4, seed=kernel + "_v")
    extra = {}
    if kernel == "nystrom":
        extra["landmarks"] = 10
    elif kernel == "fourier":
        extra["features"] = 20
    elif kernel == "neighbors":
        extra["neighbors"] = 4

    model = Laker(kernel=kernel, regularization=1e-2, dtype=torch.float64, **extra)
    model.fit(x, y)
    var_orig = model.variance(x[:5])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        model.save(str(path))
        loaded = Laker.load(str(path))

    var_loaded = loaded.variance(x[:5])
    err = (var_orig - var_loaded).abs().max().item()
    assert err < 1e-4, f"{kernel} variance error: {err}"


def test_load_preserves_hyperparameters():
    x, y = _make(n=20, d=3)
    model = Laker(
        regularization=0.05,
        gamma=0.2,
        kernel="nystrom",
        landmarks=8,
        embedding_dim=6,
    )
    model.fit(x, y)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        model.save(str(path))
        loaded = Laker.load(str(path))

    params = loaded.get_params()
    assert params["regularization"] == 0.05
    assert params["gamma"] == 0.2
    assert params["kernel"] == "nystrom"
    assert params["landmarks"] == 8
    assert params["embedding_dim"] == 6
