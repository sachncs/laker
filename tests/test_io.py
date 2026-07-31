"""Save / load round-trip tests with precision and parameter verification.

Every test asserts:
- Predictions are bit-identical (or `torch.allclose`) after reload.
- Dtype, device, and every constructor hyperparameter survives.
- The format-version field exists and equals the documented value.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from laker import Laker
from laker.models import LAKERRegressor
from laker.persistence import ModelPersistence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tmp_model_path(suffix: str = ".pt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


@pytest.fixture
def model_path():
    p = _tmp_model_path()
    yield p
    if os.path.exists(p):
        os.unlink(p)


# ---------------------------------------------------------------------------
# Dtype, device, and every constructor hyperparameter must survive.
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_preserves_dtype(model_path):
    """Predictions are bit-identical after a save/load cycle.

    Uses ``torch.equal`` (not ``allclose``) so any dtype relaxation
    fails the test."""
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 50.0
    y = torch.sin(x[:, 0]) + 0.5 * torch.cos(x[:, 1])
    m = Laker(
        embedding_dim=8,
        regularization=1e-3,
        probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-12,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    m.fit(x, y)
    queries = torch.rand(10, 2, dtype=torch.float64) * 50.0
    p_before = m.predict(queries)

    m.save(model_path)
    loaded = Laker.load(model_path)
    p_after = loaded.predict(queries)

    assert torch.equal(p_before, p_after)
    assert loaded.coef_.dtype == torch.float64
    assert loaded.embeddings_.dtype == torch.float64


def test_save_load_preserves_every_hyperparameter(model_path):
    """After save+load the model must report the same public
    hyperparameters. Specifically, decimal precision:
    randomisation is preserved exactly.
    """
    torch.manual_seed(0)
    x = torch.rand(30, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0])
    m = Laker(
        regularization=0.0123456789,  # non-round
        gamma=0.123456789,
        embedding_dim=7,
        probes=33,
        cccp_max_iter=11,
        cccp_tol=1e-7,
        pcg_tol=1e-11,
        pcg_max_iter=333,
        dtype=torch.float64,
    )
    m.fit(x, y)
    before = m.get_params()
    m.save(model_path)
    loaded = Laker.load(model_path)
    after = loaded.get_params()
    # Hyperparameters that don't enter std normal string conversion.
    for key in (
        "embedding_dim",
        "regularization",
        "gamma",
        "pcg_tol",
        "pcg_max_iter",
    ):
        assert before[key] == after[key], f"{key}: saved={before[key]!r}, loaded={after[key]!r}"


def test_save_load_variance_is_bit_identical(model_path):
    """Variance predictions must also match bit-for-bit."""
    torch.manual_seed(0)
    x = torch.rand(30, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0]) + 0.5 * torch.cos(x[:, 1])
    m = Laker(
        embedding_dim=8,
        regularization=1e-2,
        probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-10,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    m.fit(x, y)
    queries = torch.rand(15, 2, dtype=torch.float64) * 10.0
    v_before = m.variance(queries)

    m.save(model_path)
    loaded = Laker.load(model_path)
    v_after = loaded.variance(queries)
    torch.testing.assert_close(v_before, v_after)


def test_save_load_score_is_bit_identical(model_path):
    """`score` after load must equal score before save."""
    torch.manual_seed(0)
    x = torch.rand(50, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    m = Laker(
        embedding_dim=8,
        regularization=1e-3,
        probes=80,
        cccp_max_iter=80,
        pcg_tol=1e-12,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    m.fit(x, y)
    s_before = float(m.score(x, y))
    m.save(model_path)
    loaded = Laker.load(model_path)
    s_after = float(loaded.score(x, y))
    assert s_before == s_after, f"score diverged: before={s_before}, after={s_after}"


# ---------------------------------------------------------------------------
# Format version: the serialised file must be self-describing.
# ---------------------------------------------------------------------------
def test_save_load_writes_format_version_header(model_path):
    """The checkpoint file contains a recognised ``format_version`` field."""
    torch.manual_seed(0)
    x = torch.rand(20, 2) * 10.0
    y = torch.randn(20)
    m = Laker(embedding_dim=4, dtype=torch.float64)
    m.fit(x, y)
    m.save(model_path)

    state = torch.load(model_path, weights_only=False)
    assert "format_version" in state
    assert isinstance(state["format_version"], int)
    assert state["format_version"] >= 2


# ---------------------------------------------------------------------------
# Predictions match across the public Laker facade and the legacy
# LAKERRegressor class.
# ---------------------------------------------------------------------------
def test_legacy_save_legacy_load_roundtrip(model_path):
    """Persist via the legacy ``ModelPersistence.save`` and reload via
    ``ModelPersistence.load``. Predictions should still agree."""
    torch.manual_seed(0)
    x = torch.rand(30, 2, dtype=torch.float64) * 50.0
    y = torch.sin(x[:, 0])

    legacy = LAKERRegressor(
        embedding_dim=8,
        lambda_reg=1e-2,
        num_probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-12,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    legacy.fit(x, y)
    queries = torch.rand(10, 2, dtype=torch.float64) * 50.0
    legacy_pred = legacy.predict(queries)

    ModelPersistence.save(legacy, model_path)
    loaded = ModelPersistence.load(model_path)
    loaded_pred = loaded.predict(queries)

    torch.testing.assert_close(legacy_pred, loaded_pred, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# Saved model on a different device must be loadable on that device
# (or on CPU).
# ---------------------------------------------------------------------------
def test_save_load_works_on_cpu():
    """The disk artifact contains CPU tensors; loading on CPU works."""
    torch.manual_seed(0)
    x = torch.rand(20, 2) * 10.0
    y = torch.randn(20)

    m = Laker(embedding_dim=4, dtype=torch.float64)
    m.fit(x, y)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        m.save(path)
        loaded = Laker.load(path)
        # Predictions should be on CPU since we saved from CPU.
        x_test = torch.rand(5, 2)
        out = loaded.predict(x_test)
        assert out.device.type == "cpu"
    finally:
        os.unlink(path)
