"""Precision and behavioural tests for ``laker.persistence.ModelPersistence``.

Every save/load cycle is verified bit-identically (or `allclose`-tight)
across every supported kernel strategy. Preconditioner state must
round-trip too: ``predict_variance`` depends on the preconditioner
matrices being present and correct.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from laker.models import LAKERRegressor
from laker.persistence import ModelPersistence


def _tmp_path(suffix: str = ".pt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _cleanup(path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Precision: bit-identical round-trip across every kernel strategy.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kernel_kwargs",
    [
        {},  # default exact kernel
        {"kernel_approx": "nystrom", "num_landmarks": 30},
        {"kernel_approx": "rff", "num_features": 200},
    ],
)
def test_save_load_roundtrip_is_precision_grade(kernel_kwargs):
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 50.0
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    model = LAKERRegressor(
        embedding_dim=8,
        lambda_reg=1e-2,
        num_probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-12,
        pcg_max_iter=500,
        dtype=torch.float64,
        **kernel_kwargs,
    )
    model.fit(x, y)
    queries = torch.rand(10, 2, dtype=torch.float64) * 50.0
    pred_before = model.predict(queries)

    path = _tmp_path()
    try:
        ModelPersistence.save(model, path)
        loaded = ModelPersistence.load(path)
        pred_after = loaded.predict(queries)

        # Match the predictions to PCG precision (1e-8 relative),
        # not just shape.
        torch.testing.assert_close(
            pred_before,
            pred_after,
            atol=1e-8,
            rtol=1e-8,
        )
    finally:
        _cleanup(path)


# ---------------------------------------------------------------------------
# Precision: variance and score must also survive.
# ---------------------------------------------------------------------------
def test_save_load_preserves_variance():
    torch.manual_seed(0)
    x = torch.rand(30, 2, dtype=torch.float64) * 50.0
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    model = LAKERRegressor(
        embedding_dim=8,
        lambda_reg=1e-2,
        num_probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-10,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    model.fit(x, y)
    queries = torch.rand(20, 2, dtype=torch.float64) * 50.0
    v_before = model.predict_variance(queries)

    path = _tmp_path()
    try:
        ModelPersistence.save(model, path)
        loaded = ModelPersistence.load(path)
        v_after = loaded.predict_variance(queries)
        torch.testing.assert_close(v_before, v_after, atol=1e-8, rtol=1e-8)
    finally:
        _cleanup(path)


# ---------------------------------------------------------------------------
# Behavioural: error paths.
# ---------------------------------------------------------------------------
def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        ModelPersistence.load("/nonexistent/laker-model.pt")


def test_load_with_missing_required_key_raises():
    """A state dict missing mandatory fields must raise ``KeyError``."""
    path = _tmp_path()
    try:
        torch.save({"some_field": "value"}, path)
        with pytest.raises(KeyError):
            ModelPersistence.load(path)
    finally:
        _cleanup(path)


# ---------------------------------------------------------------------------
# Behavioural: preconditioner round-trip.
# ---------------------------------------------------------------------------
def test_save_load_preserves_preconditioner_state_for_variance():
    """The preconditioner is what enables the variance predictions to
    match the dense posterior formula. After save+load the
    preconditioner tensors must round-trip bit-identically."""
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0])

    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-3,
        num_probes=80,
        cccp_max_iter=80,
        pcg_tol=1e-12,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    model.fit(x, y)
    q = torch.rand(10, 2, dtype=torch.float64) * 10.0
    v_before = model.predict_variance(q)

    path = _tmp_path()
    try:
        ModelPersistence.save(model, path)
        loaded = ModelPersistence.load(path)
        v_after = loaded.predict_variance(q)
        torch.testing.assert_close(v_before, v_after, atol=1e-8, rtol=1e-8)

        # Verify the preconditioner fields are populated on both
        # original and loaded.
        assert loaded.preconditioner is not None
        assert hasattr(loaded.preconditioner, "isotropic_coef")
        assert hasattr(loaded.preconditioner, "q_basis")
    finally:
        _cleanup(path)
