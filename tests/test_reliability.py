"""Behavioural + precision tests for end-to-end reliability.

These tests cover behaviour that isn't a unit-level mathematical
contract: encoder determinism, mixed-precision stability, save/load
with custom embedding modules, save/load with low-rank kernels.
"""

from __future__ import annotations

# Local helper module under tests/ that defines a custom encoder.
import sys
import tempfile
from pathlib import Path

import torch

from laker import Laker
from laker.kernels import Attention as Exact
from laker.models import LAKERRegressor

sys.path.insert(0, str(Path(__file__).parent))
from custom_embed import CustomEmbedding  # noqa: E402


# ---------------------------------------------------------------------------
# Determinism: same seed → identical weights and identical outputs.
# ---------------------------------------------------------------------------
def test_position_embedding_determinism():
    """Two ``PositionEmbedding`` instances built with the same seed
    produce identical weights and identical forward outputs.
    """
    from laker.embed import Position

    torch.manual_seed(0)
    emb1 = Position(input_dim=2, embedding_dim=10, seed=42, dtype=torch.float64)
    emb2 = Position(input_dim=2, embedding_dim=10, seed=42, dtype=torch.float64)

    x = torch.randn(5, 2, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(emb1(x), emb2(x))

    for p1, p2 in zip(emb1.mlp.parameters(), emb2.mlp.parameters()):
        torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# Mixed precision: bfloat16 embeddings with float32 PCG completes
# without errors and yields finite outputs.
# ---------------------------------------------------------------------------
def test_bfloat16_mixed_precision_finishes_and_produces_finite_outputs():
    """With ``embedding_dtype=bfloat16`` and ``dtype=float32`` the
    fit completes and predictions are finite.
    """
    torch.manual_seed(0)
    n = 100
    x = torch.rand(n, 2, dtype=torch.float32) * 100.0
    y = torch.randn(n, dtype=torch.float32)
    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        embedding_dtype=torch.bfloat16,
        dtype=torch.float32,
        verbose=False,
    )
    model.fit(x, y)
    assert model.embeddings.dtype == torch.float32
    assert model.alpha is not None
    preds = model.predict(x[:5])
    assert torch.isfinite(preds).all()


# ---------------------------------------------------------------------------
# condition_number returns a positive finite value.
# ---------------------------------------------------------------------------
def test_condition_number_is_positive_finite():
    """``Laker.condition`` returns a positive finite value."""
    torch.manual_seed(0)
    n = 80
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)
    model = Laker(
        embedding_dim=10,
        regularization=1e-2,
        gamma=1e-1,
        probes=50,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    model.fit(x, y)
    kappa = float(model.condition())
    assert 1.0 < kappa < float("inf"), f"condition={kappa}"


# ---------------------------------------------------------------------------
# Save/load: custom embedding module round-trip.
# ---------------------------------------------------------------------------
def test_custom_embedding_save_load_bit_identical_predictions():
    """A model with a custom encoder round-trips through save/load
    with bit-identical predictions.
    """
    torch.manual_seed(0)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)
    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        embedding_module=CustomEmbedding(),
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        loaded = LAKERRegressor.load(path)
        x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
        with torch.no_grad():
            torch.testing.assert_close(model.predict(x_test), loaded.predict(x_test))
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Save/load: low-rank kernel type preservation.
# ---------------------------------------------------------------------------
def test_low_rank_kernel_save_load_preserves_strategy():
    """A Nyström model persists with its ``kernel_operator`` class intact
    after a save/load cycle.
    """
    from laker.kernels import NystromAttention

    torch.manual_seed(0)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)
    model = LAKERRegressor(
        embedding_dim=10,
        lambda_reg=1e-2,
        gamma=1e-1,
        num_probes=30,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-6,
        pcg_max_iter=500,
        kernel_approx="nystrom",
        num_landmarks=30,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        loaded = LAKERRegressor.load(path)
        assert loaded.kernel_approx == "nystrom"
        assert isinstance(loaded.kernel_operator, NystromAttention)
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# End-to-end reliability: predict at training points equals Gram @ coef_
# with PCG precision.
# ---------------------------------------------------------------------------
def test_full_pipeline_predicts_at_training_with_precision():
    """An end-to-end ``Laker`` fit produces predictions on the training
    set that match ``Gram @ coef_`` to PCG precision.
    """
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1)) + 0.01 * torch.randn(n, dtype=torch.float64)
    model = Laker(
        kernel="exact",
        embedding_dim=8,
        regularization=1e-3,
        probes=80,
        cccp_max_iter=200,
        pcg_tol=1e-12,
        pcg_max_iter=2000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    preds = model.predict(x)
    direct = Exact(model.embeddings_, lambda_reg=1e-3, dtype=torch.float64)
    expected = direct.kernel_eval(model.embeddings_, model.embeddings_) @ model.coef_
    torch.testing.assert_close(preds, expected, atol=1e-8, rtol=1e-8)
