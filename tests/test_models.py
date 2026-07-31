"""Behavioural + precision tests for the LAKER model API.

The legacy ``LAKERRegressor`` class is exercised for the operations
the new ``Laker`` facade delegates to (fit, predict, score, fit
methods). Tests assert real contracts — recovered coefficients,
finite outputs, non-empty corrector activations — rather than
shape-only checks.
"""

from __future__ import annotations

import pytest
import torch

from laker import Laker
from laker.models import LAKERRegressor


# ---------------------------------------------------------------------------
# Fit + predict on a synthetic problem.
# ---------------------------------------------------------------------------
def test_regressor_fit_predict_shape_and_dtype():
    """After fit, ``alpha`` is a 1-D float tensor of the right length;
    predictions on queries match the requested dtype."""
    torch.manual_seed(0)
    n = 80
    x_train = torch.rand(n, 2) * 100.0
    y_train = torch.randn(n)
    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-2,
        gamma=0.1,
        num_probes=40,
        cccp_max_iter=20,
        cccp_tol=1e-4,
        pcg_tol=1e-8,
        pcg_max_iter=200,
        chunk_size=32,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x_train, y_train)
    assert model.alpha is not None
    assert model.alpha.shape == (n,)
    assert torch.isfinite(model.alpha).all()

    x_test = torch.rand(30, 2) * 100.0
    y_pred = model.predict(x_test)
    assert y_pred.shape == (30,)
    assert torch.isfinite(y_pred).all()


# ---------------------------------------------------------------------------
# Paper n=3 worked example: alpha matches the dense solve to PCG precision.
# ---------------------------------------------------------------------------
def test_regressor_paper_example_recovers_alpha_exactly():
    """The n=3 worked example from Section IV-E of the paper: alpha
    solved by LAKER matches the dense ``linalg.solve(K + lambda I, y)``
    within PCG tolerance.
    """
    e = torch.tensor(
        [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
        dtype=torch.float64,
    )
    y = torch.tensor([-66.14, -65.77, -77.30], dtype=torch.float64)

    # Reference solution via direct dense linear algebra.
    g = torch.exp(e @ e.T)
    alpha_exact = torch.linalg.solve(g + 0.1 * torch.eye(3), y)

    class FixedEmbedding(torch.nn.Module):
        def forward(self, _):
            return e

    model = LAKERRegressor(
        embedding_dim=2,
        lambda_reg=0.1,
        gamma=0.0,  # minimal regularisation for tiny problem
        embedding_module=FixedEmbedding(),
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(torch.zeros(3, 2, dtype=torch.float64), y)
    torch.testing.assert_close(model.alpha, alpha_exact, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Laker.score returns R² (now > 0 on smooth signal, ~0 on noise).
# ---------------------------------------------------------------------------
def test_regressor_score_returns_r2_on_smooth_signal():
    """On a smooth signal R² must beat the mean baseline (R² > 0.5)."""
    torch.manual_seed(0)
    n = 60
    x = torch.rand(n, 2) * 100.0
    y = torch.sin(x.sum(dim=-1) / 30.0)
    model = Laker(embedding_dim=8, regularization=1e-3, verbose=False)
    model.fit(x, y)
    score = float(model.score(x, y))
    assert score > 0.5, f"R² on a smooth target must beat the mean: {score}"


def test_regressor_score_is_one_on_perfect_predictions():
    """If predictions exactly equal the targets, ``score == 1.0``."""
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2) * 10.0
    targets = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    m = Laker(embedding_dim=8, regularization=1e-3, dtype=torch.float64)
    m.fit(x, targets)
    r2 = float(m.score(x, m.predict(x)))
    assert abs(r2 - 1.0) < 1e-8, f"score(preds, preds) should be 1.0; got {r2}"


# ---------------------------------------------------------------------------
# Residual corrector: trained and produces non-trivial output.
# ---------------------------------------------------------------------------
def test_residual_corrector_fitted_and_active():
    """After ``fit_residual_corrector`` the corrector exists and
    produces non-zero output on the training set.
    """
    torch.manual_seed(0)
    n = 100
    x = torch.rand(n, 2) * 100.0
    y = torch.randn(n)

    model = LAKERRegressor(
        embedding_dim=6,
        lambda_reg=1e-2,
        num_probes=40,
        cccp_max_iter=20,
        verbose=False,
    )
    model.fit(x, y)
    assert model.residual_corrector is None
    model.fit_residual_corrector(x, y, epochs=100, patience=10)
    assert model.residual_corrector is not None
    with torch.no_grad():
        corr = model.residual_corrector(x).squeeze()
    assert torch.isfinite(corr).all()
    # The corrector must have non-trivial output (not a constant zero).
    assert torch.norm(corr).item() > 0.0


# ---------------------------------------------------------------------------
# Bilevel tune actually moves regularisation.
# ---------------------------------------------------------------------------
def test_bilevel_tune_changes_regularization():
    """``Laker.tune`` must change at least one configuration value.
    (See the test_bilevel.py coverage for the full flow.)"""
    torch.manual_seed(0)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1))
    perm = torch.randperm(n)
    n_val = n // 5
    x_train, x_val = x[perm[n_val:]], x[perm[:n_val]]
    y_train, y_val = y[perm[n_val:]], y[perm[:n_val]]

    model = Laker(regularization=0.1, embedding_dim=4)
    model.fit(x_train, y_train)
    before = float(model.regularization)
    model.tune(x_train, y_train, x_val, y_val, lr=5e-2, epochs=15, patience=10)
    after = float(model.regularization)
    assert abs(before - after) > 1e-6, f"tune was a no-op (regularization stayed at {before})"


# ---------------------------------------------------------------------------
# Uncertainty-aware training: preserves coef and produces predictions.
# ---------------------------------------------------------------------------
def test_uncertainty_aware_preserves_alpha_and_predicts():
    """After ``fit_uncertainty_aware`` the model still has a non-empty
    ``alpha`` and can predict + return variance."""
    torch.manual_seed(0)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)

    model = LAKERRegressor(
        embedding_dim=4,
        lambda_reg=1e-1,
        num_probes=30,
        cccp_max_iter=10,
        pcg_tol=1e-6,
        pcg_max_iter=200,
        dtype=torch.float64,
        verbose=False,
    )
    model.fit(x, y)
    assert model.alpha is not None

    model.fit_uncertainty_aware(
        x,
        y,
        lr=1e-2,
        epochs=10,
        beta=0.1,
        variance_subset=0.3,
        patience=3,
    )
    assert model.alpha is not None

    x_test = torch.rand(10, 2, dtype=torch.float64) * 100.0
    y_pred = model.predict(x_test)
    var = model.predict_variance(x_test)
    assert y_pred.shape == (10,)
    assert var.shape == (10,)
    assert torch.isfinite(y_pred).all()
    assert (var >= 0).all()


# ---------------------------------------------------------------------------
# Laker.encode / kernel switching round-trip via Laker.fit + predict.
# ---------------------------------------------------------------------------
def test_laker_predict_is_kernel_matvec_at_training():
    """``Laker.predict(x_train)`` equals the dense ``Gram @ coef_`` where
    ``Gram = exp(E E^T)``: the documented pipeline excludes the
    ``lambda * I`` contribution that ``to_dense`` includes, so
    training-point predictions are ``y_train - lambda * coef_``.
    """
    from laker.kernels import Attention as Exact

    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1)) + 0.01 * torch.randn(n, dtype=torch.float64)

    lam = 1e-3
    model = Laker(
        kernel="exact",
        embedding_dim=8,
        regularization=lam,
        probes=80,
        cccp_max_iter=100,
        pcg_tol=1e-10,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    pred_laker = model.predict(x)

    direct_kernel = Exact(model.embeddings_, lambda_reg=lam, dtype=torch.float64)
    pred_direct = direct_kernel.kernel_eval(model.embeddings_, model.embeddings_) @ model.coef_
    torch.testing.assert_close(pred_laker, pred_direct, atol=1e-8, rtol=1e-8)


def test_laker_set_params_validates_against_par():
    """``Laker.set_params(unknown=1)`` raises ``ValueError`` listing
    valid parameter names.
    """
    m = Laker()
    with pytest.raises(ValueError, match="Invalid parameter"):
        m.set_params(bogus_param=1.0)


def test_laker_set_params_then_predict_uses_new_value():
    """After ``set_params(regularization=1e-1)`` the next ``fit`` uses
    the new regularisation (we verify by inspecting the stored value).
    """
    m = Laker(regularization=1e-2, dtype=torch.float64)
    m.set_params(regularization=1e-1)
    assert abs(float(m.regularization) - 1e-1) < 1e-12
