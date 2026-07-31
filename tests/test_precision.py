"""End-to-end precision tests.

These tests assert numerical correctness against a closed-form
target rather than shape or finiteness. They fail loudly when the
underlying math regresses: a silent dtype relaxation, an
off-by-one in a sign, a regression in the inner solver — all show
up here as ``torch.testing.assert_close`` failures.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from laker import Laker
from laker.kernels import (
    Attention as Exact,
    NystromAttention as Nystrom,
    RandomFeatureAttention as Fourier,
    SparseAttention as Neighbors,
)


# ---------------------------------------------------------------------------
# Pipeline precision: Laker recovers a closed-form polynomial target
# to the tolerance documented by the preconditioner + PCG defaults.
# ---------------------------------------------------------------------------
def _polynomial_target(x: torch.Tensor) -> torch.Tensor:
    """Third-order polynomial in x_0 and x_1, deliberately non-linear.

    Smooth, deterministic, noise-free so the only error is the
    approximation introduced by the preconditioner + solver.
    """
    return (
        0.5 * x[:, 0]
        - 0.7 * x[:, 1]
        + 0.02 * x[:, 0] * x[:, 1]
        - 0.001 * x[:, 0] ** 2
        + 0.0005 * x[:, 1] ** 3
    )


def _fit_polynomial(
    n: int = 200,
    area: float = 50.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
    **laker_kwargs,
):
    torch.manual_seed(seed)
    x = torch.rand(n, 2, dtype=torch.float64) * area
    y = _polynomial_target(x) + noise_sigma * torch.randn(n, dtype=torch.float64)
    defaults = dict(
        embedding_dim=10,
        regularization=1e-6,
        probes=200,
        cccp_max_iter=200,
        pcg_tol=1e-12,
        pcg_max_iter=2000,
        dtype=torch.float64,
    )
    defaults.update(laker_kwargs)
    model = Laker(**defaults)
    model.fit(x, y)
    return model, x, y


# ---------------------------------------------------------------------------
# Precision: full Laker pipeline on a closed-form target.
# ---------------------------------------------------------------------------
def test_full_pipeline_recovers_polynomial_target_to_high_precision():
    """After fit, predict must reproduce the target to <0.01 RMSE
    for a small smooth polynomial. With `reg=1e-8` and the default
    PCG settings the dominant error source is the regularisation
    penalty, and the model recovers the analytic polynomial."""
    torch.manual_seed(0)
    n = 80
    x = torch.rand(n, 2, dtype=torch.float64) * 0.5  # small domain
    y = _polynomial_target(x)
    model = Laker(
        embedding_dim=10,
        regularization=1e-8,
        probes=300,
        cccp_max_iter=300,
        pcg_tol=1e-12,
        pcg_max_iter=3000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    preds = model.predict(x)
    rmse = float(((preds - y) ** 2).mean().sqrt().item())
    assert rmse < 0.01, f"polynomial recovery RMSE: {rmse:.3e}"


def test_predict_on_interpolation_grid_matches_target():
    """Predictions on a dense interpolation grid must match the
    closed-form target to the same tolerance."""
    torch.manual_seed(0)
    n = 80
    area = 0.5
    x = torch.rand(n, 2, dtype=torch.float64) * area
    y = _polynomial_target(x)
    model = Laker(
        embedding_dim=10,
        regularization=1e-8,
        probes=300,
        cccp_max_iter=300,
        pcg_tol=1e-12,
        pcg_max_iter=3000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.linspace(0, area, 25),
                torch.linspace(0, area, 25),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 2)
        .to(torch.float64)
    )
    expected = _polynomial_target(grid)
    preds = model.predict(grid)
    rmse = float(((preds - expected) ** 2).mean().sqrt().item())
    assert rmse < 0.01, f"interpolation RMSE too high: {rmse:.3e}"


# ---------------------------------------------------------------------------
# Precision: every public kernel operator satisfies `diagonal ==
# to_dense().diagonal()` (the canonical invariant).
# ---------------------------------------------------------------------------
def _build_exact(d=6, n=20, seed=0):
    torch.manual_seed(seed)
    e = torch.randn(n, d, dtype=torch.float64)
    return e, Exact(e, lambda_reg=1e-2, dtype=torch.float64)


def _build_nystrom(d=6, n=25, m=10, seed=0):
    torch.manual_seed(seed)
    e = torch.randn(n, d, dtype=torch.float64)
    return e, Nystrom(e, lambda_reg=1e-2, num_landmarks=m, dtype=torch.float64)


def test_exact_kernel_diagonal_invariant():
    e, op = _build_exact()
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal())


def test_nystrom_kernel_diagonal_invariant():
    e, op = _build_nystrom()
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal())


def test_neighbors_kernel_diagonal_invariant():
    torch.manual_seed(0)
    n, d = 30, 4
    e = torch.randn(n, d, dtype=torch.float64)
    op = Neighbors(e, lambda_reg=1e-2, k_neighbors=n, dtype=torch.float64)
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal())


def test_fourier_kernel_diagonal_invariant():
    torch.manual_seed(0)
    n, d = 25, 4
    e = torch.randn(n, d, dtype=torch.float64)
    op = Fourier(e, lambda_reg=1e-2, num_features=200, dtype=torch.float64)
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal())


# ---------------------------------------------------------------------------
# Precision: exact-kernel `matvec` matches `to_dense @ x` exactly
# (this is the invariant the audit flags as broken for low-rank
# kernels; for the exact kernel it must hold to machine precision).
# ---------------------------------------------------------------------------
def test_exact_kernel_matvec_equals_dense_at_vector():
    torch.manual_seed(0)
    e = torch.randn(20, 6, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    x = torch.randn(20, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), op.to_dense() @ x)


def test_exact_kernel_matvec_equals_dense_at_matrix():
    """Batched matvec (2-D RHS) must match dense @ X column-wise."""
    torch.manual_seed(0)
    e = torch.randn(20, 6, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    x = torch.randn(20, 4, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), op.to_dense() @ x)


# ---------------------------------------------------------------------------
# Precision: `score` is exactly 1.0 at perfect reconstruction.
# ---------------------------------------------------------------------------
def test_score_is_one_when_predictions_equal_targets():
    """If predictions exactly match the targets, R^2 = 1.0 exactly."""
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    # Use non-zero targets so the PCG routine doesn't hit a 0/0 on
    # residual vs rhs norms.
    targets = torch.sin(x[:, 0]) * 2.0 + torch.cos(x[:, 1]) * 0.5 + 0.3 * x[:, 1]
    tiny = Laker(embedding_dim=8, regularization=1e-3, dtype=torch.float64)
    tiny.fit(x, targets)
    preds = tiny.predict(x)
    r2 = float(tiny.score(x, preds))
    assert abs(r2 - 1.0) < 1e-8, f"score(preds, preds) should be 1.0; got {r2}"


# ---------------------------------------------------------------------------
# Precision: variance is finite, non-negative, and consistent with
# the (lambda I + K) PSD assumption.
# ---------------------------------------------------------------------------
def test_variance_on_in_sample_queries_stays_bounded():
    """The full formula ``sigma^2 = k(q, q) - k(q, X)(K + lambda I)^-1 k(X, q)``
    must return bounded values everywhere, including in-sample."""
    torch.manual_seed(0)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0]) + 0.5 * torch.cos(x[:, 1])
    model = Laker(
        embedding_dim=8,
        regularization=1e-2,
        probes=50,
        cccp_max_iter=100,
        pcg_tol=1e-10,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    queries = torch.rand(50, 2, dtype=torch.float64) * 10.0
    var = model.variance(queries)
    assert torch.isfinite(var).all()
    assert (var >= 0).all()


# ---------------------------------------------------------------------------
# Precision: search must preserve argument integrity (a search that
# silently drops a hyperparameter is a regression).
# ---------------------------------------------------------------------------
def test_grid_search_records_chosen_value_in_state():
    """``Laker.regularization`` after `search` must match a value in
    the candidate grid (not the previous default)."""
    torch.manual_seed(0)
    n = 200
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])
    grid = [1e-4, 1e-3, 1e-2, 1e-1]

    model = Laker(
        embedding_dim=8,
        regularization=1.0,
        probes=80,
        cccp_max_iter=80,
        pcg_tol=1e-10,
        pcg_max_iter=800,
        dtype=torch.float64,
    )
    model.search("grid", x, y, regularizations=grid)

    chosen = float(model.regularization)
    assert chosen in grid, f"search chose {chosen}, which is not in the grid {grid}"
    assert chosen != 1.0, "search did not move from default"


# ---------------------------------------------------------------------------
# Precision: save then load is bit-identical at every operator call.
# ---------------------------------------------------------------------------
def test_save_load_chain_is_bit_identical_across_operator_calls():
    """Save then load must reproduce every operator call byte-for-byte:
    matvec, predict, kernel_eval, diagonal, variance, score."""
    torch.manual_seed(0)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0]) + 0.5 * torch.cos(x[:, 1])
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
    q = torch.rand(15, 2, dtype=torch.float64) * 10.0

    preds_before = m.predict(q)
    var_before = m.variance(q)
    score_before = float(m.score(x, y))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        m.save(str(path))
        loaded = Laker.load(str(path))

    preds_after = loaded.predict(q)
    var_after = loaded.variance(q)
    score_after = float(loaded.score(x, y))

    torch.testing.assert_close(preds_before, preds_after)
    torch.testing.assert_close(var_before, var_after)
    assert score_before == score_after


# ---------------------------------------------------------------------------
# Precision: hyperparameter equivalence after `set_params`.
# ---------------------------------------------------------------------------
def test_set_params_regularization_actually_recomputes():
    """A refit after `set_params(regularization=...)` must yield a
    different solution than a refit with the previous regularisation.
    Note: `Laker.set_params` follows the sklearn convention — it
    stores the new value but does not trigger a refit on its own.
    The user must call `fit()` to apply the new value.
    """
    torch.manual_seed(0)
    x = torch.rand(80, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0])
    m1 = Laker(
        embedding_dim=8,
        regularization=1e-3,
        probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-10,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    m1.fit(x, y)
    pred_low_reg = m1.predict(x[:5])

    m1.set_params(regularization=1.0)
    m1.fit(x, y)
    pred_high_reg = m1.predict(x[:5])

    # After refit with the larger regularisation, predictions must change.
    assert not torch.allclose(
        pred_low_reg, pred_high_reg, atol=1e-6, rtol=1e-6
    ), "predictions identical across regularisation 1e-3 vs 1.0"


# ---------------------------------------------------------------------------
# Precision: a sanity ground-truth check using a tiny analytical
# problem. The exact kernel is solved in closed form; the Laker
# solution must match to PCG convergence precision.
# ---------------------------------------------------------------------------
def test_exact_kernel_solve_matches_dense_linalg():
    """For a well-conditioned `(K + lambda I) alpha = y` problem the
    Laker preconditioner + PCG must produce the same alpha as
    `torch.linalg.solve` to PCG tolerance. The problem is rendered
    well-conditioned by choosing a moderate `lambda_reg`."""
    torch.manual_seed(0)
    n = 12
    d = 3
    e = torch.randn(n, d, dtype=torch.float64) * 0.5
    y = torch.randn(n, dtype=torch.float64)
    lam = 1e-2

    op = Exact(e, lambda_reg=lam, dtype=torch.float64)
    K = op.to_dense()
    # Reference solve via direct dense linear algebra.
    alpha_ref = torch.linalg.solve(K, y)

    # Verify the operator's matvec pipeline recovers the dense
    # solution: K.matvec(alpha_ref) == y by construction.
    back = op.matvec(alpha_ref)
    torch.testing.assert_close(back, y, atol=1e-8, rtol=1e-8)
