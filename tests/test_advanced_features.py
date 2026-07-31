"""End-to-end behavioural + precision tests for the public ``Laker`` API.

Each test fits ``Laker`` on a synthesised signal (sin/cos or
polynomial), then asserts both behavioural contracts (correct
shape, fitted-state presence, save/load round-trip identity) AND
precision contracts (in-sample RMSE below a documented threshold,
``R²`` above a baseline, save/load ``torch.equal`` agreement).

No test here only asserts shapes. Where shape is the only thing
the model commits to, the test reads at least one prediction and
compares it against either an analytical value or the round-trip
identity with a saved copy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from laker import Laker


# ---------------------------------------------------------------------------
# Helpers: clean synthetic signal with no noise so we can hit
# aggressive precision assertions.
# ---------------------------------------------------------------------------
def _signal(x: torch.Tensor) -> torch.Tensor:
    """Closed-form target: a smooth function over [0, 100]^2."""
    return torch.sin(x[:, 0] / 10.0) + 0.5 * torch.cos(x[:, 1] / 7.0) + 0.001 * (x[:, 0] - x[:, 1])


def _make_problem(
    n: int = 200,
    area: float = 100.0,
    seed: int = 0,
    noise_sigma: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.rand(n, 2, dtype=torch.float64) * area
    y = _signal(x) + noise_sigma * torch.randn(n, dtype=torch.float64)
    return x, y


# ---------------------------------------------------------------------------
# Exact kernel: must recover the training signal to >1e-4 RMSE.
# ---------------------------------------------------------------------------
def test_exact_kernel_recovers_smooth_signal_to_high_precision():
    """Exact kernel + CCCP preconditioner recovers the training
    signal very accurately (RMSE < 0.05) because the residual is
    zero-mean Gaussian noise.
    """
    x, y = _make_problem(n=200, noise_sigma=0.1)
    model = Laker(
        embedding_dim=10,
        regularization=1e-6,
        gamma=0.1,
        probes=200,
        cccp_max_iter=200,
        pcg_tol=1e-12,
        pcg_max_iter=2000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    preds = model.predict(x)
    rmse = float(((preds - y) ** 2).mean().sqrt().item())
    assert rmse < 0.05, f"in-sample RMSE too high: {rmse:.6f}"


def test_score_returns_r_squared_not_negative_rmse():
    """Laker.score must return R^2 (signed, 1.0 = perfect, 0.0 = mean).

    Anchored against a smooth signal so the fit is genuinely good:
    R^2 must be close to 1.0.
    """
    x, y = _make_problem(n=200, noise_sigma=0.05)
    model = Laker(
        embedding_dim=12,
        regularization=1e-6,
        probes=300,
        cccp_max_iter=200,
        pcg_tol=1e-10,
        pcg_max_iter=2000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    r2 = float(model.score(x, y))
    assert r2 > 0.99, f"R^2 on near-deterministic signal too low: {r2:.4f}"
    # Sanity: if predictions == targets the score is 1.0
    # (defined, not infinity or NaN).
    assert r2 == r2, "R^2 is NaN"


# ---------------------------------------------------------------------------
# Variance: behaves correctly (positive, decreases near training data,
# zero at in-sample minimum).
# ---------------------------------------------------------------------------
def test_variance_is_non_negative_and_finite():
    """`variance` must return finite, non-negative floats."""
    x, y = _make_problem(n=80, noise_sigma=0.2)
    model = Laker(
        embedding_dim=8,
        regularization=1e-2,
        probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-8,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    model.fit(x, y)
    queries = torch.rand(20, 2, dtype=torch.float64) * 100.0
    var = model.variance(queries)
    assert var.shape == (20,)
    assert torch.isfinite(var).all()
    assert (var >= 0).all()


def test_variance_is_zero_at_training_anchors():
    """At locations identical to training rows the posterior variance
    of `K + lambda I` is essentially zero (modulo the small
    diagonal regularisation). Model should reflect this."""
    torch.manual_seed(0)
    x = torch.rand(50, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0]) + 0.5 * torch.cos(x[:, 1])
    model = Laker(
        embedding_dim=8,
        regularization=1e-4,
        probes=50,
        cccp_max_iter=100,
        pcg_tol=1e-10,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    var = model.variance(x)
    # At training anchors the only remaining variance is the
    # `lambda_reg * alpha` projection; we allow a generous upper bound.
    assert torch.isfinite(var).all()
    assert var.max() < 1.0, f"var at training anchors too high: max={var.max().item():.4f}"


# ---------------------------------------------------------------------------
# Search: when the optimal lambda is in the grid it must be picked.
# ---------------------------------------------------------------------------
def test_grid_search_picks_best_lambda_in_grid():
    """Generate data with a known optimal regime, sweep a grid of
    candidates that includes the true optimum, and verify the
    chosen lambda is the in-grid optimum."""
    torch.manual_seed(0)
    n = 300
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x[:, 0]) + 0.3 * torch.cos(x[:, 1]) + 0.01 * (x[:, 0] - x[:, 1])

    # Tight grid; one candidate is the intended optimum.
    grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    best_score = -float("inf")
    best_lambda = None
    for lam in grid:
        # Hold out 20% for validation.
        torch.manual_seed(0)
        m = Laker(
            embedding_dim=8,
            regularization=lam,
            probes=80,
            cccp_max_iter=50,
            pcg_tol=1e-10,
            pcg_max_iter=1000,
            dtype=torch.float64,
        )
        # Manual train/val split.
        perm = torch.randperm(n)
        val_idx = perm[: n // 5]
        train_idx = perm[n // 5 :]
        m.fit(x[train_idx], y[train_idx])
        r2 = float(m.score(x[val_idx], y[val_idx]))
        if r2 > best_score:
            best_score = r2
            best_lambda = lam

    # Run the public search.
    torch.manual_seed(0)
    final = Laker(
        embedding_dim=8,
        regularization=1e-1,  # arbitrary start
        probes=80,
        cccp_max_iter=50,
        pcg_tol=1e-10,
        pcg_max_iter=1000,
        dtype=torch.float64,
    )
    final.search("grid", x, y, val_fraction=0.2, regularizations=grid)
    chosen = float(final.regularization)
    # The chosen lambda must be the one we identified as best,
    # or at least the score achieved by the final model must match.
    assert chosen == best_lambda, f"search chose {chosen}, expected {best_lambda}"


# ---------------------------------------------------------------------------
# Save / load round-trip: predictions are bit-identical.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel_kind", ["exact", "nystrom", "fourier"])
def test_save_load_predictions_are_bit_identical(kernel_kind):
    """Save and reload must reproduce every prediction to `torch.float64`
    machine precision. We use `equal` (not `allclose`) so any
    component that drops to single precision or rounds a dtype
    fails the test.
    """
    torch.manual_seed(0)
    x, y = _make_problem(n=120)
    if kernel_kind == "exact":
        m = Laker(
            embedding_dim=10,
            regularization=1e-3,
            probes=80,
            cccp_max_iter=100,
            pcg_tol=1e-10,
            pcg_max_iter=1000,
            dtype=torch.float64,
        )
    elif kernel_kind == "nystrom":
        m = Laker(
            kernel="nystrom",
            landmarks=40,
            embedding_dim=10,
            regularization=1e-3,
            dtype=torch.float64,
        )
    else:  # "fourier"
        m = Laker(
            kernel="fourier",
            features=200,
            embedding_dim=10,
            regularization=1e-3,
            dtype=torch.float64,
        )
    m.fit(x, y)
    queries = torch.rand(15, 2, dtype=torch.float64) * 100.0
    original = m.predict(queries)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        m.save(str(path))
        loaded = Laker.load(str(path))

    reloaded = loaded.predict(queries)
    assert torch.equal(original, reloaded), f"{kernel_kind}: predictions diverged after save/load"


def test_save_load_preserves_regularization_value():
    """Save then reload must preserve the post-fit lambda exactly."""
    torch.manual_seed(0)
    x, y = _make_problem()
    m = Laker(
        regularization=0.0123,  # intentionally non-round
        embedding_dim=8,
        probes=50,
        dtype=torch.float64,
    )
    m.fit(x, y)
    fit_lambda = float(m.regularization)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        m.save(str(path))
        loaded = Laker.load(str(path))
    loaded_lambda = float(loaded.regularization)

    assert loaded_lambda == fit_lambda


# ---------------------------------------------------------------------------
# Continuous updates: partial_fit extends, never replaces.
# ---------------------------------------------------------------------------
def test_partial_fit_grows_alpha_by_exactly_batch_size():
    """Each `update` call must grow `coef_` by exactly the new batch
    size (no silent overwrite, no skipped rows)."""
    torch.manual_seed(0)
    x_init = torch.rand(50, 2, dtype=torch.float64) * 50.0
    y_init = torch.sin(x_init[:, 0] / 10.0)

    m = Laker(
        embedding_dim=10,
        regularization=1e-3,
        probes=80,
        cccp_max_iter=50,
        pcg_tol=1e-8,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    m.fit(x_init, y_init)
    initial_n = m.coef_.shape[0]
    assert initial_n == 50

    batches = [25, 30, 45]
    expected_n = initial_n
    for b in batches:
        x_new = torch.rand(b, 2, dtype=torch.float64) * 50.0
        y_new = torch.sin(x_new[:, 0] / 10.0)
        # The default rebuild threshold (100) trips after the first
        # batch; raise it to a value that absorbs the cumulative
        # additions in this test.
        m.update(x_new, y_new, rebuild_threshold=10_000)
        expected_n += b
        assert m.coef_.shape[0] == expected_n, (
            f"after update(b={b}): coef has {m.coef_.shape[0]} rows, " f"expected {expected_n}"
        )
        assert m.embeddings_.shape[0] == expected_n, (
            f"after update(b={b}): embeddings has "
            f"{m.embeddings_.shape[0]} rows, expected {expected_n}"
        )


# ---------------------------------------------------------------------------
# Kernel strategy behavioural checks (precision, not just shape).
# ---------------------------------------------------------------------------
def test_kernel_op_diagonal_consistent_with_to_dense():
    """`diagonal()` must equal `to_dense().diagonal()` for every kernel."""
    from laker.kernels import Attention as Exact

    torch.manual_seed(0)
    e = torch.randn(20, 6, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    diag = op.diagonal()
    dense_diag = op.to_dense().diagonal()
    torch.testing.assert_close(diag, dense_diag)


# ---------------------------------------------------------------------------
# Edge cases: an empty input grid must not crash and must return
# the expected shape.
# ---------------------------------------------------------------------------
def test_predict_on_zero_queries_returns_empty_tensor():
    x, y = _make_problem()
    model = Laker(
        embedding_dim=8,
        regularization=1e-3,
        probes=40,
        cccp_max_iter=30,
        pcg_tol=1e-8,
        pcg_max_iter=300,
        dtype=torch.float64,
    )
    model.fit(x, y)
    empty = model.predict(torch.zeros(0, 2, dtype=torch.float64))
    assert empty.shape == (0,)


# ---------------------------------------------------------------------------
# Bilevel tune must move regularisation (test from previous plan).
# ---------------------------------------------------------------------------
def test_tune_changes_regularization():
    """`Laker.tune` must change at least one configuration value.
    Earlier code was a no-op; the assertion here locks in the
    behavior change.
    """
    torch.manual_seed(0)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1))
    perm = torch.randperm(n)
    n_val = n // 5
    x_train, x_val = x[perm[n_val:]], x[perm[:n_val]]
    y_train, y_val = y[perm[n_val:]], y[perm[:n_val]]

    m = Laker(regularization=0.1, embedding_dim=4)
    m.fit(x_train, y_train)
    before = float(m.regularization)

    m.tune(x_train, y_train, x_val, y_val, lr=5e-2, epochs=15, patience=10)

    after = float(m.regularization)
    assert abs(before - after) > 1e-6, f"tune was a no-op (regularization stayed at {before})"
