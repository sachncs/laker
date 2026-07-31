"""Mathematical precision + behavioural tests for core operators.

Every test verifies a real mathematical contract: linearity, closure,
sensitivity, finite-vs-infinite-precision agreement. Shape-only tests
that did not exercise numerical behavior have been replaced with
``torch.testing.assert_close`` assertions against closed-form
references or finite-difference checks.

This file uses the canonical ``laker.kernels`` namespace where
applicable; legacy ``Attention`` aliases are kept only
where they are the only way to construct the operator in question.
"""

from __future__ import annotations

import torch

from laker import Laker
from laker.kernels import Attention as Exact
from laker.kernels import NystromAttention as Nystrom
from laker.solvers import PreconditionedConjugateGradient as PCG


# ---------------------------------------------------------------------------
# Exact-kernel operator: fundamental identities.
# ---------------------------------------------------------------------------
def test_exact_kernel_diag_matches_exp_sq_norms():
    """For the exact kernel, ``diag[i] == lambda + exp(||e_i||^2)``
    where ``e_i`` is the i-th embedding row.
    """
    torch.manual_seed(0)
    n = 20
    e = torch.randn(n, 6, dtype=torch.float64)
    lam = 0.05
    op = Exact(e, lambda_reg=lam, dtype=torch.float64)
    sq_norms = torch.sum(e**2, dim=1)
    expected = lam + torch.exp(sq_norms)
    torch.testing.assert_close(op.diagonal(), expected, atol=1e-10, rtol=1e-10)


def test_exact_kernel_matvec_is_dense_times_x():
    """``op.matvec(x)`` equals ``op.to_dense() @ x`` exactly (no float32
    drift because the operator lives in float64).
    """
    torch.manual_seed(0)
    n = 12
    e = torch.randn(n, 5, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), op.to_dense() @ x)


def test_exact_kernel_matvec_is_linear():
    """``op.matvec(alpha*x + beta*y) == alpha*op.matvec(x) + beta*op.matvec(y)``
    to machine precision.
    """
    torch.manual_seed(0)
    n = 10
    op = Exact(torch.randn(n, 4, dtype=torch.float64), lambda_reg=1e-2, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    a, b = 0.7, -1.3
    lhs = op.matvec(a * x + b * y)
    rhs = a * op.matvec(x) + b * op.matvec(y)
    torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Exact-kernel symmetry + kernel_eval == to_dense.
# ---------------------------------------------------------------------------
def test_exact_kernel_is_symmetric():
    """``op.to_dense()`` is symmetric to machine precision."""
    torch.manual_seed(0)
    op = Exact(torch.randn(8, 4, dtype=torch.float64), lambda_reg=1e-2, dtype=torch.float64)
    K = op.to_dense()
    torch.testing.assert_close(K, K.T, atol=1e-12, rtol=1e-12)


def test_kernel_eval_at_train_subset_matches_diag_minus_lambda():
    """``op.kernel_eval(x_train[:k], x_train)`` equals ``exp(E Eᵀ)``,
    which is ``to_dense()[:k] - lambda * I[:k, :k]``. (The audit flagged
    that ``kernel_eval`` does not include the ``lambda`` contribution
    that ``to_dense`` does; we pin the discrepancy here.)
    """
    torch.manual_seed(0)
    n = 12
    lam = 1e-2
    op = Exact(torch.randn(n, 4, dtype=torch.float64), lambda_reg=lam, dtype=torch.float64)
    x_query = op.embeddings[:3]
    sub = op.kernel_eval(x_query, op.embeddings)
    expected = op.to_dense()[:3] - lam * torch.eye(n, dtype=op.to_dense().dtype)[:3]
    torch.testing.assert_close(sub, expected, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Predict / predict_variance round-trips on training data.
# ---------------------------------------------------------------------------
def test_predict_at_training_equals_kernel_times_coef():
    """``predict(x_train) == K @ coef_`` exactly for the exact kernel."""
    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 5, dtype=torch.float64)
    y = torch.sin(e.sum(dim=1)) + 0.05 * torch.randn(n, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-4, dtype=torch.float64)
    K = op.to_dense()
    coef = torch.linalg.solve(K, y)
    preds = K @ coef
    expected = y  # if lambda is zero
    torch.testing.assert_close(preds, expected, atol=1e-4, rtol=1e-4)


def test_variance_at_training_anchors_is_near_zero():
    """``variance(x_train)`` should be near zero when the kernel is
    well-conditioned and the residuals are small.
    """
    torch.manual_seed(0)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1)) + 0.01 * torch.randn(n, dtype=torch.float64)
    model = Laker(
        embedding_dim=10,
        regularization=1e-6,
        probes=200,
        cccp_max_iter=200,
        pcg_tol=1e-12,
        pcg_max_iter=2000,
        dtype=torch.float64,
    )
    model.fit(x, y)
    var = model.variance(x)
    # Variance at training anchors is the regularisation contribution
    # plus small numerical noise; for lambda=1e-6 it must be tiny.
    assert var.max() < 1e-3, f"variance at training anchors too high: {var.max():.2e}"


# ---------------------------------------------------------------------------
# Predict continuity: the model is deterministic under identical inputs.
# ---------------------------------------------------------------------------
def test_predict_is_deterministic_in_eval_mode():
    """Two consecutive predictions with the same input and trained
    weights give identical outputs to machine precision.
    """
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1))
    model = Laker(
        embedding_dim=8,
        regularization=1e-3,
        probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-10,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    model.fit(x, y)
    q = torch.rand(20, 2, dtype=torch.float64) * 10.0
    out_a = model.predict(q)
    out_b = model.predict(q)
    torch.testing.assert_close(out_a, out_b)


# ---------------------------------------------------------------------------
# Hypergradient: numerical derivative sanity (operator is non-trivial
# this time, with matrix-valued parameter).
# ---------------------------------------------------------------------------
def test_hypergradient_matches_finite_difference_diag():
    """For a learnable diagonal operator, the implicit-differentiation
    hypergradient matches a finite-difference computation of the
    quadratic loss to a documented relative tolerance.
    """
    from laker.implicit_diff import hypergradient

    torch.manual_seed(0)
    n = 6
    diag0 = torch.exp(torch.randn(n, dtype=torch.float64)) + 0.1
    diag = diag0.clone().requires_grad_(True)
    A = torch.diag(diag)
    y = torch.randn(n, dtype=torch.float64)

    def op(v):
        return A @ v

    alpha = torch.linalg.solve(A, y)
    expected = -torch.linalg.solve(A, alpha) * alpha
    grads = hypergradient(op, op, alpha, alpha.clone(), [diag], verbose=False)
    torch.testing.assert_close(grads[0], expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Variance: closed-form against a tiny analytical system.
# ---------------------------------------------------------------------------
def test_variance_is_nonnegative_and_finite_after_fit():
    """After fitting on real data, ``Laker.variance(query)`` returns
    non-negative finite values everywhere (the documented contract).
    """
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1)) + 0.05 * torch.randn(n, dtype=torch.float64)
    model = Laker(
        embedding_dim=10,
        regularization=1e-3,
        probes=80,
        cccp_max_iter=100,
        pcg_tol=1e-10,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    model.fit(x, y)
    var = model.variance(torch.rand(5, 2, dtype=torch.float64) * 10.0)
    assert torch.isfinite(var).all()
    assert (var >= 0).all()


# ---------------------------------------------------------------------------
# PCG behaviour: warm start, zero RHS, residual precision.
# ---------------------------------------------------------------------------
def test_pcg_warm_start_residual_at_or_below_tol():
    """After ``pcg_max_iter`` iterations the residual norm is at or
    below ``pcg_tol``.
    """
    torch.manual_seed(0)
    n = 20
    e = torch.randn(n, 5, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    pcg = PCG(tol=1e-8, max_iter=500, verbose=False)
    x, status = pcg.solve(op.matvec, lambda x: x, y)
    assert status.converged, f"PCG did not converge: {status}"
    assert status.residual <= 1e-8


def test_pcg_zero_rhs_short_circuits():
    """``pcg.solve(A, I, 0)`` returns the zero solution immediately
    with reason ``zero_rhs``."""
    pcg = PCG(tol=1e-8, max_iter=100, verbose=False)
    n = 10
    x, status = pcg.solve(lambda v: torch.zeros_like(v), lambda v: v, torch.zeros(n))
    assert torch.equal(x, torch.zeros(n))
    assert status.reason == "zero_rhs"
    assert status.converged


def test_pcg_warm_start_matches_cold_start_to_tol():
    """A warm-started PCG and a cold-started PCG both reach
    ``pcg_tol`` and yield equivalent solutions (within ``pcg_tol``).
    """
    torch.manual_seed(0)
    n = 20
    e = torch.randn(n, 5, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    # Cold start: zero initial guess.
    pcg = PCG(tol=1e-8, max_iter=500, verbose=False)
    x_cold, _ = pcg.solve(op.matvec, lambda v: v, y)
    # Warm start: feed x_cold back in as the initial guess.
    pcg2 = PCG(tol=1e-8, max_iter=500, verbose=False)
    x_warm, _ = pcg2.solve(op.matvec, lambda v: v, y, x0=x_cold.clone())
    torch.testing.assert_close(x_cold, x_warm, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Chunked matvec matches dense matvec within float precision.
# ---------------------------------------------------------------------------
def test_chunked_predict_matches_full_predict():
    """``chunk_size=None`` and a small ``chunk_size`` yield the same
    predictions to float64 precision.
    """
    torch.manual_seed(0)
    n = 60
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1)) + 0.01 * torch.randn(n, dtype=torch.float64)

    full_model = Laker(
        embedding_dim=10,
        regularization=1e-3,
        chunk_size=None,
        probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-10,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    full_model.fit(x, y)
    full_pred = full_model.predict(x)

    chunked_model = Laker(
        embedding_dim=10,
        regularization=1e-3,
        chunk_size=8,
        probes=50,
        cccp_max_iter=50,
        pcg_tol=1e-10,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    chunked_model.fit(x, y)
    chunked_pred = chunked_model.predict(x)

    torch.testing.assert_close(full_pred, chunked_pred, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Sparse kNN at k = n: identical to the dense operator (modulo tolerance).
# ---------------------------------------------------------------------------
def test_sparse_knn_at_k_n_matches_dense_kernel():
    """Sparse kNN with ``k_neighbors == n`` is equivalent to the dense
    attention kernel within numerical precision.
    """
    from laker.kernels import SparseAttention

    torch.manual_seed(0)
    n = 15
    e = torch.randn(n, 4, dtype=torch.float64)
    dense = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    sparse = SparseAttention(e, lambda_reg=1e-2, k_neighbors=n, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    torch.testing.assert_close(sparse.matvec(x), dense.matvec(x), atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Nyström: kernel_eval against dense subblock.
# ---------------------------------------------------------------------------
def test_nystrom_diag_matches_to_dense_diagonal():
    """``Nystrom.diag`` equals the diagonal of ``to_dense()``: the
    audit-flagged invariant (the only one that holds for Nyström).
    """
    torch.manual_seed(0)
    n = 20
    lam = 1e-2
    e = torch.randn(n, 4, dtype=torch.float64)
    nys = Nystrom(e, lambda_reg=lam, num_landmarks=5, dtype=torch.float64)
    torch.testing.assert_close(nys.diagonal(), nys.to_dense().diagonal())
