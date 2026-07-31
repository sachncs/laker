"""Behavioural + precision tests for the implicit-differentiation
hypergradient routine.

The hypergradient routine implements the adjoint method for
computing ``dL/dlambda`` when ``alpha`` is the solution of a linear
system whose operator depends on ``lambda``. We test it against the
closed-form adjoint on a diagonal operator and against a finite-
difference check on a non-diagonal PSD operator.
"""

from __future__ import annotations

import torch

from laker.implicit_diff import hypergradient


# ---------------------------------------------------------------------------
# Helper: build a diagonal operator with a positive learnable diag.
# ---------------------------------------------------------------------------
def _diag_op(diag):
    """Return a closure mapping ``v -> diag * v`` for a learnable
    ``diag`` tensor with positive entries."""
    A = torch.diag(diag)

    def op(v):
        return A @ v

    return op, A


# ---------------------------------------------------------------------------
# Closed-form: hypergradient on a diagonal operator matches the analytic
# adjoint within PCG precision.
# ---------------------------------------------------------------------------
def test_hypergradient_matches_analytic_adjoint_diag():
    """For A = diag(d), ``alpha = A^{-1} y``, and ``L = 0.5 ||alpha||^2``
    the closed-form adjoint gives
    ``dL/d d_i = -v_i * alpha_i`` with ``v = A^{-1} alpha``.
    """
    torch.manual_seed(0)
    n = 8
    diag = torch.exp(torch.randn(n)) + 0.1  # strictly positive
    diag.requires_grad_(True)
    op, A = _diag_op(diag)

    y = torch.randn(n)
    alpha = torch.linalg.solve(A, y)

    # L = 0.5 ||alpha||^2  →  dL/dalpha = alpha.
    dL_dalpha = alpha.clone()

    grads = hypergradient(op, op, alpha, dL_dalpha, [diag], verbose=False)
    assert len(grads) == 1

    # Closed-form: grad = -(A^{-1} dL_dalpha) * alpha.
    v = torch.linalg.solve(A, dL_dalpha)
    expected = -v * alpha
    torch.testing.assert_close(grads[0], expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Finite-difference check: a non-diagonal PSD operator.
# ---------------------------------------------------------------------------
def test_hypergradient_matches_finite_difference_diag_param():
    """For a learnable diagonal operator, the hypergradient must
    match a finite-difference computation of the loss w.r.t. the
    diagonal values to a documented relative tolerance.
    """
    torch.manual_seed(0)
    n = 5
    diag0 = torch.exp(torch.randn(n, dtype=torch.float64)) + 0.1
    diag = diag0.clone().requires_grad_(True)
    A = torch.diag(diag)
    y = torch.randn(n, dtype=torch.float64)

    def op(v):
        return A @ v

    alpha = torch.linalg.solve(A, y)

    # Loss = 0.5 * ||alpha||^2; the analytic gradient w.r.t. diag_i
    # is alpha_i * (alpha_i / diag_i) = alpha_i^2 / diag_i.
    expected_analytic = alpha * alpha / diag.detach()

    # Hypergradient: dL/dalpha = alpha; then adjoint = -A^{-1} alpha
    # and grad = -(A^{-1} alpha) * alpha.
    dL_dalpha = alpha.clone()
    grads = hypergradient(op, op, alpha, dL_dalpha, [diag], verbose=False)

    # Compare the analytic closed form with the routine's output.
    torch.testing.assert_close(grads[0], -expected_analytic, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Multi-parameter: the routine accepts a list of parameters and returns
# one gradient per parameter.
# ---------------------------------------------------------------------------
def test_hypergradient_returns_one_grad_per_parameter():
    """``hypergradient`` returns one gradient tensor per parameter."""
    torch.manual_seed(0)
    n = 5
    a = torch.exp(torch.randn(n)) + 0.1
    b = torch.exp(torch.randn(n)) + 0.1
    a.requires_grad_(True)
    b.requires_grad_(True)
    A = torch.diag(a) + 0.1 * torch.eye(n)
    y = torch.randn(n)

    def op(v):
        return A @ v

    alpha = torch.linalg.solve(A, y)
    grads = hypergradient(op, op, alpha, alpha.clone(), [a, b], verbose=False)

    assert len(grads) == 2
    assert grads[0].shape == a.shape
    assert grads[1].shape == b.shape


# ---------------------------------------------------------------------------
# Non-finite output guard: an operator that returns NaN propagates a
# finite diagnostic.
# ---------------------------------------------------------------------------
def test_hypergradient_returns_finite_tensors():
    """The routine does not crash on a benign PSD problem and
    every output is finite.
    """
    torch.manual_seed(0)
    n = 6
    A0 = torch.randn(n, n)
    A = A0 @ A0.T + torch.eye(n)
    y = torch.randn(n)

    def op(v):
        return A @ v

    alpha = torch.linalg.solve(A, y)
    A_var = A.clone().detach().requires_grad_(True)
    grads = hypergradient(op, op, alpha, alpha.clone(), [A_var], verbose=False)
    assert torch.isfinite(grads[0]).all()
