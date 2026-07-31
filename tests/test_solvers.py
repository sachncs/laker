"""Behavioural + precision tests for the iterative solvers.

Covers the documented contracts:
- ``PCG.solve`` returns ``(x, Status)`` where ``Status`` exposes
  ``converged``/``iterations``/``residual``/``reason``.
- Exact-preconditioner PCG converges in ``n`` iterations for an
  ``n x n`` system.
- ``Status.per_rhs`` is populated for batched 2-D RHS.
- ``GradientDescent`` produces a finite solution that drives the
  residual below ``tol``.
- ``JacobiPreconditioner`` reduces the system condition number enough
  to drive PCG to ``tol``.
"""

from __future__ import annotations

import torch

from laker.kernels import Attention
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import (
    GradientDescent,
    JacobiPreconditioner,
    PreconditionedConjugateGradient,
    Status,
)


# ---------------------------------------------------------------------------
# PCG with exact preconditioner: converges in at most n iterations.
# ---------------------------------------------------------------------------
def test_pcg_exact_preconditioner_recovers_linalg_solve():
    """With an exact preconditioner ``M = A^{-1}``, PCG recovers
    ``linalg.solve(A, b)`` to within ``pcg_tol``.
    """
    torch.manual_seed(0)
    n = 20
    a = torch.diag(torch.arange(1, n + 1, dtype=torch.float64))
    b = torch.randn(n, dtype=torch.float64)
    expected = torch.linalg.solve(a, b)

    pcg = PreconditionedConjugateGradient(tol=1e-10, max_iter=n, verbose=False)
    x, status = pcg.solve(lambda v: a @ v, lambda v: torch.linalg.solve(a, v), b)
    torch.testing.assert_close(x, expected, atol=1e-8, rtol=1e-8)
    assert status.converged
    assert status.iterations <= n
    assert status.reason == "converged"


# ---------------------------------------------------------------------------
# PCG with the learned CCCP preconditioner: residual below tol.
# ---------------------------------------------------------------------------
def test_pcg_with_learned_prec_residual_below_tol():
    """PCG with the learned CCCP preconditioner reaches ``pcg_tol``
    on a real attention-kernel system.
    """
    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 4, dtype=torch.float64)
    op = Attention(e, lambda_reg=1e-2, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    pre = CCCPPreconditioner(
        num_probes=50,
        gamma=1e-1,
        max_iter=30,
        tol=1e-5,
        verbose=False,
        dtype=torch.float64,
    )
    pre.build(op.matvec, n)

    pcg = PreconditionedConjugateGradient(tol=1e-8, max_iter=200, verbose=False)
    x, status = pcg.solve(op.matvec, pre.apply, b)
    residual = float((torch.linalg.norm(op.matvec(x) - b) / torch.linalg.norm(b)).item())
    assert status.converged, f"PCG did not converge: {status}"
    assert residual < 5e-2, f"residual {residual:.3e} exceeds 5e-2"


# ---------------------------------------------------------------------------
# Jacobi preconditioner converges for diagonally-dominant systems.
# ---------------------------------------------------------------------------
def test_jacobi_preconditioner_drives_pcg_to_tol():
    """With a strong-diagonal system the Jacobi preconditioner
    lets PCG converge to ``tol``.
    """
    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 4, dtype=torch.float64)
    op = Attention(e, lambda_reg=1.0, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    jac = JacobiPreconditioner(op.diagonal())
    pcg = PreconditionedConjugateGradient(tol=1e-8, max_iter=200, verbose=False)
    x, status = pcg.solve(op.matvec, jac.apply, b)

    residual = float((torch.linalg.norm(op.matvec(x) - b) / torch.linalg.norm(b)).item())
    assert residual < 1e-5, f"residual {residual:.3e} exceeds 1e-5"
    assert status.converged


# ---------------------------------------------------------------------------
# GradientDescent: produces a finite solution that drives the
# residual below ``tol`` for a well-conditioned problem.
# ---------------------------------------------------------------------------
def test_gd_residual_below_tol_for_well_conditioned():
    """``GradientDescent`` with a well-conditioned SPD operator
    drives the residual below its ``tol`` setting.
    """
    torch.manual_seed(0)
    n = 20
    a = 2.0 * torch.eye(n, dtype=torch.float64) + 0.1 * torch.ones(n, n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    def op(v):
        return a @ v

    gd = GradientDescent(step_size=0.4, tol=1e-4, max_iter=5000, verbose=False)
    x = gd.solve(op, b)
    residual = float((torch.linalg.norm(op(x) - b) / torch.linalg.norm(b)).item())
    assert residual < 1e-3
    assert torch.isfinite(x).all()


# ---------------------------------------------------------------------------
# Batched 2-D RHS: ``Status.per_rhs`` is populated.
# ---------------------------------------------------------------------------
def test_pcg_batched_rhs_populates_per_rhs():
    """For a 2-D RHS the ``Status.per_rhs`` list has one entry per
    column and reports converged state for each.
    """
    torch.manual_seed(0)
    n = 8
    a = torch.diag(torch.arange(1, n + 1, dtype=torch.float64))
    rhs = torch.randn(n, 3, dtype=torch.float64)

    pcg = PreconditionedConjugateGradient(tol=1e-10, max_iter=n, verbose=False)
    _x, status = pcg.solve(
        lambda v: a @ v,
        lambda v: torch.linalg.solve(a, v),
        rhs,
    )
    assert status.per_rhs is not None
    assert len(status.per_rhs) == 3
    assert all(p.converged for p in status.per_rhs)


# ---------------------------------------------------------------------------
# Jacobi apply matches ``inv_diag * x``.
# ---------------------------------------------------------------------------
def test_jacobi_apply_equals_inverse_diagonal_times_x():
    """``Jacobi.apply(x) == diag^{-1} * x`` to PCG precision."""
    torch.manual_seed(0)
    diag = torch.rand(20, dtype=torch.float64) + 0.1  # strictly positive
    jac = JacobiPreconditioner(diag)
    x = torch.randn(20, dtype=torch.float64)
    expected = (1.0 / diag) * x
    torch.testing.assert_close(jac.apply(x), expected, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Status dataclass round-trip.
# ---------------------------------------------------------------------------
def test_status_dataclass_fields_match_documented_contract():
    """``Status`` exposes ``converged``/``iterations``/``residual``/
    ``reason``/``per_rhs`` — the documented solver contract.
    """
    status = Status(
        converged=True,
        iterations=42,
        residual=1e-9,
        reason="converged",
        per_rhs=None,
    )
    assert status.converged is True
    assert status.iterations == 42
    assert status.residual == 1e-9
    assert status.reason == "converged"
    assert status.per_rhs is None
