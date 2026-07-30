"""Solver status tests.

Captures the contract that every solver entry must return ``(x, status)``
where ``status`` carries ``converged``, ``iterations``, ``residual``,
``reason``. These tests currently xfail on the legacy solver; Step 12
makes them pass.
"""
from __future__ import annotations

import pytest
import torch

from laker.kernels import (
    AttentionKernelOperator,
)


def _psd(n: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(n, n)
    return a @ a.T + torch.eye(n)


@pytest.mark.xfail(reason="PCG must return (x, status); lands in Step 12")
def test_pcg_returns_x_and_status():
    """PCG returns (x, status)."""
    from laker.solvers import Solve
    n = 20
    K = _psd(n)
    rhs = torch.randn(n)
    x, status = Solve.pcg(tol=1e-8, max_iter=200, verbose=False).solve(
        K @ (lambda v: v), None, rhs
    )
    assert hasattr(status, "converged")
    assert hasattr(status, "iterations")
    assert hasattr(status, "residual")
    assert hasattr(status, "reason")


@pytest.mark.xfail(reason="Zero RHS short-circuit; lands in Step 12")
def test_pcg_zero_rhs_converges():
    from laker.solvers import Solve
    n = 20
    K = _psd(n)
    rhs = torch.zeros(n)
    x, status = Solve.pcg(tol=1e-8, max_iter=200, verbose=False).solve(
        lambda v: K @ v, None, rhs
    )
    assert torch.allclose(x, torch.zeros_like(rhs), atol=1e-7)
    assert status.converged is True


@pytest.mark.xfail(reason="Breakdown handling; lands in Step 12")
def test_pcg_breakdown_on_indefinite():
    """``abs(p @ Ap) <= eps * ||p|| * ||Ap||`` triggers breakdown reason."""
    from laker.solvers import Solve
    n = 4

    def indefinite(v):
        # Negative eigenvalue on one direction.
        return torch.tensor([-v[0], v[1], v[2], v[3]], dtype=v.dtype)

    rhs = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _, status = Solve.pcg(tol=1e-12, max_iter=200, verbose=False).solve(
        indefinite, None, rhs
    )
    assert status.reason in ("breakdown", "nonfinite", "nonpos-definite")


@pytest.mark.xfail(reason="Per-RHS status; lands in Step 12")
def test_pcg_batched_per_rhs_status():
    """Batched PCG returns per-RHS residuals, not one aggregate."""
    from laker.solvers import Solve
    n = 6
    K = _psd(n)
    rhs = torch.randn(n, 3)
    _, status = Solve.pcg(tol=1e-8, max_iter=500, verbose=False).solve(
        lambda v: K @ v, None, rhs
    )
    assert status.per_rhs is not None
    assert len(status.per_rhs) == 3


def test_descent_returns_x():
    """Unpreconditioned descent returns a tensor (legacy contract)."""
    from laker.solvers import Solve
    n = 10
    K = _psd(n)
    rhs = torch.randn(n)
    out = Solve.descent(step_size=1.0, max_iter=200, verbose=False).solve(
        lambda v: K @ v, rhs
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (n,)


def test_kernel_matvec_basic():
    """Sanity check on the exact kernel matvec."""
    n = 5
    torch.manual_seed(0)
    e = torch.randn(n, 3)
    op = AttentionKernelOperator(e, lambda_reg=1e-2)
    v = torch.randn(n)
    out = op.matvec(v)
    assert out.shape == (n,)
    assert torch.isfinite(out).all()
