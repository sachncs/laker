"""Solver status tests.

Captures the contract that PCG.solve returns ``(x, status)`` where
``status`` carries ``converged``, ``iterations``, ``residual``,
``reason``, and (for batched 2-D solves) ``per_rhs``.
"""

from __future__ import annotations

import pytest
import torch

from laker.kernel import Exact
from laker.solvers import PreconditionedConjugateGradient as PCG, GradientDescent, Status


def _psd(n: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(n, n)
    return a @ a.T + torch.eye(n)


def test_pcg_returns_x_and_status():
    """``solve`` returns the solution and a ``Status`` object."""
    n = 20
    K = _psd(n)
    rhs = torch.randn(n)
    x, status = PCG(tol=1e-8, max_iter=200, verbose=False).solve(lambda v: K @ v, lambda x: x, rhs)
    assert isinstance(status, Status)
    assert status.converged is True
    assert status.iterations > 0
    assert status.reason == "converged"
    assert x.shape == (n,)


def test_pcg_zero_rhs_returns_zero_rhs_status():
    """Zero RHS yields ``Status(reason="zero_rhs")`` and zero solution."""
    n = 20
    K = _psd(n)
    rhs = torch.zeros(n)
    x, status = PCG(tol=1e-8, max_iter=200, verbose=False).solve(lambda v: K @ v, lambda x: x, rhs)
    assert torch.allclose(x, torch.zeros_like(rhs), atol=1e-7)
    assert status.converged is True
    assert status.reason == "zero_rhs"


def test_pcg_breakdown_on_indefinite():
    """Indefinite operator raises ``RuntimeError`` (audit Step 12)."""
    from laker.solvers import PreconditionedConjugateGradient

    def indefinite(v):
        return torch.tensor([-v[0], v[1], v[2], v[3]], dtype=v.dtype)

    rhs = torch.tensor([1.0, 0.0, 0.0, 0.0])
    pcg = PreconditionedConjugateGradient(tol=1e-12, max_iter=200, verbose=False)
    with pytest.raises(RuntimeError, match="breakdown"):
        pcg.solve(indefinite, lambda x: x, rhs)


def test_pcg_batched_per_rhs_status():
    """Batched 2-D PCG returns one ``Status`` per RHS column."""
    n = 6
    K = _psd(n)
    rhs = torch.randn(n, 3)
    _, status = PCG(tol=1e-8, max_iter=500, verbose=False).solve(lambda v: K @ v, lambda x: x, rhs)
    assert status.per_rhs is not None
    assert len(status.per_rhs) == 3
    assert all(p.converged is True for p in status.per_rhs)


def test_descent_returns_x():
    """Unpreconditioned descent returns a tensor (legacy contract)."""
    n = 10
    K = _psd(n)
    rhs = torch.randn(n)
    out = GradientDescent(step_size=1.0, max_iter=200, verbose=False).solve(lambda v: K @ v, rhs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (n,)


def test_kernel_matvec_basic():
    """Sanity check on the exact kernel matvec."""
    n = 5
    torch.manual_seed(0)
    e = torch.randn(n, 3)
    op = Exact(e, lambda_reg=1e-2)
    v = torch.randn(n)
    out = op.matvec(v)
    assert out.shape == (n,)
    assert torch.isfinite(out).all()
