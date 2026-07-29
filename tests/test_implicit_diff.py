"""Tests for implicit differentiation (hypergradient computation)."""

import torch

from laker.implicit_diff import hypergradient


def test_hypergradient_diagonal():
    """Hypergradient for a diagonal operator should match analytic formula.

    For A = diag(a) where a is learnable, and loss = 0.5 * ||alpha||^2,
    the hypergradient dL/da_i = -v_i * alpha_i where v = A^{-1} dL_dalpha.
    For L = 0.5 * ||alpha||^2, dL_dalpha = alpha.
    """
    torch.manual_seed(42)
    n = 10
    diag = torch.exp(torch.randn(n, requires_grad=True))  # positive
    A = torch.diag(diag)
    y = torch.randn(n)
    alpha = torch.linalg.solve(A, y)

    def op(v):
        return A @ v

    dL_dalpha = alpha.clone()
    grads = hypergradient(op, op, alpha, dL_dalpha, [diag], verbose=False)

    assert len(grads) == 1
    v = torch.linalg.solve(A, dL_dalpha)
    expected = -v * alpha
    torch.testing.assert_close(grads[0], expected, rtol=1e-4, atol=1e-4)
