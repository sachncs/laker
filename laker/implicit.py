"""Implicit differentiation through PCG fixed-point for hypergradients.

Computes ``dL/d theta_k = -v^T (dA/d theta_k) alpha^*`` via the adjoint
method: one PCG solve for ``v`` plus per-parameter autograd.
"""

from __future__ import annotations

from typing import Callable, List

import torch

from laker.solve import PCG


def hypergradient(
    op: Callable[[torch.Tensor], torch.Tensor],
    prec: Callable[[torch.Tensor], torch.Tensor],
    alpha: torch.Tensor,
    dl: torch.Tensor,
    params: List[torch.Tensor],
    tol: float = 1e-6,
    max_iter: int = 500,
    verbose: bool = False,
) -> List[torch.Tensor]:
    """Compute hypergradients via the adjoint method.

    Args:
        op: Callable applying ``A(theta)``.
        prec: Callable applying the preconditioner.
        alpha: Fixed-point solution ``alpha^*`` (detached).
        dl: Gradient of the outer loss w.r.t. ``alpha``, same shape.
        params: Parameters to differentiate with respect to.
        tol: PCG tolerance for the adjoint solve.
        max_iter: Max iterations for the adjoint solve.
        verbose: Whether to log adjoint progress.

    Returns:
        List of hypergradient tensors (one per parameter, same shape).
    """
    pcg = PCG(tol=tol, max_iter=max_iter, verbose=verbose)
    v, _status = pcg.solve(op=op, prec=prec, rhs=dl)

    alpha_c = alpha.detach()
    v_c = v.detach()

    with torch.enable_grad():
        alpha_g = alpha_c.clone().requires_grad_(True)
        a_alpha = op(alpha_g)
        scalar = torch.dot(v_c, a_alpha)

    grads = []
    for p in params:
        if p.requires_grad:
            g = torch.autograd.grad(scalar, p, retain_graph=True, allow_unused=True)[0]
            if g is None:
                g = torch.zeros_like(p)
            grads.append(-g)
        else:
            grads.append(torch.zeros_like(p))

    return grads


__all__ = ["hypergradient"]