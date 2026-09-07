# Preconditioned Conjugate Gradient

PCG is the only iterative solver LAKER uses for the linear system
`(G + λ I) α = y`. The CCCP preconditioner
([algorithms/cccp.md](cccp.md)) makes PCG converge in `O(1)` to
`O(N_r)` iterations, where `N_r` is the probe count, regardless of
the problem size `n`. This document covers the algorithm and the
implementation in `laker.solve.PCG`.

## Algorithm

PCG solves the symmetric positive-definite system `A x = b` by
iteratively refining an estimate of `x` in the direction of decreasing
residual. The matrix `A` is only accessed via its matvec.

```
x_0 = 0                          (cold start; or user-provided warm start)
r_0 = b - A x_0 = b               (initial residual)
z_0 = P^{-1} r_0                 (preconditioned residual)
p_0 = z_0
ρ_0 = r_0^T z_0

For k = 0, 1, 2, ... until convergence:
  A p_k = A · p_k                 (one matvec with the operator)
  α_k = ρ_k / (p_k^T A p_k)        (step size)
  x_{k+1} = x_k + α_k · p_k
  r_{k+1} = r_k - α_k · A p_k
  z_{k+1} = P^{-1} r_{k+1}
  ρ_{k+1} = r_{k+1}^T z_{k+1}
  β_k = ρ_{k+1} / ρ_k
  p_{k+1} = z_{k+1} + β_k · p_k
```

The loop terminates when `‖r_k‖ / ‖b‖ ≤ tol` or `k ≥ max_iter`. The
preconditioner `P^{-1}` is applied on every iteration, and the cost of
`P^{-1}` dominates the cost of the matvec when `N_r ≪ n`.

## Implementation in `laker.solve.PCG`

The class is in `laker.solve.PCG`. Usage:

```python
from laker.solve import PCG

pcg = PCG(tol=1e-6, max_iter=1000, verbose=False)
x, report = pcg.solve(op, prec, rhs, x0=None)
print(f"converged in {pcg.iterations} iterations, residual={report.residual:.2e}")
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tol` | 1e-10 | Relative residual tolerance `‖r‖ / ‖b‖` |
| `max_iter` | None | Max iterations; default `n` (the system dimension) |
| `verbose` | True | Log convergence progress |
| `restart` | None | Recompute residual every `restart` iterations (mitigates round-off) |
| `eps` | None | Breakdown threshold; default `sqrt(finfo(dtype).eps)` |
| `autograd` | False | Use out-of-place updates for autograd compatibility |

`PCG.solve` returns `(solution, Report)` where `Report` is a
`@dataclass` with:
- `converged: bool`
- `iterations: int`
- `residual: float` (relative residual)
- `reason: str` (`"converged"` / `"max_iter"` / `"zero_rhs"` / `"breakdown"`)
- `per: Optional[List[Report]]` — per-RHS statuses for 2-D batch solves

## Batched RHS

When `rhs` is 2-D with shape `(n, k)`, PCG solves the `k` systems
simultaneously using column-wise vectorised dot products. The result
`x` has shape `(n, k)` and `report.per` has `k` per-RHS reports.

## Warm start

`PCG.solve(..., x0=guess)` uses `guess` as the initial iterate. The
residual is recomputed exactly as `r_0 = b - A x_0` to avoid carrying
forward stale residual state. This is the warm-start path used by
`Laker.path` and `Laker.continuation` between successive `lam` values.

## Breakdown detection

The algorithm raises `RuntimeError("PCG breakdown at iteration k:
non-positive curvature detected")` when
`p^T A p ≤ -ε · ‖p‖ · ‖A p‖` (the Cauchy-Schwarz bound). This
indicates the operator is indefinite or the preconditioner is
unsuitable. The default `eps = sqrt(finfo(dtype).eps)` is a tight
threshold matching the rounding-error floor of the dot product.

## Restart (optional)

For long iterations in `float64`, periodic exact residual
recomputation (`restart=k`) can reduce round-off accumulation. The
default `restart=None` skips this — useful in `float32` where the
subtraction `b - A x` can catastrophically cancel.

## Zero RHS

If `b = 0` (or its norm rounds to zero), PCG short-circuits to
`x = 0, report.reason = "zero_rhs"` without any matvec.

## `Descent` and `Jacobi`

`laker.solve.Descent` is bare gradient descent, used as a slow
worst-case reference in `BaseBench`. It auto-estimates the step size
via 5 rounds of power iteration.

`laker.solve.Jacobi` is the diagonal preconditioner
`P = diag(A)^{-1}`. Used in the same `BaseBench` as a fast but weak
preconditioner.

## References

- Hestenes & Stiefel (1952), *"Methods of Conjugate Gradients for
  Solving Linear Systems"*. The original CG paper.
- Saad (2003), *"Iterative Methods for Sparse Linear Systems"*. The
  standard reference; PCG with preconditioning is in Chapter 9.
- Shewchuk (1994), *"An Introduction to the Conjugate Gradient Method
  Without the Agonizing Pain"*. The classic tutorial.
