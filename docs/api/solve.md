# `laker.solve` — iterative solvers

Three solver classes: `PCG` (production), `Descent` (slow reference),
`Jacobi` (diagonal preconditioner). One `Report` dataclass.

## Classes

### `PCG`

```python
from laker.solve import PCG
pcg = PCG(tol=1e-6, max_iter=1000, verbose=False)
x, report = pcg.solve(op, prec, rhs, x0=None)
```

Preconditioned Conjugate Gradient. Used everywhere LAKER solves a
linear system.

#### Constructor

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tol` | 1e-10 | Relative residual tolerance `‖r‖ / ‖b‖` |
| `max_iter` | None | Max iterations; default `n` (the system dimension) |
| `verbose` | True | Log convergence progress |
| `restart` | None | Recompute residual every `restart` iterations |
| `eps` | None | Breakdown threshold; default `sqrt(finfo(dtype).eps)` |
| `autograd` | False | Use out-of-place updates for autograd |

#### Methods

##### `PCG.solve(op, prec, rhs, x0=None) -> (Tensor, Report)`

Solve `A x = rhs`.

| Argument | Default | Meaning |
|----------|---------|---------|
| `op` | required | Callable applying `A @ v` |
| `prec` | required | Callable applying `P^{-1} @ v` |
| `rhs` | required | Right-hand side `(n,)` or `(n, k)` |
| `x0` | None | Optional warm-start iterate |

Returns `(solution, Report)`. Raises `ValueError` if `rhs.dim() not
in (1, 2)`. Raises `RuntimeError` on breakdown.

#### Attributes after `solve`

| Attribute | Meaning |
|-----------|---------|
| `self.iterations` | Number of PCG iterations executed |
| `self.residual` | Final relative residual |
| `self.max_iter` | ... |
| `self.tol` | ... |

### `Descent`

```python
from laker.solve import Descent
gd = Descent(tol=1e-3, max_iter=50000, verbose=False)
x = gd.solve(op, rhs, x0=None)
```

Bare gradient descent. Used in `BaseBench` as a slow worst-case
reference. Auto-estimates the step size via 5 power iterations if
not given.

### `Jacobi`

```python
from laker.solve import Jacobi
J = Jacobi(diag)  # diag: the diagonal of (G + λI)
out = J.apply(x)   # element-wise multiply by 1/diag
```

Diagonal preconditioner `P = diag(A)^{-1}`. Used in `BaseBench` and
as the low-budget branch of `Adaptive`.

Constructor takes the diagonal vector; `apply` does an
element-wise multiplication.

### `Report`

```python
from laker.solve import Report
r = Report(converged=True, iterations=10, residual=1e-8, reason="converged")
```

Dataclass returned by `PCG.solve`. Fields:

| Field | Meaning |
|-------|---------|
| `converged` | `True` if relative residual met `tol` |
| `iterations` | Number of iterations executed |
| `residual` | Final `‖r‖ / ‖b‖` |
| `reason` | `"converged"` / `"max_iter"` / `"zero_rhs"` / `"breakdown"` |
| `per` | For 2-D RHS: list of per-RHS `Report`s; `None` for 1-D |

## Batched RHS

When `rhs` is 2-D `(n, k)`, PCG solves the `k` systems
simultaneously using column-wise vectorised operations. The solution
is `(n, k)` and `report.per` has `k` per-RHS reports.

## Warm start

`PCG.solve(..., x0=guess)` uses `guess` as the initial iterate. The
residual is recomputed exactly as `r_0 = b - A x_0` to avoid carrying
forward stale state. Used by `Laker.path` and `Laker.continuation`
between successive `lam` values.

## Breakdown detection

PCG raises `RuntimeError("PCG breakdown at iteration k: ...")` when
`p^T A p ≤ -ε · ‖p‖ · ‖A p‖` (the Cauchy-Schwarz bound). The default
`eps = sqrt(finfo(dtype).eps)` matches the rounding-error floor of
the dot product.

## Zero RHS

If `b = 0`, PCG short-circuits to `x = 0, report.reason = "zero_rhs"`
without any matvec.

## References

- See `algorithms/pcg.md` for the full mathematical treatment.
- Hestenes & Stiefel (1952), *"Methods of Conjugate Gradients for
  Solving Linear Systems"*. The original CG paper.
