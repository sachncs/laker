# `laker.prec` — preconditioners

Two preconditioner classes: `CCCP` (the learned one) and `Adaptive`
(automatic selection). A free function `apply_core` does the actual
preconditioner application.

## Classes

### `CCCP`

```python
from laker.prec import CCCP
prec = CCCP(
    num=None,            # probe count (auto if None)
    gamma=0.1,            # CCCP shrinkage parameter
    eps=1e-8,
    base=0.05,            # base spectral norm bound
    max_iter=200,
    tol=1e-6,
    verbose=True,
    device=...,
    dtype=...,
    probe="gaussian",     # "gaussian" or "power"
    power=3,              # power iterations for the "power" probe
)
prec.build(operator, n, seed=None)
out = prec.apply(x)  # apply P^{-1/2} to x
prec.dense()           # materialise full n×n matrix
```

#### Constructor parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `num` | None | Number of probe vectors `N_r`. Auto: `max(200, int(2·sqrt(n)))` if `None` |
| `gamma` | 0.1 | CCCP shrinkage parameter `γ` in the regularised MLE |
| `eps` | 1e-8 | Numerical stability floor |
| `base` | 0.05 | Base spectral norm bound `ρ` (in the shrinkage formula) |
| `max_iter` | 200 | CCCP max iterations |
| `tol` | 1e-6 | Convergence tolerance on relative change of `iso` and `b` |
| `verbose` | True | Log per-iteration progress |
| `device` | `Backend.device` | Target torch device |
| `dtype` | `Backend.dtype` | Target torch dtype |
| `probe` | `"gaussian"` | Probe strategy: `"gaussian"` or `"power"` |
| `power` | 3 | Power-iteration count for the `"power"` probe |

#### `build(operator, n, seed=None) -> CCCP`

Build the preconditioner from a matvec. Returns `self` for chaining.

| Argument | Default | Meaning |
|-----------|---------|---------|
| `operator` | required | Callable applying `(G + λI) @ v` |
| `n` | required | Operator dimension |
| `seed` | None | Seed for the random probes (reproducibility) |

#### Attributes set by `build`

| Attribute | Meaning |
|-----------|---------|
| `self.size` | Operator dimension `n` |
| `self.num_probes` | Number of probes actually used (`min(N_r, n)`) |
| `self.basis` | Orthonormal basis `Q ∈ R^{n × N_r}` from QR |
| `self.tri_factor` | Upper-triangular `R` factor from QR |
| `self.iso` | Isotropic coefficient `c` |
| `self.b` | Low-rank perturbation matrix `B` in the Q basis |
| `self.evals` | Eigenvalues of the preconditioner in the Q basis |
| `self.evecs` | Eigenvectors of the preconditioner in the Q basis |

#### `apply(x) -> Tensor`

Apply `P^{-1/2}` to a vector or batch. Accepts 1-D `(n,)` or 2-D
`(n, k)` input.

Raises `RuntimeError` if `build` has not been called.

#### `dense() -> Tensor`

Materialise the full `n × n` preconditioner matrix. Debug-only
because of `O(n²)` memory cost. Raises `RuntimeError` if `build` has
not been called.

### `Adaptive`

```python
from laker.prec import Adaptive
prec = Adaptive(
    gamma=0.1, num=None,
    eps=1e-8, base=0.05,
    max_iter=200, tol=1e-6,
    verbose=True,
    device=..., dtype=...,
    probe="gaussian", power=3,
)
prec.build(operator, n, diag=None, seed=None)
```

Wrapper that runs a power-iteration diagnostic on `min(10, n)`
probes to estimate the condition number `κ`, then selects:

- `Jacobi` (diagonal) if `κ < 10³`
- `CCCP` if `κ < 10⁶`
- `CCCP` with doubled probe budget if `κ ≥ 10⁶`

The selected preconditioner is exposed as `self.inner` and applied
via `apply`.

`diag` is the operator's diagonal (required for the Jacobi
selection to work). If `diag=None` and the operator is well
conditioned, CCCP is used.

### `apply_core(x, iso, basis, evals, evecs) -> Tensor`

Free function that does the preconditioner apply given pre-computed
eigendecomposition. Used internally by `CCCP.apply` and `Adaptive.apply`.

```
P^{-1/2} · x = iso^{-1/2} · x  +  basis · evecs · diag(1/√evals - 1/√iso) · evecs^T · basis^T · x
```

## Why shrinkage matters

Without shrinkage (`base=0`), CCCP produces a maximum-likelihood
estimate of the operator spectrum in the probe span. If the operator
is rank-deficient or has a high condition number, the preconditioner
becomes ill-conditioned itself.

The shrinkage `ρ` blends the MLE estimate with a scaled isotropic
prior. The CCCP iteration drives `ρ` from `base` (well-conditioned)
to a higher value (more shrinkage) when the probe span is too small
to span the operator's range.

`Math.shrink(num_probes, size, gamma, base)` is the public formula
for `ρ`. The CCCP class calls this internally with `base=0.05`,
`gamma=self.gamma`, and clips `ρ ≤ 0.5` (never collapse to isotropic).

## References

- Tao & Tan (2026), Algorithm 1 (lines 4-13). The CCCP iteration
  and the shrinkage regulariser.
- See `algorithms/cccp.md` for the full mathematical treatment.
