# CCCP preconditioner

The dominant cost of LAKER is solving `(G + λ I) α = y`. Plain PCG
on this system takes hundreds to thousands of iterations because
`G = exp(E E^T)` is ill-conditioned. The CCCP preconditioner
(shrinkage-regularised Convex-Concave Procedure) builds a data-
dependent approximation `Σ ≈ G` in factored form and uses `Σ^{-1/2}`
as the PCG preconditioner, reducing iteration count by one to three
orders of magnitude.

## Mathematical formulation

The CCCP preconditioner learns a low-rank-plus-isotropic approximation
of the operator applied to random probe vectors.

### Probe application

Generate `N_r` random probe directions `p_1, ..., p_{N_r} ∈ R^n` and
form the matrix

```
P = [ A p_1  A p_2  ...  A p_{N_r} ]  ∈  R^{n × N_r}
```

where `A = G + λ I` is the operator. We never materialise `A`, only
apply it via `matvec`.

### Normalised QR

Normalise the columns of `P` to unit norm and compute the economy QR
factorisation

```
P̂ = Q R,    Q ∈ R^{n × N_r},  R ∈ R^{N_r × N_r}
```

with `Q^T Q = I` and `R` upper-triangular.

### CCCP iteration

The CCCP iteration solves a regularised maximum-likelihood problem
for the spectrum of the operator restricted to the span of `Q`. The
iteration is:

```
For it = 0, 1, 2, ... until convergence:
  M = c · I  +  B                       (low-rank + isotropic)
  E, V = eigh(M)                         (eigendecompose in Q basis)
  proj = V^T · R                          (R projected into eigenbasis)
  scaled = 1/λ_i · proj                   (scale by inverse eigenvalues)
  inv_mr = proj^T · scaled
  denoms = diag(inv_mr) + ε
  w_k = (n / N_r) / denoms               (probe weights)
  F_γ = (1 / (1 + γ/n)) · (Σ_k w_k · q_k q_k^T + γ I)   (Q basis)
  F_shr = (1 - ρ) · F_γ + ρ · I                       (shrinkage)
  trace = (1 - ρ) · γ / (1 + γ/n) + ρ
  full_trace = trace · (n - N_r) + tr(F_shr)
  scale = n / full_trace
  c = scale · trace
  B = scale · F_shr - c · I
  Check convergence on (c, B).
```

The shrinkage factor `ρ` (controlled by `base` and `gamma`) blends the
Frobenius-regularised maximum-likelihood estimate with a trivial
isotropic estimate, ensuring the preconditioner is well-conditioned
even when the operator is rank-deficient in the probe span.

After convergence, the preconditioner is applied as

```
P^{-1/2} = c^{-1/2} · I  +  Q · V · diag(1/√λ - 1/√c) · V^T · Q^T
```

which is `O(n · N_r)` per application. The total preconditioner-build
cost is `O(N_r² · n + N_r³ · iters)`, independent of `n` beyond the
probe applications.

## Implementation in `laker.prec.CCCP`

The class is in `laker.prec.CCCP`. Usage:

```python
from laker.prec import CCCP

prec = CCCP(
    num=100,            # number of probe vectors N_r
    gamma=0.1,          # CCCP shrinkage parameter
    base=0.05,          # base spectral norm bound
    max_iter=200,        # CCCP max iterations
    tol=1e-6,           # convergence tolerance
    verbose=False,
    device=...,
    dtype=...,
)
prec.build(operator, n, seed=0)
# Apply: prec.apply(x)
```

Attributes set by `build`:
- `self.size` — operator dimension `n`
- `self.num_probes` — number of probes actually used (`min(N_r, n)`)
- `self.basis` — orthonormal basis `Q ∈ R^{n × N_r}` from QR
- `self.tri_factor` — upper-triangular factor `R` from QR
- `self.iso` — final isotropic coefficient `c`
- `self.b` — low-rank perturbation matrix `B` in the Q basis
- `self.evals`, `self.evecs` — eigen-decomposition of the preconditioner
  in the Q basis (cached for fast apply)

The `apply` method dispatches on `x.dim()` (1-D vector or 2-D batch)
and uses a shared `apply_core` helper that computes the
preconditioner apply from the cached `Q` and eigendecomposition.

The `dense` method materialises the full `n × n` preconditioner
matrix (debugging only — costs `O(n²)`).

## Probe strategy

Three probe strategies are available via the `probe` argument:
- `"gaussian"` (default) — pure i.i.d. `N(0, 1)` probes. The standard
  unbiased spectrum coverage.
- `"power"` — first 25 % of probes are power-iterated to concentrate
  on the dominant eigenvectors. The remainder stay Gaussian for
  unbiased coverage.

## `Adaptive` policy

`laker.prec.Adaptive` runs power iteration on `min(10, n)` random
probes, estimates the condition number, and selects:
- `Jacobi` preconditioner if `κ < 10³`
- `CCCP` if `κ < 10⁶`
- `CCCP` with doubled probe count if `κ ≥ 10⁶`

This is a useful default when the user does not know the operator's
spectrum in advance. The selected preconditioner is exposed as
`self.inner` and dispatched via `apply`.

## Why shrinkage matters

Without shrinkage (`base = 0`, `gamma = 0`), CCCP is the empirical
maximum-likelihood estimate of the spectrum. When the operator is
near-rank-deficient, this estimate has zero eigenvalues in the
orthogonal complement of the probe span, and `Σ^{-1/2}` blows up.

Shrinkage blends `Σ_kl` towards a scaled isotropic matrix:
`Σ_shr = (1 - ρ) · Σ_ml + ρ · (n / trace) · I`. The shrinkage
parameter `ρ` is set via `Math.shrink(num_probes, n, gamma, base)`:
`ρ = base + (1 - base) · (1 - num / n) · min(1, 10 · gamma)`,
clipped to `0.5`. This ensures:
- `ρ = base` when `num ≥ n` (full-rank operator)
- `ρ > base` when `num < n` and the problem is not yet well-conditioned
- `ρ ≤ 0.5` always (never collapse to isotropic)

## References

- Tao & Tan (2026), Algorithm 1 (lines 4-13). The CCCP iteration
  and the shrinkage regulariser.
- Yuille & Rangarajan (2003), *"The Convex-Concave Procedure"*. The
  original CCCP framework.
- Saul et al. (2003), *"Maximum likelihood and minimum
  free-energy formulations of learning"*. The connection between
  CCCP and MLE for latent-variable models.
