# `laker.math` — numerical helpers and GP surrogate

`Math` is a single class with static methods (pure functions, no
state). `GP` is a standalone class for Bayesian-optimisation
surrogates.

## `Math` (static methods)

### `Math.exp(gram, skip=False) -> Tensor`

Element-wise `exp` with dtype-aware overflow clamp. Equivalent to
`laker.kernel.exp_safe` but returns a fresh tensor (always
non-mutating).

| dtype | Clamp cap |
|-------|-----------|
| float16 | 11 |
| bfloat16 | 80 |
| float32 | 80 |
| float64 | 700 |

When `gram.requires_grad=True` or `skip=True`, the clamp is bypassed
(use only when in-range is guaranteed). See
[api/kernel.md#exp_safe](kernel.md#free-functions) for details.

### `Math.pdf(x) -> Tensor`

Standard normal PDF: `exp(-x²/2) / sqrt(2π)`.

### `Math.cdf(x) -> Tensor`

Standard normal CDF via the Abramowitz-Stegun 7.1.26
approximation:

```
CDF(x) = 1 - φ(|x|) · poly(1/(1 + 0.2316·|x|))    for x ≥ 0
CDF(x) =     φ(|x|) · poly(1/(1 + 0.2316·|x|))    for x < 0
```

where `poly(t) = 0.319·t - 0.357·t² + 1.781·t³ - 1.821·t⁴ + 1.330·t⁵`.
Approximation error is `O(1e-7)`.

### `Math.sinh(x) -> Tensor`

Stable `sinh` (currently a thin wrapper over `torch.sinh`).

### `Math.normalize(mat) -> Tensor`

Scale a positive-definite matrix so `trace = n` where `n` is the
matrix dimension. Returns the input unchanged if the trace is
below `1e-30`.

### `Math.latin(n, dims, seed=None) -> np.ndarray`

Latin-hypercube sample of shape `(n, dims)` in `[0, 1]`. Used by
`Search.bayes` for the initial-point design.

### `Math.power(operator, n, num=10, seed=0) -> float`

Estimate the spectral norm of a linear operator via power
iteration. Returns the operator norm estimate.

### `Math.chol(matrix, eps=1e-8) -> Tensor`

Cholesky factorisation with diagonal-jitter fallback. If the input
is not positive-definite (e.g. negative entries on the diagonal),
retries with `matrix + eps · I`.

### `Math.bytes(dtype) -> int`

Bytes per scalar for memory budget calculations. Returns
`dtype.itemsize` if available, else 4.

### `Math.inv(diagonal, eps=1e-12) -> Tensor`

Invert a 1-D diagonal, clamping small / negative entries to `eps`.

### `Math.seed() -> Optional[int]`

Return the global LAKER seed if set, else `None`.

### `Math.seed_set(value) -> None`

Seed torch, numpy, and the `LAKER_SEED` env var.

### `Math.dense(sparse) -> Tensor`

Materialise a sparse tensor to dense; no-op for dense inputs.

### `Math.shrink(probes, size, gamma, base=0.05) -> float`

Adaptive shrinkage parameter for the CCCP preconditioner. Returns:

- `base` if `probes >= size` (full-rank operator)
- `base + (1 - base) · (1 - probes/size) · min(1, 10·gamma)` otherwise
- Clipped to `[0, 0.5]`

### `Math.eigh(matrix, eps=1e-10) -> (Tensor, Tensor)`

Eigendecomposition with eigenvalue clamping for PSD safety.
Equivalent to `torch.linalg.eigh` but with `min=eps` on the
eigenvalues. Returns `(eigenvalues, eigenvectors)`.

## `GP` (class)

```python
from laker.math import GP
gp = GP(
    bounds,             # np.ndarray of shape (d, 2)
    log=[0, 1],         # indices of dimensions to log-transform
    sigma=1.0,          # signal std
    scale=0.2,          # RBF length scale
    noise=1e-4,         # observation noise std
)
```

| Method | Signature | Meaning |
|--------|-----------|---------|
| `predict` | `(X_new) -> (mean, var)` | Posterior predictive mean and variance |
| `fit` | `(X, y)` | Tune length scale by maximising marginal likelihood; then fit |
| `improve` | `(X_new, xi=0.01) -> ei` | Expected Improvement at each candidate |

The internal length scale is tuned by grid search over
`logspace(-2, 0, 20)` (20 candidates) maximising the marginal
likelihood. The GP uses an RBF kernel with optional log-transform on
selected dimensions.

## Free functions

### `pdf_np(x) -> np.ndarray`

Standard normal PDF for numpy arrays.

### `cdf_np(x) -> np.ndarray`

Standard normal CDF via Abramowitz-Stegun 7.1.26 for numpy arrays.
Used internally by `GP.improve`.

## Where they're used

- `Math.exp` is used everywhere `exp_safe` is used in the kernel
  operators.
- `Math.shrink` is used by `laker.prec.CCCP` to set the CCCP shrinkage
  parameter `rho`.
- `Math.normalize` is used in the bilinear-shape kernel construction
  (legacy code).
- `Math.eigh` is used by `Math.eigh`, `laker.prec.CCCP`, and
  `laker.kernel.Spectrum`.
- `GP` is used by `laker.search.Search.bayes` as the surrogate.
