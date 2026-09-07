# `laker.kernel` — kernel operators

Seven kernel operators implementing the same protocol:

```python
op.matvec(x)        # apply operator to vector or batch
op.diag()           # return diagonal
op.dense()          # materialise full n×n matrix
op.eval(x, y=None)  # evaluate K(x, y) (no diagonal)
```

All operators store their embeddings, regularization weight, and
metadata as instance attributes.

## Classes

### `Kernel` (Protocol)

A structural type describing the four required methods. Not
instantiated.

### `Exact`

```python
from laker.kernel import Exact
k = Exact(embeddings, lam=0.01, chunk=None, dtype=torch.float64)
```

Direct exponential attention kernel `G = exp(E E^T)` with optional
chunked matvec. Most accurate, `O(n²)` cost.

`chunk=None` auto-selects based on memory budget (`Backend.chunk`).
`chunk=N` uses N-sized tiles.

### `Nystrom`

```python
from laker.kernel import Nystrom
k = Nystrom(embeddings, lam=0.01, num=200, method="greedy", dtype=torch.float64)
```

Nyström low-rank approximation `G ≈ K_nm K_mm^{-1} K_nm^T` with
`method="greedy"` (default) or `method="leverage"` landmark selection.

### `Fourier`

```python
from laker.kernel import Fourier
k = Fourier(embeddings, lam=0.01, num=64, dtype=torch.float64)
```

Random Fourier feature approximation. Reproducible via internal seed.

### `Neighbors`

```python
from laker.kernel import Neighbors
k = Neighbors(embeddings, lam=0.01, k=10, dtype=torch.float64)
```

Sparse top-`k` Euclidean k-NN graph, symmetrised, made positive
definite via diagonal dominance. `matvec` uses `torch.sparse.mm`.

### `Grid`

```python
from laker.kernel import Grid
k = Grid(embeddings, lam=0.01, grid_size=64, dtype=torch.float64)
```

Structured Kernel Interpolation (SKI) on a product grid. Use
`grid_size=64` for `d=2`, larger for higher `d`.

### `Hybrid`

```python
from laker.kernel import Hybrid
k = Hybrid(embeddings, lam=0.01, alpha=0.5, num=200, k=10, dtype=torch.float64)
```

Two-scale Nyström + Neighbors with `alpha` blend.

### `Spectrum`

```python
from laker.kernel import Spectrum
k = Spectrum(embeddings, lam=0.01, knots=5, dtype=torch.float64)
```

Spectral-shaped kernel: `K = U diag(exp(g(σ²))) U^T` with `g` a
learned monotone spline.

### `Shaper`

```python
from laker.kernel import Shaper
s = Shaper(knots=5)
s.set(lo=0.0, hi=1.0, device="cpu", dtype=torch.float64)
out = s(x)  # monotone function evaluated at x
```

The learned spline used by `Spectrum`. Generally you don't need to
instantiate this directly.

## Free functions

### `exp_safe(gram, out=None, skip=False) -> Tensor`

Element-wise `exp` with dtype-aware overflow clamp. Used internally
by all kernel operators.

| Argument | Default | Meaning |
|-----------|---------|---------|
| `gram` | required | Input tensor |
| `out` | None | If given, write the result in-place into this tensor |
| `skip` | False | If `True`, skip the clamp (use only when in-range is guaranteed) |

Clamp caps: `float16` → 11, `float32` / `bfloat16` → 80, `float64` →
700. Skips the clamp entirely for `gram.requires_grad` tensors
(autograd-incompatible with the in-place form) or when `skip=True`.

### `exact_matvec(embeddings, lam, x, skip=False) -> Tensor`

Non-chunked matvec for the exact kernel. Used internally by
`Exact.matvec` and as a hot path for small problems.

### `weights(x, grid) -> (indices, weights)`

Multilinear interpolation weights for SKI grid interpolation. Given
`x: (n, d)` in `[0, 1]^d` and `grid: list[d]` of sorted 1-D grid
coordinates, returns `(indices, weights)` each of shape `(n, 2^d)`.
Row sums of `weights` are exactly 1.

## Selecting a kernel

See [Choosing a kernel](../guides/choosing_kernel.md) for the
speed/accuracy trade-off table and when to use which.

## Numerical accuracy

| Kernel | Typical `matvec` error vs `Exact` |
|--------|----------------------------------|
| `Exact` | 0 (reference) |
| `Nystrom` (`num=200`) | `1e-7` |
| `Fourier` (`num=128`) | `1e-5` |
| `Neighbors` (`k=10`) | `1e-3` (sparse) |
| `Grid` (`grid_size=64`) | `1e-3` |
| `Spectrum` | `1e-6` |

Tested in `tests/test_kernel.py`.
