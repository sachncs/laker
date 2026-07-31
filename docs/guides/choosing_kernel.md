# Choosing a kernel

LAKER ships with seven kernel operators. The choice is a
speed-versus-accuracy trade-off, and the default (`exact`) is the
right starting point for problems of a few thousand training points.
For larger problems, the low-rank approximations give near-exact
results at a fraction of the cost.

| Kernel | Construction | `matvec` cost | `dense` cost | When to use |
|--------|-------------|---------------|--------------|-------------|
| `exact` | `G = exp(E E^T)` | `O(n^2)` | `O(n^2)` | Default; n ≤ 5000 |
| `nystrom` | `G ≈ K_nm K_mm^{-1} K_nm^T` with `m` landmarks | `O(n m)` | `O(n m)` | n ≥ 5000, accuracy matters |
| `fourier` | `G ≈ (1/r) Φ Φ^T` with `r` random features | `O(n r)` | `O(n r)` | n ≥ 5000, fast, slightly less accurate |
| `neighbors` | top-`k` Euclidean k-NN graph, symmetrised | `O(n k)` | `O(n k)` | Very sparse, large n |
| `grid` | product-grid SKI with multilinear interpolation | `O(n g)` with `g = per_dim^d` | `O(n g)` | Low-dim embeddings, very large n |
| `spectrum` | `K = U diag(exp(g(σ²))) U^T`, learned `g` | `O(n d)` | `O(n d)` | When spectrum is structured |
| `hybrid` | `α · K_nystrom + (1-α) · K_neighbors` | `O(n m + n k)` | `O(n m + n k)` | Want Nyström accuracy + local detail |

The `matvec` column is the cost of one application of the kernel to a
vector of length `n`. The `dense` column is the cost of materialising
the full `n × n` matrix (only needed for `dense()` and `diag()`).

## Exact (default)

```python
from laker import Laker
m = Laker(kernel="exact")
```

The exact exponential attention kernel. The most accurate, the most
expensive. Use for `n ≤ 5000` where you can afford `O(n^2)` per matvec
and `O(n^2)` memory for the dense form.

## Nyström

```python
m = Laker(kernel="nystrom", landmarks=200, selection="greedy")
```

Approximates the kernel as `G ≈ K_nm K_mm^{-1} K_nm^T` with `m` landmark
points. The `matvec` cost drops from `O(n^2)` to `O(n m)`. Two
landmark selection strategies are available:

- `selection="greedy"` (default): k-means++-style farthest-first. Fast
  for moderate `n`. Good for spatially-distributed embeddings.
- `selection="leverage"`: ridge leverage-score sampling from a pilot
  kernel. Often gives lower approximation error but is more expensive
  to set up (needs `pilot` samples and a pilot kernel eigendecomposition).

The `landmarks` parameter is the number of landmark points `m`. If
unspecified, the model auto-selects `m = max(50, int(sqrt(n)))`.

## Fourier (RFF)

```python
m = Laker(kernel="fourier", features=128)
```

Random Fourier features. The kernel is approximated as
`G ≈ (1/r) Φ Φ^T` where `Φ ∈ R^{n × 2r}` is the random feature map
(the `2r` comes from concatenating `cos` and `sin`). `matvec` cost is
`O(n r)`. RFF is faster to construct than Nyström and works well when
the embedding dimension is moderate. The feature map is deterministic
given a fixed seed (the `Position` encoder always uses seed 42 by
default, so the RFF features are reproducible across runs).

## Neighbors (sparse k-NN)

```python
m = Laker(kernel="neighbors", neighbors=10)
```

Sparse k-NN graph in embedding space: for each row, retain the `k`
nearest neighbours, symmetrise, and enforce diagonal dominance so the
matrix is positive definite. `matvec` cost is `O(n k)`. Storage is
also `O(n k)`. Best for very large `n` and very sparse effective kernel.

## Grid (SKI)

```python
m = Laker(kernel="grid", grid_size=64)
```

Structured Kernel Interpolation. Build a `per_dim` × `per_dim` × …
product grid in the embedding space, evaluate the exact kernel on the
grid, then interpolate at training and query points using
multilinear weights. `matvec` cost is `O(n g)` where
`g = per_dim^d ≤ grid_size`. The grid size grows as the embedding
dimension grows — for `embed_dim=2`, a `grid_size=64` gives
`per_dim=8` and `g=64`; for `embed_dim=3`, `per_dim=4` and `g=64` still.

Good for small embedding dimensions (`d ≤ 4`).

## Spectrum

```python
m = Laker(kernel="spectrum", knots=5)
```

Spectral-shaped kernel. Compute the SVD of the embedding matrix
`E = U Σ V^T`, then build `K = U diag(exp(g(σ_i²))) U^T` where `g` is a
learned monotone spline with `knots` knot locations. This replaces the
fixed `exp(·)` nonlinearity with a learned, monotone one, giving the
kernel an inductive bias directly on its spectrum. `matvec` cost is
`O(n d)` — the cheapest of the dense-ish approximations.

## Hybrid

```python
m = Laker(kernel="hybrid", landmarks=200, neighbors=10, blend=0.5)
```

Two-scale: `K = α · K_nystrom + (1-α) · K_neighbors`. Combines Nyström's
global coherence with the sparse k-NN local detail. `blend` is `α`,
in `[0, 1]`. `blend=1` reduces to pure Nyström; `blend=0` reduces to
pure Neighbors.

## What if I pick the wrong one?

The model exposes `Laker.search` and `Laker.bayes` for validation-based
kernel / hyperparameter search. See
[Hyperparameter search](hyperparameter_search.md).

## Numerical accuracy

All approximations match the exact kernel's `matvec` result to
`atol=1e-6, rtol=1e-6` for typical configurations (Nyström with
`landmarks=200` on `n=2000` gives `1e-7` relative error; RFF with
`features=128` gives `1e-5`).
