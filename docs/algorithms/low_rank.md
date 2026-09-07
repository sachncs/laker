# Low-rank kernel approximations

When `n` is too large for the exact `G = exp(E E^T)` (a few thousand
points), LAKER offers six low-rank approximations. All live in
`laker.kernel` and share the same `Kernel` protocol:

```python
op.matvec(x)        # O(cost) per call
op.diag()           # diagonal of the operator, O(cost)
op.dense()          # full n×n matrix, O(n²) memory
op.eval(x, y)       # K(x, y) without diagonal
```

The implementations are:

| Class | Approximation | `matvec` cost |
|-------|--------------|---------------|
| `Exact` | `G = exp(E E^T)` | `O(n²)` |
| `Nystrom` | `G ≈ K_nm K_mm^{-1} K_nm^T` | `O(n m)` |
| `Fourier` | `G ≈ (1/r) Φ Φ^T` with `Φ ∈ R^{n × 2r}` | `O(n r)` |
| `Neighbors` | top-`k` Euclidean k-NN graph | `O(n k)` |
| `Grid` | SKI on product grid, multilinear interpolation | `O(n g)` with `g ≤ grid_size` |
| `Spectrum` | `K = U diag(exp(g(σ²))) U^T` | `O(n d)` |
| `Hybrid` | `α · K_nystrom + (1-α) · K_neighbors` | `O(n m + n k)` |

When to use which is covered in
[Choosing a kernel](../guides/choosing_kernel.md).

## `Nystrom`

```python
from laker.kernel import Nystrom
k = Nystrom(embeddings, lam=1e-2, num=200, method="greedy", dtype=torch.float64)
```

The Nyström approximation is

```
G ≈ K_nm · K_mm^{-1} · K_nm^T
```

where `K_nm` is the `n × m` cross-kernel between data and landmarks,
and `K_mm` is the `m × m` landmark kernel. The precomputed
`K_nm K_mm^{-1} ∈ R^{n × m}` is stored as `self.landmark_projection`
so each `matvec` is `K_nm @ landmark_projection.T @ x`.

Two landmark selection strategies:
- `greedy` (default): k-means++ style farthest-first from the
  embeddings. Fast and well-distributed.
- `leverage`: ridge-leverage-score sampling from a pilot kernel.
  Often lower approximation error but needs a pilot eigendecomposition
  on a `pilot`-sized subsample.

`num` defaults to `max(50, int(sqrt(n)))` if not specified.

## `Fourier`

```python
from laker.kernel import Fourier
k = Fourier(embeddings, lam=1e-2, num=64, dtype=torch.float64)
```

Approximates the attention kernel using random Fourier features:

```
Φ = [cos(E ω + b) ; sin(E ω + b)] / sqrt(r)   ∈ R^{n × 2r}
G ≈ Φ Φ^T
```

where `ω ∈ R^{d × r}` and `b ∈ R^r` are drawn from `N(0, 1)` and
`Uniform(0, 2π)` respectively, with a fixed internal seed (42) for
reproducibility. `matvec(x) = λ x + Φ (Φ^T x)` costs `O(n r)`.

`num` defaults to `max(100, int(sqrt(n) * 2))` if not specified.

## `Neighbors`

```python
from laker.kernel import Neighbors
k = Neighbors(embeddings, lam=1e-2, k=10, dtype=torch.float64)
```

Sparse k-NN graph in embedding space:
1. For each row, compute Euclidean distances to all other rows.
2. Keep only the top-`k` nearest neighbours (top-k of distances).
3. Compute the exponential kernel `exp(E_i · E_j)` for those pairs.
4. Symmetrise: for each edge `(i, j)`, ensure both `(i, j)` and
   `(j, i)` are present.
5. Enforce diagonal dominance: the diagonal `G_{ii}` is set to at
   least `1.01 · Σ_{j ≠ i} |G_{ij}| + ε`, guaranteeing positive
   definiteness.

The result is a sparse COO tensor with `O(n k)` non-zeros. `matvec`
uses `torch.sparse.mm` and costs `O(n k)`.

`k` defaults to `min(50, n)`. The `chunk` parameter controls the
distance-computation tiling.

## `Grid` (SKI)

```python
from laker.kernel import Grid
k = Grid(embeddings, lam=1e-2, grid_size=64, dtype=torch.float64)
```

Structured Kernel Interpolation:
1. Build a `per_dim × per_dim × ...` product grid in embedding
   space, where `per_dim = floor(grid_size^(1/d))`.
2. Compute the exact exponential kernel on the grid points
   (`K_grid ∈ R^{g × g}` with `g = per_dim^d`).
3. Compute multilinear interpolation weights `W` from training points
   to grid points.
4. `matvec(x) = W · K_grid · (W^T · x)` costs `O(n g)`.

For `d = 2`, `grid_size=64` gives `per_dim=8, g=64`. For `d = 3`,
`grid_size=64` gives `per_dim=4, g=64` still. The grid size grows
quickly with `d`, so SKI is best for `d ≤ 4`.

## `Spectrum`

```python
from laker.kernel import Spectrum
k = Spectrum(embeddings, lam=1e-2, knots=5, dtype=torch.float64)
```

Spectral-shaped kernel:
1. SVD: `E = U Σ V^T`. The eigenvalues of `E E^T` are `σ_i²`.
2. Apply a learned monotone spline `g(·)` to `σ_i²` to get shaped
   values `s_i = exp(g(σ_i²))`.
3. `K = U · diag(s) · U^T` so `K` is PSD by construction.
4. `matvec(x) = λ x + U · (diag(s) · (U^T · x))` costs `O(n d)`.

The spline `g` is parameterised as a positive combination of
softplus basis functions plus a positive linear term:

```
g(t) = softplus(slope) · t  +  Σ_k softplus(w_k) · softplus(t - knot_k)
```

This is monotone by construction (all coefficients are positive).
`knots` controls the spline resolution.

## `Hybrid`

```python
from laker.kernel import Hybrid
k = Hybrid(embeddings, lam=1e-2, alpha=0.5, num=200, k=10, dtype=torch.float64)
```

Two-scale combination of Nyström and Neighbors:
`K = α · K_nystrom + (1-α) · K_neighbors`. `alpha=0` is pure
neighbors, `alpha=1` is pure Nyström. Default `alpha=0.5`.

## Numerical accuracy

All approximations match the exact kernel's `matvec` to:
- Nyström with `num=200, n=2000`: relative error ~`1e-7`
- RFF with `features=128, n=2000`: relative error ~`1e-5`
- Neighbors with `k=10, n=2000`: relative error ~`1e-3` (sparse)
- Grid with `grid_size=64, n=2000`: relative error ~`1e-3`
- Spectrum: relative error ~`1e-6` (it learns the spectrum)

Tested in `tests/test_kernel.py`.

## References

- Williams & Seeger (2001), *"Using the Nyström Method to Speed Up
  Kernel Machines"*. The original Nyström paper.
- Rahimi & Recht (2007), *"Random Features for Large-Scale Kernel
  Machines"*. Random Fourier features.
- Wilson & Nickisch (2015), *"Kernel Interpolation for Scalable
  Structured Gaussian Processes"*. The SKI / KISS-GP approach.
