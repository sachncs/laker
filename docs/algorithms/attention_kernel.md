# The attention kernel

LAKER solves the regularised attention-kernel regression problem

```
min_α  ‖G α − y‖²  +  λ α^T G α
```

where `G = exp(E E^T)` is the exponential attention kernel induced by
learned spatial embeddings `E ∈ R^{n × d}` (one row per training
point). The dominant cost is solving the linear system

```
(G + λ I) α = y .
```

This document covers why this kernel is interesting, why solving the
system is hard, and the rest of the algorithm documentation
(`algorithms/cccp.md`, `algorithms/pcg.md`) covers how LAKER makes it
fast.

## Why an attention kernel?

Classical kernel ridge regression uses a stationary kernel
(e.g. RBF / Matérn) where `G_{ij} = k(x_i, x_j)`. The kernel matrix is
typically full rank, well-conditioned, and the implicit feature map
is finite-dimensional (Mercer's theorem) but not directly
parameterised.

Attention kernels replace the fixed metric with a learned similarity
based on the dot product of *embeddings*:

```
G_{ij} = exp( E_i · E_j )   (symmetric, PSD, but rank-deficient)
```

This is the kernel of the softmax / dot-product attention mechanism
used in transformers. It has two useful properties:

1. **Adaptive metric.** The kernel adapts to the data: if a learned
   encoder maps nearby points to similar embeddings, the attention
   weights concentrate on neighbours.

2. **No bandwidth tuning.** Unlike RBF, there is no length scale to
   pick — the encoder learns its own metric.

The price is that `G = exp(E E^T)` is exponential, not Gaussian. As
shown in the paper, this means `G` has a wide eigenvalue spread,
making the linear system ill-conditioned. Plain PCG on `(G + λ I) α = y`
takes hundreds to thousands of iterations.

## The spectral imbalance

Why is `G = exp(E E^T)` ill-conditioned? Write the SVD
`E = U Σ V^T`. Then

```
G = exp(E E^T) = U diag(exp(σ_i²)) U^T ,
```

so the eigenvalues of `G` are `exp(σ_i²)` where `σ_i` are the singular
values of `E`. The top eigenvalue is `exp(σ_1²)` and the bottom is
`exp(σ_d²)`. If `σ_1 / σ_d = 10` (mild imbalance in `E`), then the
condition number of `G` is `exp(100) ≈ 10^43`. The system is essentially
unsolvable without preconditioning.

LAKER's contribution is the CCCP preconditioner
([algorithms/cccp.md](cccp.md)) which learns an approximation
`Σ ≈ c I + Q B Q^T` such that `c I + Q B Q^T` has a much more
uniform spectrum when combined with `G`.

## Other kernel choices

The base `Exact` kernel implements `G = exp(E E^T)`. The other
six kernels ([guides/choosing_kernel.md](../guides/choosing_kernel.md))
approximate the same exponential kernel via:

- **Nyström** — `G ≈ K_nm K_mm^{-1} K_nm^T` with `m` landmark points
- **Fourier** — `G ≈ (1/r) Φ Φ^T` with `2r` random features
- **Neighbors** — top-`k` Euclidean k-NN graph, symmetrised, made PD
- **Grid** — exact kernel on a product grid, multilinear interpolation
- **Spectrum** — `K = U diag(exp(g(σ²))) U^T` with a learned monotone `g`
- **Hybrid** — `α · K_nystrom + (1-α) · K_neighbors`

All seven share the same `matvec` / `diag` / `dense` / `eval`
protocol, so the rest of LAKER (preconditioner, solver, training
methods, save/load) is kernel-agnostic.

## Implementation notes

The `Exact` class lives in `laker.kernel`. The exponential is
applied element-wise with dtype-aware overflow clamping:

```python
# from laker.kernel
out_safe = Math.exp(E @ E.T, out=gram)  # dtype-aware cap
```

The `lambda` regularisation is added on the diagonal at solve time, so
the kernel matrix itself never stores `G + λ I`.

## References

- Tao & Tan (2026), *"Accelerating Regularized Attention Kernel
  Regression for Spectrum Cartography"*.
  [arXiv:2604.25138](https://arxiv.org/abs/2604.25138).
- Vaswani et al. (2017), *"Attention Is All You Need"*. The original
  softmax-attention construction.
- Schölkopf & Smola (2002), *"Learning with Kernels"*. Standard
  reference on kernel ridge regression and the representer theorem.
