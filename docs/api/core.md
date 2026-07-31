# `laker.core` — pipeline composition

`Core` is the stateless composition root that wires together
embeddings, kernel construction, preconditioner construction, the PCG
solve, and prediction.

`Laker` owns a `Core` instance and delegates to it. Users normally
don't construct `Core` directly, but advanced users can compose it
to build custom pipelines.

## Constructor

```python
from laker.core import Core

core = Core(
    embed_dim=10,         # output dim of the default Position encoder
    lam=1e-2,             # ridge weight
    gamma=0.1,            # CCCP shrinkage parameter
    num=None,             # CCCP probe count (auto if None)
    eps=1e-8, base=0.05,
    cccp_max=200, cccp_tol=1e-6,
    pcg_tol=1e-6, pcg_max=1000,
    chunk=None,
    encoder=None,         # custom encoder module; defaults to Position
    kernel="exact",       # exact | nystrom | fourier | neighbors | grid | spectrum | hybrid
    landmarks=None, features=None, neighbors=None, grid_size=None,
    distributed=False,
    blend=0.5,
    selection="greedy", pilot=1000, knots=5,
    prec_kind="cccp",      # cccp | adaptive
    embed_dtype=None,      # defaults to dtype
    device=None, dtype=None,
    verbose=True,
)
```

## Methods

### `Core.embed(x) -> (Tensor, nn.Module)`

Compute embeddings for `x` of shape `(n, d)`. Returns
`(embeddings, encoder)` where `embeddings` has shape `(n, embed_dim)`.

If `self.encoder` is set, it is used. Otherwise a fresh `Position`
encoder is created on `self.device` with `dtype=self.embed_dtype`.

The output embeddings are cast to `self.dtype` if `embed_dtype ≠ dtype`.

### `Core.kernel(embed, lam=None) -> Kernel`

Build a kernel operator for the given embeddings.

| Argument | Default | Meaning |
|----------|---------|---------|
| `embed` | required | Training embeddings `(n, embed_dim)` |
| `lam` | `self.lam` | Override the ridge weight |

The kernel type is selected by `self.kernel` (and `self.distributed`).
For `n > 5000` and `chunk=None`, an automatic chunk size is chosen
to bound peak memory.

### `Core.build_prec(op, n, gamma=None, num=None, seed=None, diag=None) -> CCCP | Adaptive`

Build a preconditioner for the operator `op`. Returns either a
`CCCP` (default) or `Adaptive` (when `prec_kind="adaptive"`). The
adaptive variant runs a quick power-iteration diagnostic to choose
between Jacobi, CCCP, and aggressive-CCCP.

| Argument | Default | Meaning |
|----------|---------|---------|
| `op` | required | Callable applying the operator to a vector |
| `n` | required | Operator dimension |
| `gamma` | `self.gamma` | CCCP shrinkage override |
| `num` | `self.num` | Probe count override |
| `seed` | None | Seed for the probe generator |
| `diag` | None | Diagonal of the operator (for adaptive selection) |

### `Core.solve(kernel_op, prec_op, rhs, x0=None) -> (Tensor, int)`

Solve `(K + λ I) α = rhs` using `laker.solve.PCG`. Returns
`(α, iterations)`. The lambda used is whatever was passed to
`Core.kernel`. Uses `Backend.autocast()` if enabled.

### `Core.predict(x, enc, embed, kernel_op, alpha, corrector=None) -> Tensor`

Predict at query locations `x`. The prediction is
`K(x_query, x_train) @ alpha` plus optionally the corrector's output
on `x`.

| Argument | Meaning |
|----------|---------|
| `x` | Query locations `(m, d)` |
| `enc` | The trained encoder module |
| `embed` | Training embeddings `(n, embed_dim)` |
| `kernel_op` | The kernel operator |
| `alpha` | Solution vector `(n,)` |
| `corrector` | Optional residual corrector |

For large `m` or `n`, the matvec is chunked to bound memory at
`O(chunk_size * n)`.

### `Core.variance(x, enc, embed, kernel_op, prec_op, alpha, lam) -> Tensor`

Predictive posterior variance `σ²(x) = k(x, x) - k(x, X)^T (K + λI)^{-1} k(x, X)`.

For the Fourier kernel, uses the closed-form Woodbury identity (no
PCG solve needed). For other kernels, runs `k` batched PCG solves
(one per query chunk).

### `Core.predict_train(x, enc, embed, kernel_op, alpha, corrector=None) -> Tensor`

Differentiable version of `predict` — no `torch.no_grad()` block, so
the computation graph is preserved. Used by `learn` and `calibrate`.

### `Core.predict_var_train(x, enc, embed, kernel_op, prec_op, alpha, lam) -> Tensor`

Differentiable variance proxy. For the Fourier kernel, uses the
closed-form expression. For other kernels, uses a soft-min
distance-to-manifold proxy:
`σ²(x) = λ + Σ_i softmax(-‖E_x − E_{x_i}‖²) · ‖E_x − E_{x_i}‖²`.

### `Core.condition(kernel_op, prec_op) -> float`

Estimate `κ(P^{-1} K)` via power iteration. Used by `Laker.condition()`
which is itself used in tests and `BaseBench`.

## Where it's used

`Laker` constructs a single `Core` in its `__init__` and calls its
methods. The `Stream`, `Search`, `Trainer`, and `Bilevel` helpers
all share the same `Core` via `Laker._core`. This means kernel
construction, preconditioner build, solve, and predict are all
centralised in one place.
