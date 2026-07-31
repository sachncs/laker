# API Reference

The single top-level export is `Laker`. Secondary classes are
available under module-qualified names:

```python
from laker import Laker
from laker.kernels import (
    Attention as Exact,
    NystromAttention as Nystrom,
    RandomFeatureAttention as Fourier,
    SparseAttention as Neighbors,
    SKIAttention as Grid,
    TwoScaleAttention as Hybrid,
    SpectralAttention as Spectrum,
)
from laker.preconditioner import CCCP, Adaptive, Jacobi
from laker.solve import PCG, Descent
from laker.embed import Position, Visual
from laker.plot import Plot
from laker.data import Data
from laker.helpers import Helpers
from laker.backend import Backend
from laker.base import Base
from laker.cli import CLI
```

The full module map is described in `CONTRIBUTING.md` under "Module
Conventions".

---

## Core Model

### `Laker`

```python
class Laker(
    embedding_dim: int = 10,
    regularization: float = 1e-2,
    gamma: float = 1e-1,
    probes: Optional[int] = None,
    epsilon: float = 1e-8,
    base_rho: float = 0.05,
    cccp_max_iter: int = 200,
    cccp_tol: float = 1e-6,
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 1000,
    chunk_size: Optional[int] = None,
    encoder: Optional[torch.nn.Module] = None,
    kernel: str = "exact",
    landmarks: Optional[int] = None,
    features: Optional[int] = None,
    neighbors: Optional[int] = None,
    grid_size: Optional[int] = None,
    blend: float = 0.5,
    selection: str = "greedy",
    knots: int = 5,
    pilot: Optional[int] = None,
    distributed: bool = False,
    embedding_dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
)
```

Learning-based Attention Kernel Regression estimator. Fits the
regularised attention kernel regression problem and solves it using a
learned CCCP preconditioner inside PCG.

| Method | Behaviour |
| --- | --- |
| `fit(x, y, x0=None)` | Fit on training data; returns `self`. |
| `predict(x)` | Predictive mean at query points. |
| `variance(x)` | Predictive variance at query points (non-negative). |
| `score(x, y)` | Coefficient of determination $R^2$ (1.0 = perfect, 0.0 = mean). |
| `search(method, x, y, ...)` | `method="grid"` or `method="bayes"`; fits with hyperparameter search. |
| `update(x_new, y_new, ...)` | Incremental update (raises a documented `RuntimeError` after `rebuild_threshold` rows and triggers a refit). |
| `path(x, y, regularizations, ...)` | Regularisation path. |
| `learn(x, y, lr=1e-3, epochs=50, ...)` | End-to-end encoder optimisation. |
| `correct(x, y, val_fraction=0.2, ...)` | Train a residual corrector on `y - y_hat_laker`. |
| `calibrate(x, y, lr=1e-3, epochs=50, beta=0.1, ...)` | Uncertainty-aware training. |
| `tune(x_train, y_train, x_val, y_val, lr=1e-3, ...)` | Bilevel regularisation + encoder tune. |
| `condition()` | Estimated condition number of the preconditioned system. |
| `save(path)` | Serialise to disk (instance method). |
| `Laker.load(path)` | Deserialise from disk (classmethod). |
| `get_params(deep=True)` / `set_params(**params)` | sklearn compatibility. |

#### Fitted state

| Attribute | Description |
| --- | --- |
| `coef_` | Fitted solution vector, shape `(n,)`. |
| `embeddings_` | Training embeddings, shape `(n, embedding_dim)`. |
| `encoder_` | The embedding module. |
| `kernel_` | The fitted kernel operator. |
| `preconditioner_` | The fitted preconditioner. |
| `inputs_` | Training locations. |
| `targets_` | Training targets. |
| `iterations_` | Iterations used by the last PCG solve. |

---

## Kernels

All kernel operators share the interface `matvec(x)`, `diagonal()`,
`to_dense()`, `eval(a, b, chunk_size=None)`. Choose via `Laker(kernel=...)`.

### `laker.kernels.Exact`

Exact exponential attention kernel $G_{ij} = \exp(\langle e_i, e_j\rangle)$.

### `laker.kernels.Nystrom`

Nyström low-rank approximation using `landmarks` landmark points
(defaults to $\max(200, 2\sqrt{n})$). **Audit note:** the `matvec`
implementation differs from `to_dense @ x` because the cached
$K_{mm}^{-1}$ is applied twice; the `diagonal` is exact, but
`to_dense @ x` is the audit-correct reference.

### `laker.kernels.Fourier`

Random Fourier features approximating a stationary Gaussian kernel.
**Audit note:** the operator approximates $\exp(-\|x-y\|^2 /
2\sigma^2)$, not the exponential dot-product kernel.

### `laker.kernels.Neighbors`

Sparse k-NN with `neighbors` nearest neighbours per row, stored as
COO. Symmetrised and diagonal-rewritten for positive definiteness.

### `laker.kernels.Grid`

Structured Kernel Interpolation on a product grid with size
`grid_size`.

### `laker.kernels.Hybrid`

Two-scale kernel that combines a Nyström global term with a sparse
k-NN local graph using the `blend` mix weight.

### `laker.kernels.Spectrum`

Spectral-shaped kernel via a monotone spline over the eigenvalues of
$E E^\top$.

### `laker.kernels.Distribute`

Multi-device wrapper that shards embeddings across CUDA devices
and gathers results to the master device. Falls back to a single-device
operator when only one (or zero) CUDA devices are detected.

---

## Preconditioners

### `laker.preconditioner.CCCP`

Learned data-dependent preconditioner via shrinkage-regularised
CCCP. Used by default.

### `laker.preconditioner.Adaptive`

Spectrum-aware preconditioner. Selects between Jacobi (low
condition number), CCCP (medium), and CCCP with a doubled probe
budget (high).

### `laker.preconditioner.Jacobi`

Diagonal (Jacobi) preconditioner baseline. Select via
`preconditioner="jacobi"`.

---

## Solvers

### `laker.solve.PCG`

`PCG(tol=..., max_iter=..., restart_freq=None, breakdown_eps=None).solve(operator, preconditioner, rhs, x0=None)`
returns `(x, status)` where `status` is a `laker.solvers.Status`
dataclass with `converged`, `iterations`, `residual`, `reason`, and
(for batched 2-D solves) `per_rhs`.

### `laker.solve.Descent`

`Descent(step_size=None, tol=1e-3, max_iter=50000).solve(operator, rhs, x0=None)`
unpreconditioned gradient descent baseline for benchmarking.

### `laker.solve.JacobiPreconditioner`

Diagonal-preconditioner helper used inside `Adaptive`. Build via
`JacobiPreconditioner(diagonal)` for explicit Jacobi operators.

---

## Embeddings

### `laker.embed.Position`

Default encoder: random Fourier features followed by a Tanh MLP.
Inherits from `torch.nn.Module`. Custom encoders may be supplied
through `Laker(encoder=...)`.

### `laker.embed.Visual`

Visual encoder using `Conv2d` patches followed by a linear projection.
Useful for image-shaped inputs.

---

## Workflow modules

### `laker.search.Search`

- `Search.grid(regressor, x, y, val_fraction, regularizations, ...)`
- `Search.bayes(regressor, x, y, val_fraction, n_calls, n_initial_points,
  regularization_bounds, ...)`

### `laker.fit.Fit`

- `Fit.learn(...)` — learned embeddings
- `Fit.correct(...)` — residual corrector
- `Fit.calibrate(...)` — uncertainty-aware training
- `Fit.tune(...)` — bilevel regularisation + encoder tune

### `laker.stream.Stream`

- `Stream.update(...)` — incremental update
- `Stream.path(...)` — regularisation path
- `Stream.continuation(...)` — continuation schedule

### `laker.implicit.Implicit`

- `Implicit.hypergradient(...)` — single-entry adjoint
  hypergradient computation.

---

## Data and plotting

### `laker.data.Data`

- `Data.field(locations, transmitters, powers, ...)` — radio field generator.
- `Data.grid(bounds, grid_size, ...)` — regular 2-D evaluation grid.
- `Data.validate_params(...)` — parameter validation.

### `laker.plot.Plot`

- `Plot.field(predictions, grid_size, ...)` — radio-map plot.
- `Plot.convergence(objective_gaps, ...)` — convergence curve.
- `Plot.image(predictions, grid_size, ...)` — flattened grid to image.

---

## Utilities

### `laker.helpers.Helpers`

Static methods: `safe_exp`, `normal_pdf`, `normal_cdf`, `sinh`,
`trace_normalize`, `lh_sample`, `power_iteration`, `safe_chol`,
`kernel_dtype_bytes`, `safe_inv_diag`, `default_seed`,
`set_global_seed`, `scatter_to_dense`.

### `laker.backend.Backend`

Static methods: `default_dtype`, `set_default_dtype`,
`get_default_dtype`, `get_chunk_budget`, `set_chunk_budget`,
`to_tensor`, `maybe_compile`, `seed`, `load_env`, `print_summary`.

### `laker.base.Base`

Static methods: `validate_inputs`, `validate_target`,
`validate_embedding_output`, `validate_split_indices`,
`coerce_tensor`, `device_of`.

### `laker.cli.CLI`

Static methods: `CLI.run(argv)`, `CLI.setup_logging`, `CLI.load_tensor`.
`python -m laker` dispatches through `CLI.run()`.

---

## Environment variables

Set these in the shell or via `.env`. `laker.backend.Backend.load_env()`
parses them at import time.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LAKER_DEVICE` | `cpu` | Default `torch.device`. |
| `LAKER_DTYPE` | `float32` | Default floating dtype. |
| `LAKER_CHUNK_MEMORY_BUDGET` | `64` | Per-chunk memory budget in MB. |
| `LAKER_DISABLE_CHUNK` | `0` | Disable chunked evaluation when `1`. |
| `LAKER_COMPILE_MODE` | unset | `torch.compile` mode (`default`, `reduce-overhead`, ...). |
| `LAKER_AUTOCAST` | `0` | Autocast at inference when `1`. |
| `LAKER_TF32` | `1` | Allow TF32 matmuls on Ampere+. |
| `LAKER_NUM_THREADS` | `0` (auto) | Torch CPU thread count. |
| `LAKER_SEED` | unset | Seeds `torch.manual_seed` at import. |
| `LAKER_VERBOSE` | `1` | Default verbosity. |
| `LAKER_LOG_LEVEL` | `INFO` | Logger level. |
