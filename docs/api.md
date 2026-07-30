# API Reference

Public surface of LAKER. The single top-level export is `Laker`;
secondary classes are available under module-qualified names.

```python
from laker import Laker
from laker.kernel import Nystrom, Fourier, Neighbors, Grid, Hybrid, Spectrum, Exact
from laker.preconditioner import CCCP, Adaptive, Jacobi
from laker.solve import PCG, Descent
from laker.embed import Position, Visual
from laker.plot import Plot
from laker.data import Data
```

Module map is defined in CONTRIBUTING.md ("Module Conventions").

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
    encoder: Optional[nn.Module] = None,
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

Learning-based Attention Kernel Regression estimator. Fits the regularised
attention kernel regression problem and solves it efficiently using a learned
preconditioner inside PCG.

**Key methods:**

- `fit(x, y, x0=None)` — Fit the model to sparse measurements.
- `predict(x)` — Reconstruct the radio field at query locations.
- `variance(x)` — Predictive variance at query locations.
- `score(x, y)` — Coefficient of determination (R²).
- `search(method, x, y, ...)` — Fit with grid or Bayesian search.
- `update(x_new, y_new, ...)` — Incremental update with new observations.
- `path(x, y, regularizations, ...)` — Regularisation path.
- `learn(x, y, lr=1e-3, epochs=50, ...)` — End-to-end embedding optimisation.
- `correct(x, y, val_fraction=0.2, epochs=200, ...)` — Train a residual corrector.
- `calibrate(x, y, lr=1e-3, epochs=50, beta=0.1, ...)` — Uncertainty-aware training.
- `tune(x_train, y_train, x_val, y_val, lr=1e-3, epochs=20, ...)` — Bilevel hyperparameter learning.
- `condition()` — Estimated condition number of the preconditioned system.
- `save(path)` — Serialise to disk (instance method).
- `Laker.load(path)` — Deserialise from disk (classmethod).
- `get_params(deep=True)` / `set_params(**params)` — sklearn compatibility.

---

## Kernels

All kernel operators share the interface `forward(x)`, `diagonal()`,
`dense()`, and `eval(a, b, chunk_size)`.

### `laker.kernel.Exact`

Exact exponential attention kernel.

### `laker.kernel.Nystrom`

Nyström low-rank approximation using `landmarks` landmark points.

### `laker.kernel.Fourier`

Random Fourier feature approximation (stationary Gaussian kernel).

### `laker.kernel.Neighbors`

Sparse k-NN approximation stored as a COO tensor.

### `laker.kernel.Grid`

Structured Kernel Interpolation on a product grid.

### `laker.kernel.Hybrid`

Two-scale kernel combining global Nyström + local sparse k-NN with
`blend` mixing weight.

### `laker.kernel.Spectrum`

Spectral-shaped attention kernel via fixed monotone spectrum shaper.

### `laker.kernel.Distribute`

Multi-GPU wrapper that shards embeddings across CUDA devices.

---

## Preconditioners

### `laker.preconditioner.CCCP`

Learned data-dependent preconditioner via shrinkage-regularised CCCP.

### `laker.preconditioner.Adaptive`

Spectrum-aware preconditioner with power-iteration-biased probes and
orthogonalised blocks. Selected via `preconditioner="adaptive"`.

### `laker.preconditioner.Jacobi`

Diagonal (Jacobi) preconditioner baseline.

---

## Solvers

### `laker.solve.PCG`

Preconditioned conjugate gradient solver. Returns `(x, status)` where
`status` carries `converged`, `iterations`, `residual`, `reason`.

### `laker.solve.Descent`

Unpreconditioned gradient descent baseline. Returns `(x, status)`.

---

## Embeddings

### `laker.embed.Position`

Deterministic position-driven embedding module (random Fourier features
+ MLP). Inherits from `torch.nn.Module`.

### `laker.embed.Visual`

Visual feature embedding with patch tokens.

---

## Workflow modules

### `laker.search.Search`

Static-method API for validation-based hyperparameter search:

- `Search.grid(...)` — Grid search.
- `Search.bayes(...)` — Bayesian Optimisation.

### `laker.fit.Fit`

Static-method API for advanced training modes:

- `Fit.learn(...)` — Learned embeddings.
- `Fit.correct(...)` — Residual corrector.
- `Fit.calibrate(...)` — Uncertainty-aware training.
- `Fit.tune(...)` — Bilevel hyperparameter learning.

### `laker.stream.Stream`

Static-method API for online updates and paths:

- `Stream.update(...)` — Incremental update.
- `Stream.path(...)` — Regularisation path.
- `Stream.continuation(...)` — Continuation schedule.

### `laker.implicit.Implicit`

Static-method API for adjoint-based hypergradient computation:

- `Implicit.hypergradient(...)` — Single entry point.

---

## Data and plotting

### `laker.data.Data`

Static-method API for synthetic data:

- `Data.field(...)` — Radio field generator.
- `Data.grid(...)` — Regular 2-D evaluation grid.
- `Data.validate_params(...)` — Parameter validation.

### `laker.plot.Plot`

Static-method API for visualisation:

- `Plot.field(...)` — Radio-map plot.
- `Plot.convergence(...)` — Convergence curve.
- `Plot.image(...)` — Flattened grid to image conversion.

---

## Utilities

### `laker.helpers.Helpers`

Numerical-stability and RNG helpers:

- `Helpers.normal_pdf`, `Helpers.normal_cdf`, `Helpers.sinh`,
  `Helpers.trace_normalize`, `Helpers.lh_sample`,
  `Helpers.power_iteration`, `Helpers.to_dense_kron`,
  `Helpers.safe_chol`, `Helpers.kernel_dtype_bytes`,
  `Helpers.safe_inv_diag`, `Helpers.default_seed`,
  `Helpers.set_global_seed`, `Helpers.scatter_to_dense`,
  `Helpers.safe_exp`.

### `laker.backend.Backend`

Environment and tensor utility namespace:

- `Backend.default_dtype`, `Backend.set_default_dtype`,
  `Backend.get_default_dtype`, `Backend.get_chunk_budget`,
  `Backend.set_chunk_budget`, `Backend.to_tensor`,
  `Backend.maybe_compile`, `Backend.seed`, `Backend.load_env`,
  `Backend.print_summary`.

### `laker.base.Base`

Validation and tensor coercion:

- `Base.validate_inputs`, `Base.validate_target`,
  `Base.validate_embedding_output`, `Base.validate_split_indices`,
  `Base.coerce_tensor`, `Base.device_of`.

### `laker.cli.CLI`

Argparse entry points:

- `CLI.run(argv)` — single main entry.

---

## Removed names (post-refactor)

The following symbols are no longer exported after the refactor:

- `LAKERRegressor`, `LAKERCore`, `EmbeddingTrainer`,
  `HyperparameterSearch`, `StreamingUpdater`, `ModelPersistence`,
  `BilevelOptimizer`, `RadioFieldGenerator`, `Visualizer`,
  `GPSurrogate`, `Executor`, `ExampleExecutor`, `BenchmarkExecutor`,
  `SolverBenchmark`, `BaselineBenchmark`, `BenchmarkResult`,
  `PerformanceBenchmarkSuite`, `ReproducibleBenchmarkSuite`,
  `ApproximationBenchmarkSuite`, `BaselineComparison`,
  `AttentionKernelOperator`, `NystromAttentionKernelOperator`,
  `RandomFeatureAttentionKernelOperator`,
  `SparseKNNAttentionKernelOperator`, `SKIAttentionKernelOperator`,
  `TwoScaleAttentionKernelOperator`,
  `SpectralAttentionKernelOperator`,
  `DistributedAttentionKernelOperator`,
  `MonotoneSpectrumShaper`, `CCCPPreconditioner`,
  `AdaptivePreconditioner`, `JacobiPreconditioner`,
  `PreconditionedConjugateGradient`, `GradientDescent`,
  `ResidualCorrector`, `PositionEmbedding`,
  `get_default_device`, `set_default_device`, `generate_grid`,
  `generate_radio_field`, `plot_convergence`, `plot_radio_map`,
  `baseline_solver`, `baseline_laker_vs_baselines`.
