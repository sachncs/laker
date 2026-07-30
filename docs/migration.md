# Migration: `LAKERRegressor` → `Laker`

LAKER used to expose a class named `LAKERRegressor`. The current
release replaces that surface with `laker.Laker`, a single facade
that owns configuration and fitted state directly. This document
maps every legacy API to its current equivalent and lists the
legacy names that have been removed.

---

## Class import

| Before | After |
| --- | --- |
| `from laker import LAKERRegressor` | `from laker import Laker` |
| `from laker.models import LAKERRegressor` | `from laker import Laker` |

---

## Constructor parameters

| Legacy | Current |
| --- | --- |
| `lambda_reg` | `regularization` |
| `kernel_approx` | `kernel` |
| `embedding_module` | `encoder` |
| `num_probes` | `probes` |
| `num_landmarks` | `landmarks` |
| `num_features` | `features` |
| `k_neighbors` | `neighbors` |
| `grid_size` | `grid_size` (unchanged) |
| `twoscale_alpha` | `blend` |
| `spectral_knots` | `knots` |
| `landmark_method` | `selection` |
| `landmark_pilot_size` | `pilot` |
| `preconditioner_strategy` | `preconditioner` (`"cccp"` or `"adaptive"`) |
| `None` (exact kernel) | `kernel="exact"` |
| `"rff"` | `kernel="fourier"` |
| `"knn"` | `kernel="neighbors"` |
| `"ski"` | `kernel="grid"` |
| `"twoscale"` | `kernel="hybrid"` |
| `"spectral"` | `kernel="spectrum"` |
| `"nystrom"` | `kernel="nystrom"` |

The values `"rff"`, `"knn"`, `"ski"`, `"twoscale"`, `"spectral"`,
`"nystrom"` still work as alias inputs and are mapped to the new
names internally.

---

## Method names

| Legacy | Current |
| --- | --- |
| `predict(x)` | `predict(x)` (unchanged) |
| `predict_variance(x)` | `variance(x)` |
| `score(x, y)` (negative RMSE) | `score(x, y)` (R² — `1.0 = perfect`, `0.0 = mean`) |
| `score_r2(x, y)` (R²) | `score(x, y)` (merged) |
| `condition_number()` | `condition()` |
| `fit_with_search(x, y, val_fraction=..., lambda_reg_grid=..., ...)` | `search("grid", x, y, val_fraction=..., regularizations=..., ...)` |
| `fit_with_bo(x, y, n_calls=..., lambda_reg_bounds=..., ...)` | `search("bayes", x, y, n_calls=..., regularization_bounds=..., ...)` |
| `partial_fit(x_new, y_new, ...)` | `update(x_new, y_new, ...)` |
| `fit_path(x, y, lambda_reg_grid=..., ...)` | `path(x, y, regularizations=..., ...)` |
| `fit_continuation(x, y, lambda_max=..., lambda_min=..., n_stages=..., ...)` | `continuation(x, y, lambda_max=..., lambda_min=..., n_stages=..., ...)` (also exposed via `path` for the structured schedule) |
| `fit_learned_embeddings(x, y, ...)` | `learn(x, y, ...)` |
| `fit_residual_corrector(x, y, ...)` | `correct(x, y, ...)` |
| `fit_uncertainty_aware(x, y, ...)` | `calibrate(x, y, ...)` |
| `fit_bilevel(x_train, y_train, x_val, y_val, ...)` | `tune(x_train, y_train, x_val, y_val, ...)` |

---

## Fitted-state attributes

| Legacy | Current |
| --- | --- |
| `alpha` | `coef_` |
| `embeddings` | `embeddings_` |
| `embedding_model` | `encoder_` |
| `kernel_operator` | `kernel_` |
| `preconditioner` (fitted state) | `preconditioner_` |
| `residual_corrector` | `corrector_` |
| `x_train` | `inputs_` |
| `y_train` | `targets_` |
| `pcg_iterations_` | `iterations_` |
| `partial_fit_count` | `_updates` |
| `core`, `search`, `streaming`, `trainer`, `persistence` | removed |

Note the trailing underscore convention: fitted-state attributes use
`_` per the sklearn idiom; internal scratch attributes use `_` but
are not part of the public surface.

---

## Removed names

The following symbols are no longer importable from `laker` after
the migration:

- `LAKERRegressor`, `LAKERCore`
- `EmbeddingTrainer`, `HyperparameterSearch`, `StreamingUpdater`,
  `ModelPersistence`, `BilevelOptimizer`
- `RadioFieldGenerator`, `Visualizer`, `GPSurrogate`
- `Executor`, `ExampleExecutor`, `BenchmarkExecutor`
- `SolverBenchmark`, `BaselineBenchmark`, `BenchmarkResult`
- `PerformanceBenchmarkSuite`, `ReproducibleBenchmarkSuite`,
  `ApproximationBenchmarkSuite`, `BaselineComparison`
- `AttentionKernelOperator`, `NystromAttentionKernelOperator`,
  `RandomFeatureAttentionKernelOperator`,
  `SparseKNNAttentionKernelOperator`, `SKIAttentionKernelOperator`,
  `TwoScaleAttentionKernelOperator`, `SpectralAttentionKernelOperator`,
  `DistributedAttentionKernelOperator`, `MonotoneSpectrumShaper`
- `CCCPPreconditioner`, `AdaptivePreconditioner`,
  `JacobiPreconditioner`, `PreconditionedConjugateGradient`,
  `GradientDescent`, `ResidualCorrector`, `PositionEmbedding`
- `get_default_device`, `set_default_device`
- `generate_grid`, `generate_radio_field`
- `plot_convergence`, `plot_radio_map`

If you are still importing any of these, search-and-replace using
the table above.

---

## Migration walkthrough

**Before** (`LAKERRegressor`):

```python
import torch
from laker.models import LAKERRegressor

model = LAKERRegressor(
    embedding_dim=10,
    lambda_reg=1e-2,
    gamma=0.1,
    num_probes=80,
    kernel_approx="nystrom",
    num_landmarks=200,
)
model.fit(x_train, y_train)
preds = model.predict(x_test)
var = model.predict_variance(x_test)
```

**After** (`Laker`):

```python
import torch
from laker import Laker

model = Laker(
    embedding_dim=10,
    regularization=1e-2,
    gamma=0.1,
    probes=80,
    kernel="nystrom",
    landmarks=200,
)
model.fit(x_train, y_train)
preds = model.predict(x_test)
var = model.variance(x_test)
```

The save/load and CLI surfaces are unchanged; the persistence file
uses `format_version=2` (the previous serialisation was upgraded).

---

## Why we changed

The legacy surface had:

- Long compound class names (`NystromAttentionKernelOperator` and
  friends) that obscured the API.
- Parameter names that didn't match the paper (`lambda_reg`, `gamma`)
  and made it harder to find the canonical name in `pyproject.toml`.
- Method names with redundant prefixes (`fit_with_*`) that violated
  the single-class-per-file rule.
- A separate `LAKERRegressor` class plus a hidden `LAKERCore`
  plus three service classes (`EmbeddingTrainer`,
  `HyperparameterSearch`, `StreamingUpdater`, `ModelPersistence`),
  which fragmented behaviour across many objects.

The new `Laker` class owns configuration and fitted state directly,
delegates heavy work to the `laker.<module>.<Class>` namespace, and
keeps a single `fit` / `predict` / `variance` / `score` /
`save` / `load` entry point at the top level.

For a deeper look at the rename plan, see `git log --oneline --grep
"refactor:"`.
