# Naming

Single source of truth for names. Update this file before any code change.

## Public top-level

| Name | Module | Kind |
|---|---|---|
| `Laker` | `laker.model` | model |
| `Laker.load` | `laker.model` | classmethod |

## Classes (module-qualified)

| Module | Class |
|---|---|
| `laker.kernel` | `Exact`, `Nystrom`, `Fourier`, `Neighbors`, `Grid`, `Hybrid`, `Spectrum`, `Distribute` |
| `laker.preconditioner` | `CCCP`, `Adaptive`, `Jacobi` |
| `laker.solve` | `PCG`, `Descent` |
| `laker.embed` | `Position`, `Visual` |
| `laker.plot` | `Plot` |
| `laker.data` | `Data` |
| `laker.helpers` | `Helpers` |
| `laker.backend` | `Backend` |
| `laker.base` | `Base` |
| `laker.fit` | `Fit` |
| `laker.search` | `Search` |
| `laker.stream` | `Stream` |
| `laker.implicit` | `Implicit` |
| `laker.cli` | `CLI` |

## Fitted state on `Laker`

| Old | New |
|---|---|
| `alpha` | `coef_` |
| `embeddings` | `embeddings_` |
| `embedding_model` | `encoder_` |
| `kernel_operator` | `kernel_` |
| `preconditioner` (fitted state) | `preconditioner_` |
| `residual_corrector` | `corrector_` |
| `x_train` | `inputs_` |
| `y_train` | `targets_` |
| `pcg_iterations_` | `iterations_` |
| `path_` | `path_` (kept) |
| `partial_fit_count` | `_updates` |
| `core`, `search`, `streaming`, `trainer`, `persistence` | removed |
| `HYPERPARAMS` | removed |

## Method names on `Laker`

| Old | New |
|---|---|
| `fit` | `fit` |
| `predict` | `predict` |
| `predict_variance` | `variance` |
| `score` (negative RMSE) | `score` (R²) |
| `score_r2` | removed (merged into `score`) |
| `condition_number` | `condition` |
| `fit_with_search` / `fit_with_bo` | `search(method=...)` |
| `partial_fit` | `update` |
| `fit_path` / `fit_continuation` | `path(start=..., stop=..., stages=...)` |
| `fit_learned_embeddings` | `learn` |
| `fit_residual_corrector` | `correct` |
| `fit_uncertainty_aware` | `calibrate` |
| `fit_bilevel` | `tune` |

## Parameters

| Old | New |
|---|---|
| `lambda_reg` | `regularization` |
| `kernel_approx` | `kernel` |
| `None` exact | `"exact"` |
| `"rff"` | `"fourier"` |
| `"knn"` | `"neighbors"` |
| `"ski"` | `"grid"` |
| `"twoscale"` | `"hybrid"` |
| `"spectral"` | `"spectrum"` |
| `twoscale_alpha` | `blend` |
| `num_probes` | `probes` |
| `num_landmarks` | `landmarks` |
| `num_features` | `features` |
| `k_neighbors` | `neighbors` |
| `spectral_knots` | `knots` |
| `landmark_method` | `selection` |
| `landmark_pilot_size` | `pilot` |
| `embedding_module` | `encoder` |
| `preconditioner_strategy` | `preconditioner` |
| `lambda_reg_grid` | `regularizations` |
| `pcg_iters` result key | `iterations` |
| `final_rel_res` result key | `residuals` |

## Env vars (documented in `.env.example` and README)

| Var | Default | Purpose |
|---|---|---|
| `LAKER_DTYPE` | `float32` | default floating dtype |
| `LAKER_DEVICE` | `cpu` | default torch device |
| `LAKER_CHUNK_MEMORY_BUDGET` | `0` (auto) | bytes budget per chunk |
| `LAKER_DISABLE_CHUNK` | `0` | disable chunking |
| `LAKER_COMPILE_MODE` | `default` | torch.compile mode |
| `LAKER_AUTOCAST` | `0` | autocast at inference |
| `LAKER_TF32` | `1` | enable TF32 matmuls |
| `LAKER_NUM_THREADS` | `0` (auto) | torch thread count |
| `LAKER_SEED` | unset | global RNG seed |
| `LAKER_VERBOSE` | `1` | default verbosity |
| `LAKER_LOG_LEVEL` | `INFO` | logger level |

## Removed from public surface (after refactor)

`AdaptivePreconditioner`, `JacobiPreconditioner`,
`NystromAttentionKernelOperator`, `RandomFeatureAttentionKernelOperator`,
`SparseKNNAttentionKernelOperator`, `SKIAttentionKernelOperator`,
`TwoScaleAttentionKernelOperator`, `SpectralAttentionKernelOperator`,
`DistributedAttentionKernelOperator`, `AttentionKernelOperator`,
`MonotoneSpectrumShaper`, `LAKERRegressor`, `LAKERCore`,
`EmbeddingTrainer`, `HyperparameterSearch`, `StreamingUpdater`,
`ModelPersistence`, `BilevelOptimizer`, `RadioFieldGenerator`,
`Visualizer`, `GPSurrogate`, `Executor`, `ExampleExecutor`,
`BenchmarkExecutor`, `PerformanceBenchmarkSuite`,
`ReproducibleBenchmarkSuite`, `ApproximationBenchmarkSuite`,
`BaselineComparison`, `SolverBenchmark`, `BenchmarkResult`,
`get_default_device`, `set_default_device`, `generate_grid`,
`generate_radio_field`, `plot_convergence`, `plot_radio_map`,
`baseline_laker_vs_baselines`, `baseline_solver`.

## Decisions

- `forgetting_factor`: replaced with `warm_start_scale` (honest label).
- RFF: docstring records the operator as a stationary Gaussian kernel,
  not the exponential dot-product kernel. Single source for truth.
- Multi-transmitter `Data.field`: linear-power summation in dBm.
