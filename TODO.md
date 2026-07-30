# LAKER Refactor Plan

Single-source atomic plan. Each step is one PR. Every step ends with the public
import `from laker import Laker` working and `pytest tests/ -x` green on the
scoped tests for that step.

Conventions:
- One primary class per module. Helpers exist only as `@staticmethod`.
- Public top-level re-exports live in `laker/__init__.py`. Nothing else.
- Underscore-prefixed modules are gone; everything is one word.
- Each step has its own acceptance gate before merging.

## Step 0 — Freeze naming, exports, and removal lists

Goal: stop drift between docs, code, and tests before any code change.

Files:
- `pyproject.toml` — `[tool.laker] public_api = ["Laker"]`.
- `laker/__init__.py` — re-export only `Laker`; placeholder for the rest.
- `docs/api.md` — list `Laker` and module-qualified secondary classes.
- `CONTRIBUTING.md` — rules: one class per file, helpers `@staticmethod`,
  no legacy aliases.
- `NAMING.md` — new, single source of truth. Holds:
  - rename table (current → target for classes, methods, attributes,
    parameters, env vars);
  - removal list (currently exported names that will not be re-exported
    after refactor).

Per-step gate:
- `grep -rn '<old_name>' laker/ tests/ docs/ examples/ benchmarks/ &&
  echo zero matches` for every entry in the rename table.
- `python -c "from laker import Laker"`.

## Step 0.5 — Migrate docs, examples, benchmarks to use new names

Goal: every README/docstring/example references names that will exist
post-refactor. Lands before any code moves.

Files:
- `README.md`, `laker/README.md`, `docs/*.md`, `examples/*.py`,
  `benchmarks/*.py`, `benchmarks/README.md`.
- Create `examples/radio_field.py`, `examples/bilevel.py`,
  `examples/distributed_matvec.py` referenced by README, OR delete references
  (decision logged in `NAMING.md`).
- Create `docs/release.md` referenced by README.

Gate:
- `grep -rn 'SearchService\|StreamingService\|DistributedMatvec\|BilevelLearner\|NystromAttentionKernelOperator\|RandomFeatureAttentionKernelOperator\|SparseKNNAttentionKernelOperator\|SKIAttentionKernelOperator\|TwoScaleAttentionKernelOperator\|SpectralAttentionKernelOperator\|DistributedAttentionKernelOperator\|AttentionKernelOperator\|HyperparameterSearch\|StreamingUpdater\|EmbeddingTrainer\|BilevelOptimizer\|ModelPersistence\|PreconditionedConjugateGradient\|RadioFieldGenerator\|Visualizer\|GPAdaptivePreconditioner' docs/ examples/ benchmarks/ && echo zero matches`.

## Step 1 — Operator invariants tests

Goal: capture the contracts that all kernel math must satisfy before math fixes.

Files:
- `tests/test_invariants.py`.

Tests (each parameterized over kernel name):
- `Kernel.matvec(v) == to_dense() @ v` for 1-D and 2-D RHS.
- `Kernel.diagonal() == diag(to_dense())`.
- `Kernel.eval(train, train) == to_dense() - lam * I`.
- `Kernel.matvec(v) == matvec_via_eval_then_dense(v)` for batched shapes.

Gate:
- `pytest tests/test_invariants.py -x`. Currently expected to fail; landing
  fail-loud with `@pytest.mark.xfail(reason="contract before fix")` tags so
  CI blocks math regressions during steps 13a–c.

## Step 2 — Solver status tests

Goal: PCG must return status.

Files:
- `tests/test_solve_status.py`.

Tests:
- PCG returns `(x, status)`. `status.converged` is bool; `status.iterations`
  is int; `status.residual` is float; `status.reason` is string.
- PCG on zero RHS converges immediately.
- PCG on rank-deficient PSD matrix hits breakdown with reason `breakdown`.
- PCG on NaN operator raises with reason `nonfinite`.
- Batched PCG returns per-RHS status, not one aggregate.
- `Descent` returns status with iteration/residual/reason.

Gate:
- `pytest tests/test_solve_status.py -x`. Expected to xfail until Step 12.

## Step 3 — State round-trip tests

Goal: every public mutation must survive save/load.

Files:
- `tests/test_state.py`.

Tests (parameterize over kernel name, dtype, device, adaptive vs cccp,
spectral vs nystrom, with/without corrector):
- `predict` on the loaded model equals `predict` on the original
  (within `torch.testing.assert_close` defaults for dtype).
- `variance` on the loaded model equals `variance` on the original.
- `condition` on the loaded model equals `condition` on the original
  within 1e-2 relative tolerance.
- `update` after load continues from the previous state without
  zero-target substitution.
- `Laker.load(Laker.save())` is identity.

Gate:
- `pytest tests/test_state.py -x`. Expected to xfail until Step 18.

## Step 4 — Data validation tests

Goal: input and split checks land before model refactor.

Files:
- `tests/test_data_validate.py`, `tests/test_split.py`.

Tests:
- `Laker().fit(x, y[:n-1])` raises with message including `len(x)`.
- `Laker().fit(x_with_nan, y)` raises with message including `finite`.
- `Laker().fit(x_int, y)` raises with message including `floating`.
- `Laker().fit(x, y_scalar)` does not collapse to a 0-D target (no
  unconditional `.squeeze()`); `y` is `(n,)` or `(n, 1)` accepted, rejected
  otherwise.
- `Laker().score(x, y)` requires `x.shape[0] == y.shape[0]`.
- `Laker().predict(x_with_wrong_features)` raises with `embedding_dim`.
- `Laker().fit(x, y).fit(x2, y2)` is atomic: a refit error does not leave
  half-fitted state from the prior call.
- Splits from `Search`, `Stream`, `Fit` are reproducible given a
  `random_state` argument.

Gate:
- `pytest tests/test_data_validate.py tests/test_split.py -x`. Expected
  partial pass; the unconditional `.squeeze()` tests will start passing
  after Step 21; atomic refit requires Step 21.

## Step 5 — `Helpers` class

Goal: gather math/RNG helpers behind one class; no module-level public
functions.

Files:
- `laker/helpers.py` — single class `Helpers`.
- Static methods: `normal_pdf`, `normal_cdf`, `sinh`, `trace_normalize`,
  `lh_sample`, `power_iteration`, `to_dense_kron`, `safe_chol`,
  `kernel_dtype_bytes`, `safe_inv_diag`, `default_seed`,
  `set_global_seed`, `scatter_to_dense`.
- `GPSurrogate` lives here temporarily for compatibility with Step 16;
  removed once `Search` is built (Step 16) and replaced by a private
  nested helper.
- `laker/utils.py` becomes a thin re-export of `Helpers` attrs, deleted
  at end of Step 16.

Gate:
- `pytest tests/test_invariants.py tests/test_solve_status.py tests/test_state.py tests/test_data_validate.py tests/test_split.py -x`.
- `python -c "from laker.helpers import Helpers; print(sorted(s for s in dir(Helpers) if not s.startswith('_')))"` — none of the names is a leftover long name.

## Step 6 — `Backend` class

Goal: all env/dtype/device/compile management behind one class.

Files:
- `laker/backend.py` — single class `Backend`.
- Static methods: `default_dtype`, `set_default_dtype`, `get_default_dtype`,
  `get_chunk_budget`, `set_chunk_budget`, `to_tensor`, `maybe_compile`,
  `seed`, `load_env`, `print_summary`.
- All previously free functions (`to_tensor`, `maybe_compile`,
  `get_default_dtype`, `set_default_dtype`, `get_chunk_memory_budget`)
  become `@staticmethod`.
- Validate `LAKER_*` env vars at module import; raise on negative or
  non-numeric values.

Gate:
- `pytest tests/test_data_validate.py -x`.
- `python -c "from laker.backend import Backend; Backend.load_env()"`
  with no env vars set exits cleanly.

## Step 7 — `Base` shared protocols

Goal: input validation and tensor coercion centralized.

Files:
- `laker/base.py` — single class `Base`.
- Static methods: `validate_inputs`, `validate_target`,
  `validate_embedding_output`, `validate_split_indices`,
  `coerce_tensor`, `device_of`.
- All currently duplicated validation literals migrate here.

Gate:
- `pytest tests/test_data_validate.py -x`.

## Step 8 — `Embed` class hierarchy

Goal: embedding code in one module, two subclasses.

Files:
- `laker/embed.py` — single class `Embed` with subclasses `Position`,
  `Visual`.
- Static method `Embed.coerce_output` enforces finite
  `(batch, embedding_dim)` output; moves module/dtype via `Backend`.
- Static methods `Embed.from_callable`, `Embed.from_features`,
  `Embed.eval_mode`, `Embed.zero_grad`.
- `laker/embeddings.py` becomes a thin re-export, deleted at end of
  Step 21.

Gate:
- `pytest tests/test_data_validate.py -x`.

## Step 9 — `Data` class

Goal: data generation behind one class.

Files:
- `laker/data.py` — single class `Data`.
- Static methods: `Data.field`, `Data.grid`, `Data.validate_params`.
- Inline validation: `reference_distance > 0`, path-loss exponent ≥ 0,
  shadow sigma ≥ 0, transmitters/powers finite.
- Decision recorded in `NAMING.md`: multi-transmitter `field` uses
  linear-power summation in dBm, not direct dBm addition.
- `laker/data.py` (today's file) is overwritten; remove the separate
  `RadioFieldGenerator` class.

Gate:
- New `tests/test_data.py` with one-transmitter analytical equality and
  two-transmitter linear-power test.
- `pytest tests/test_data.py -x`.

## Step 10 — `Plot` class

Goal: plotting behind one class.

Files:
- `laker/plot.py` — single class `Plot`.
- Static methods: `Plot.field`, `Plot.convergence`, `Plot.image`,
  `Plot.from_grid`.
- Remove unused `extent` parameter.
- Decide orientation: `Plot.image` enforces `x` horizontal, `y` vertical
  via documented transpose.
- `laker/visualize.py` becomes a thin re-export, deleted at end of
  Step 21.

Gate:
- New `tests/test_plot.py` covers `x → horizontal`, `y → vertical`.
- `pytest tests/test_plot.py -x`.

## Step 11 — `Preconditioner` class hierarchy

Goal: preconditioners in one module, real subclasses.

Files:
- `laker/preconditioner.py` — single class `Preconditioner` with
  subclasses `CCCP`, `Adaptive`, `Jacobi`.
- `Adaptive` carries the existing branch logic but `_compute_inner`
  becomes a private `@staticmethod`.
- `apply_1d`, `apply_2d` collapse into one `apply(x)` per subclass.
- `Jacobi.apply(x)` handles batched RHS by `inv_diag[:, None] * x`.

Gate:
- `pytest tests/test_solve_status.py -x`.
- New `tests/test_preconditioner.py`: Jacobi batched tests with
  `k = 1, n-1, n`; `Adaptive` branch coverage.

## Step 12 — `Solve` class hierarchy

Goal: solvers return status, fix numerical defects.

Files:
- `laker/solve.py` — single class `Solve` with subclasses `PCG`,
  `Descent`.
- `PCG.solve` returns `(x, status)`. Status dataclass: `converged`,
  `iterations`, `residual`, `reason`, `per_rhs: list[status] | None`.
- Breakdown rule: `abs(p @ Ap) <= eps * ||p|| * ||Ap||` ⇒ mark inactive.
- Batched path: per-RHS residuals and per-RHS status.
- Zero RHS short-circuits converged for that column; non-zero RHS columns
  continue.
- `Descent.solve` returns the same status shape.
- Replace `JacobiPreconditioner` calls inside `Solve` with the Step 11
  subclass.

Gate:
- `pytest tests/test_solve_status.py -x` (xfail markers removed).

## Step 13a — `Kernel` class hierarchy, structural only

Goal: rename + restructure without math change.

Files:
- `laker/kernel.py` — single class `Kernel` with subclasses `Exact`,
  `Nystrom`, `Fourier`, `Neighbors`, `Grid`, `Hybrid`, `Spectrum`,
  `Distribute`. Each kernel exposes `forward(x)` (or `matvec`),
  `diagonal`, `dense`, `eval(a, b)`. Module-level helpers collapse
  into `@staticmethod`s.
- `laker/kernels.py` and `laker/distributed_kernels.py` become thin
  re-exports; deleted at end of this step.

Decision (Step 13 outcome): single `kernel.py` (~1500 lines) is the target
file. If it grows past 2000 lines, the plan revisits with a subpackage
split in a follow-up.

Gate:
- `pytest tests/test_invariants.py tests/test_state.py -x` (xfail only).
- `mypy laker/` clean for `kernel.py`.

## Step 13b — Kernel math repair

Goal: math contracts from Step 1 pass.

Files:
- `laker/kernel.py` only.

Fixes:
- `exp_safe(..., out=...)` returns the exponentiated buffer
  (`torch.exp(out, out=out)`); every caller uses the return value.
- `Nystrom.forward(x)` uses cached `K_nm` and `K_mm_chol.T solve` exactly
  once; `Nystrom.eval` matches the fitted operator.
- `Neighbors.forward` and `Neighbors.eval` use one consistent
  neighborhood/topology policy.
- `Grid.diagonal` returns `w_i @ K_grid @ w_i` per row (full quadratic).
- `Hybrid` densifies the sparse component before combining.
- `Distribute.eval` uses the same `exp_safe` clamp as `Distribute.matvec`.
- `Fourier` either implements the exponential dot-product kernel
  explicitly, or its docstring documents it as a Gaussian kernel.
  Decision recorded in `NAMING.md`.
- Overflow handling: all operator methods go through `Helpers.safe_exp`.

Gate:
- `pytest tests/test_invariants.py -x` (xfail removed).
- `pytest tests/test_kernels.py -x` if it exists; otherwise create
  `tests/test_kernel_math.py` with dense matvec, eval, dense diagonal,
  parity against `Helpers.exact_kmm` for each class.

## Step 13c — Operator overflow consistency

Goal: matvec/diagonal/dense/eval share overflow policy.

Files:
- `laker/kernel.py`, `laker/helpers.py`.

Checks:
- High-norm embeddings: all four ops return finite tensors; PSD preserved
  via documented bound; `Solve.PCG` accepts the operator.
- `Fourier` and `Nystrom` respect the same clamping rule per dtype;
  `bfloat16` clamp uses 80.

Gate:
- `pytest tests/test_invariants.py -x`.
- New `tests/test_kernel_overflow.py` parameterized over dtype and
  operator name.

## Step 14 — `Implicit` class

Goal: implicit differentiation via one class.

Files:
- `laker/implicit.py` — single class `Implicit`.
- Static methods: `Implicit.hypergradient` (only public entry).
- `laker/implicit_diff.py` becomes a thin re-export; deleted at end of
  this step.

Gate:
- `pytest tests/test_state.py -x` (xfail only).

## Step 15 — `Stream` class

Goal: streaming/path/continuation/update behind one class.

Files:
- `laker/stream.py` — single class `Stream`.
- Static methods: `Stream.path`, `Stream.continuation`, `Stream.update`,
  `Stream.validate_threshold`, `Stream.validate_update_shape`.
- `forgetting_factor` becomes `Stream.warm_start_scale` (option B from
  audit) OR is implemented as real observation weights (decision in
  `NAMING.md`).
- `rebuild_threshold` is exclusive (`>`) and raises a documented
  `RuntimeError` only when truly exceeded; partial_fit no longer raises
  once it has been hit, it triggers a real rebuild.
- `laker/streaming.py` becomes a thin re-export; deleted at end of
  this step.

Gate:
- New `tests/test_stream.py` with deterministic threshold, retry,
  empty-update rejection, refit-after-load.
- `pytest tests/test_stream.py -x`.

## Step 16 — `Search` class

Goal: search behind one class with private surrogate.

Files:
- `laker/search.py` — single class `Search`.
- Static methods: `Search.grid`, `Search.bayes`, `Search.validate_grid`,
  `Search.validate_bayes`.
- `GPSurrogate` becomes a private nested class inside `Search`.
- Log-scaled dimensions: search transforms both values and bounds before
  normalizing; tested in `tests/test_search.py`.
- `n_initial_points <= n_calls` enforced; failed trials filtered before
  GP fitting; latin-hypercube drawn in one call.

Gate:
- `pytest tests/test_state.py -x` (xfail only) and new
  `tests/test_search.py`.

## Step 17a — `Fit` class, structural only

Goal: create `Fit` class without changing correctness.

Files:
- `laker/fit.py` — single class `Fit`.
- Static methods (delegating to current `training.py` functions):
  `Fit.embeddings`, `Fit.corrector`, `Fit.calibration`.
- `Fit.tune` (bilevel) is **not** added yet; Step 17c owns it.
- `laker/training.py` becomes a thin re-export; deleted once 17b/17c land.

Gate:
- `pytest tests/test_state.py tests/test_data_validate.py -x`.

## Step 17b — Correctness fixes for `Fit.embeddings`, `Fit.corrector`, `Fit.calibration`

Goal: each method actually does what it claims.

Files:
- `laker/fit.py`, `laker/model.py` (interface adjustments only).

Fixes:
- `Fit.embeddings`: snapshot/restore `Laker`'s encoder state dict along with
  detached embeddings; recompute embeddings after restoration; final
  kernel/preconditioner/PCG rebuild on restored state.
- `Fit.corrector`: split before fitting the base model; train corrector on
  out-of-fold predictions; corrector moves to model device/dtype before
  inference; `model.predict` adds corrector exactly once.
- `Fit.calibration`: clamp variance to a dtype-aware positive floor before
  NLL; use the same variance formula at train and inference time; restore
  best encoder state.

Gate:
- New `tests/test_fit.py`: embedding parity, corrector out-of-fold leakage
  test, calibration floor test.
- `pytest tests/test_fit.py -x`.

## Step 17c — `Fit.tune` bilevel correctness

Goal: bilevel actually updates parameters.

Files:
- `laker/fit.py`, `laker/implicit.py`.

Fixes:
- Build kernel/preconditioner using the transformable `lambda` parameter.
- Compute hypergradient through actual `lambda`-dependent operator and
  preconditioner.
- Restore the best outer-loop state (lambda + embeddings + encoder).
- Assert non-zero finite hypergradient on a synthetic problem.

Gate:
- `tests/test_fit.py::test_tune_changes_lambda` and
  `test_tune_improves_validation`.
- `pytest tests/test_fit.py -x`.

## Step 18 — `Laker.save` / `Laker.load`

Goal: serialization lives on the model, not a peer module.

Files:
- `laker/model.py` adds two classmethods: `Laker.save(self, path)`
  (instance method) and `Laker.load(path)` (classmethod).
- Persist: every hyperparameter, operator-specific random state
  (RFF frequencies/phases, Nyström landmarks), preprocessing stats
  (encoder state), training data (`x_train`, `y_train`), preconditioner
  state, format version header.
- `weights_only=True` is the default for `torch.load`; fallback path
  removed.
- Save CPU tensors; `map_location` is honored.
- Atomic write: temp file + `os.replace`.
- `Laker.persist.load(path)` removed; step 18 deletes `laker/persistence.py`.

Gate:
- `pytest tests/test_state.py -x` (xfail removed).
- Round-trip across all kernel names, dtypes, devices (skip CUDA in CI).

## Step 19 — `CLI` class

Goal: argparse handlers behind one class.

Files:
- `laker/cli.py` — single class `CLI`.
- Static methods: `CLI.run`, `CLI.setup_logging`, `CLI.load_tensor`,
  `CLI.dispatch_fit`, `CLI.dispatch_predict`.
- `laker/__main__.py` becomes a one-liner
  (`from laker.cli import CLI; raise SystemExit(CLI.run())`).
- Windows path: line endings, `Path` operations, signal handling tested
  explicitly in `tests/test_cli.py`.

Gate:
- `pytest tests/test_cli.py -x`.
- Manual smoke: `python -m laker --help` exits 0 on Linux and macOS.

## Step 20 — One-class-per-file audit

Goal: enforce the convention.

Files:
- Convention documented in `CONTRIBUTING.md` under "Module Conventions
  (post-refactor)".
- Code review enforces one class per module; no automated gate.

Gate:
- Each new module ≤ 12 `@staticmethod`s and ≤ 300 lines; otherwise split.

## Step 21 — `Laker` facade

Goal: model owns configuration and fitted state.

Files:
- `laker/model.py` defines `Laker`.
- Constructor signature defines the only canonical hyperparameter set.
- `__getattr__`/`__setattr__` proxies removed.
- `sklearn` protocol implemented cleanly:
  - `get_params` parses strings via `Backend.dtype_from_string`.
  - `set_params` validates via constructor signature, not `hasattr`.
  - `score` returns R².
  - `predict_variance` becomes `variance`.
  - `condition_number` becomes `condition`.
  - `pcg_iterations_` becomes `iterations_`.
  - Setter invalidates fitted state when a hyperparameter changes.
- Fitted-state names follow the audit (Section "state map"): `coef_`,
  `embeddings_`, `kernel_`, `preconditioner_`, `encoder_`, `inputs_`,
  `targets_`, `iterations_`, `corrector_`.
- Atomic refit: state assigned only after PCG succeeds.
- Custom embedding contract: `Embed.coerce_output` enforces finite
  `(batch, embedding_dim)`.

Gate:
- `pytest tests/test_state.py tests/test_data_validate.py tests/test_fit.py -x`.
- New `tests/test_laker_protocol.py`: `sklearn.utils.estimator_checks`
  for default and parametrized constructors.
- `mypy laker/model.py --strict` clean.

## Step 22 — Public surface cleanup

Goal: `laker/__init__.py` and submodules match the audit's target surface.

Files:
- `laker/__init__.py`: re-exports `Laker` only.
- Removed from exports (recorded in `NAMING.md`):
  `AdaptivePreconditioner`, `JacobiPreconditioner`, `CCCPPreconditioner`,
  `PreconditionedConjugateGradient`, `GradientDescent`,
  `MonotoneSpectrumShaper`, `AdaptivePreconditioner`,
  `LAKERRegressor`, `LAKERCore`, `EmbeddingTrainer`,
  `HyperparameterSearch`, `StreamingUpdater`, `ModelPersistence`,
  `BilevelOptimizer`, `RadioFieldGenerator`, `Visualizer`, `GPSurrogate`,
  `Executor`, `ExampleExecutor`, `BenchmarkExecutor`,
  `PerformanceBenchmarkSuite`, `ReproducibleBenchmarkSuite`,
  `ApproximationBenchmarkSuite`, `BaselineComparison`,
  `SolverBenchmark`, `BenchmarkResult`.
- `laker.kernel.Nystrom`, `laker.kernel.Fourier`, `laker.kernel.Neighbors`,
  `laker.kernel.Grid`, `laker.kernel.Hybrid`, `laker.kernel.Spectrum`,
  `laker.kernel.Distribute`, `laker.kernel.Exact`,
  `laker.preconditioner.CCCP`, `laker.preconditioner.Adaptive`,
  `laker.preconditioner.Jacobi`, `laker.solve.PCG`,
  `laker.solve.Descent`, `laker.embed.Position`, `laker.embed.Visual`,
  `laker.plot.Plot`, `laker.data.Data`, `laker.helpers.Helpers`,
  `laker.backend.Backend`, `laker.base.Base`,
  `laker.fit.Fit`, `laker.search.Search`, `laker.stream.Stream`,
  `laker.implicit.Implicit`, `laker.cli.CLI`.

Gate:
- `python -c "from laker import SearchService; print('ok')"` exits non-zero.
- `python -c "from laker import Laker; print(Laker)"` exits 0.

## Step 23 — Module rename pass

Goal: convert underscored module paths to flat paths; remove alias files.

Files renamed:
- `kernel/kernels.py` → `kernel.py` (already single file after 13a–c).
- `solve/solvers.py` → `solve.py`.
- `preconditioner/preconditioner.py` → `preconditioner.py`.
- `embed/embeddings.py` → `embed.py`.
- `data/data.py` → `data.py` (single file).
- `plot/visualize.py` → `plot.py`.
- `stream/streaming.py` → `stream.py`.
- `fit/training.py` → `fit.py`.
- `state/persistence.py` → `state.py` (deleted after Step 18).
- `implicit/implicit_diff.py` → `implicit.py`.
- Drop underscore prefixes: `_base.py → base.py`, `_backend.py → backend.py`,
  `_fit.py → fit.py`, `_search.py → search.py`, `_state.py → state.py`,
  `_implicit.py → implicit.py`.

Per-rename gate:
- `grep -rn '<old_path>' laker/ tests/ docs/ examples/ benchmarks/` →
  zero matches.
- `pytest tests/ -x`.

## Step 24 — Documentation pass

Goal: README, docs, examples align with `Laker` only.

Files:
- `README.md`: remove `SearchService`/`StreamingService`/etc. table.
  Replace with `Laker` table plus module-qualified secondary classes.
  Fix `score` reference to R² not negative RMSE.
- `docs/quickstart.md`: single import line + minimal example.
- `docs/theory.md`, `docs/patterns.md`, `docs/examples.md` updated.
- `examples/basic.py`, `examples/large.py`, `examples/executor.py`
  rewritten as one example each.
- Decision from Step 0.5 honored: missing example files added or
  references removed.
- `docs/release.md` created.

Gate:
- `grep -rn 'SearchService\|StreamingService\|DistributedMatvec\|BilevelLearner' .` → zero matches.
- `pytest tests/ -x`.

## Step 25 — Versioning single source

Goal: only one place defines version.

Files:
- `pyproject.toml` — `[project] dynamic = ["version"]`; provide
  `tool.setuptools.dynamic` reading from `laker/_version.py`.
- `laker/__init__.py` reads version from `importlib.metadata`.
- `docs/conf.py` reads version from `importlib.metadata`.
- `CHANGELOG.md` updated.

Gate:
- `python -c "import laker, importlib.metadata; assert laker.__version__ == importlib.metadata.version('laker')"`.

## Step 26 — CI matrix expansion

Goal: tested on Python 3.13 and Windows.

Files:
- `.github/workflows/ci.yml`: add `3.13`; add a single `windows-latest`
  job for one Python version; add `pytest --cov=laker --cov-branch
  --cov-fail-under=90 --cov-branch-fail-under=80` gate.

Gate:
- PR run is green for 3.9–3.13 on Linux + macOS, plus one Windows run.

## Step 27 — Dependency policy

Goal: bounded versions.

Files:
- `pyproject.toml`: tighten `torch` to tested range; declare
  `python_requires`; declare upper-bound wheels for `numpy`, `scipy`,
  `matplotlib`.

Gate:
- `python -m pip install -e .` succeeds with current versions.
- `python -m pip install --dry-run laker==<test_version>.<patch>` from
  a fresh venv succeeds.

## Step 28 — Env-var documentation

Goal: every `LAKER_*` variable is documented.

Files:
- `.env.example`: list `LAKER_LOG_LEVEL`, `LAKER_DEVICE`, `LAKER_DTYPE`,
  `LAKER_COMPILE_MODE`, `LAKER_CHUNK_MEMORY_BUDGET`, `LAKER_DISABLE_CHUNK`,
  `LAKER_AUTOCAST`, `LAKER_TF32`, `LAKER_NUM_THREADS`, `LAKER_SEED`,
  `LAKER_VERBOSE`.
- `README.md` has a table of env vars with default values.
- `Backend.load_env` validates each at import time.

Gate:
- New `tests/test_env.py`: each var parses without env, rejects negative
  integers and non-numeric strings.
- `pytest tests/test_env.py -x`.

## Step 29 — Release gate

Goal: package is shippable.

Acceptance criteria (numeric):
- `pytest tests/ -x` passes with `--cov=laker --cov-branch
  --cov-fail-under=90 --cov-branch-fail-under=80`.
- Test count between 110 and 140 (target 120).
- `mypy laker/ --strict` passes.
- `python -m build` produces sdist and wheel.
- `twine check dist/*` reports `PASSED`.
- `python -m pip install --force-reinstall dist/*.whl` in a fresh venv
  containing only `torch` and `numpy`:
  - `python -c "from laker import Laker; import torch; m = Laker(); x = torch.rand(20, 2); y = torch.randn(20); m.fit(x, y); p = m.predict(x[:5]); print(p.shape)"` exits 0 and prints `torch.Size([5])`.
- `python -m laker --help` exits 0 on Linux, macOS, and Windows.
- `python -m laker fit --help` and `python -m laker predict --help`
  exit 0.
- Coverage report absent from VCS (`coverage.xml` untracked).

Files:
- `.gitignore` covers `coverage.xml`, `*.egg-info`, `dist/`, `build/`.
- Optional publish step (`twine upload`) gated behind explicit user
  approval; plan does not execute publishing without confirmation.

Gate:
- A `tools/release_check.py` runs all of the above and exits 0 on
  Linux + macOS + Windows (CI matrix run).

## Cross-step invariants

These hold for every step:

1. **One class per file.** Enforced by code review (see CONTRIBUTING.md).
2. **No module-level public functions.** Only class attributes and
   `if __name__ == "__main__":` blocks may define top-level names.
3. **No legacy aliases.** Old names disappear with the same step that
   removes them; no deprecation period.
4. **Atomics are reviewable.** No step modifies more than three modules
   outside its target module list above.
5. **Verification gate.** Each step's "Gate" section runs `pytest tests/
   -x` over its scoped tests plus the public-import smoke
   (`python -c "from laker import Laker"`).

## Phase summary

| Phase | Steps | Goal |
|---|---|---|
| 0 | 0, 0.5 | Freeze naming, migrate docs/examples |
| 1 | 1, 2, 3, 4 | Fail-loud invariant tests |
| 2 | 5–20 | Class-per-file structural refactor with math correctness fixes |
| 3 | 21, 22, 23 | `Laker` facade + public surface + module rename pass |
| 4 | 24–28 | Documentation, versioning, CI, deps, env vars |
| 5 | 29 | Release gate |
