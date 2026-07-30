# Troubleshooting

When something goes wrong, the canonical diagnostic path is:

1. Reproduce with a minimal example.
2. Enable verbose logging (`Laker(verbose=True)` or
   `LAKER_VERBOSE=1`).
3. Inspect the fitted state (`coef_`, `embeddings_`, `kernel_`,
   `preconditioner_`).

## Common errors

### `ValueError: regularization must be positive`

You passed a non-positive regularisation. `Laker` enforces
`regularization > 0` for stability of the linear system.

### `ValueError: x must be 2-D, got shape ...`

`Laker.predict(x)` requires `x` of shape `(n, d)`. If you have a
single point, wrap it: `x.unsqueeze(0)`.

### `ValueError: locations and transmitters must have same spatial dimension`

`laker.data.Data.field` requires locations and transmitters to
share the same spatial dimension. If transmitters are 2-D and
locations are 3-D, pass `(loc[:, :2], tx)` instead.

### `RuntimeError: PCG did not converge in N iterations`

PCG hit `max_iter` without reaching `pcg_tol`. Either:
- Increase `pcg_max_iter` (default `1000`).
- Relax `pcg_tol` (default `1e-6`).
- Increase `probes` so the preconditioner is more accurate.
- Switch to `dtype=torch.float64` to escape single-precision
  round-off.

### `RuntimeError: partial_fit rebuild threshold exceeded`

The cumulative incremental-update count crossed `rebuild_threshold`
(default `100`). The behaviour is documented: the next `update`
call triggers a full refit. If you do not want the rebuild, pass
`rebuild_threshold=` to a much larger value, or call `fit()`
yourself on the concatenated data.

### `FileNotFoundError: [Errno 2] No such file or directory`

`Laker.load(path)` failed because the path is wrong or the file was
moved. The on-disk file contains a `format_version` field (currently
`2`); if it is missing or has a different value, you are looking at
an incompatible artifact.

### `KeyError: ...` on `Laker.load`

The state dict is missing a required field. You are loading a
file written by a different LAKER version or a non-LAKER artifact.

### Slow fit / slow predict

The matvec dominates. Strategies:
- Switch to `kernel="nystrom"` or `kernel="fourier"` for large $n$.
- Use CUDA (`device="cuda"`).
- Reduce the embedding dimension (`embedding_dim=8` instead of
  `12`).
- Reduce `chunk_size` to limit per-call memory.

See [Performance](performance.md) for the full tuning matrix.

---

## Numerical issues

### `nan` in `coef_` or `predict`

`nan` indicates a numerical blow-up. Likely causes:
- Operator's spectrum overflowed (your embeddings have large
  norm). Try `dtype=torch.float64`.
- The preconditioner is degenerate (your `gamma` is too small).
  Try `gamma=0.1` or larger.
- The `pcg_tol` is too tight for `float32`. Increase it.

### Predictions look like noise

If `model.score(x, y)` is negative, the model is worse than
predict-the-mean. Likely causes:
- The embedding dim is too small to capture the target.
- The regularisation is too high (over-regularised).
- The embedding MLP has not been trained (try `model.learn(...)`).

If `model.score(x, y)` is exactly 1.0 over multiple different
splits, you are probably memorising the training set with
`regularization=0` and a tiny embedding — increase the data or
the noise model.

### `predictions.std() < 0.1`

The model has produced near-constant output. Possible:
- The kernel collapse (a Nyström operator with `landmarks=0` would
  give this; you cannot pass `landmarks=0`).
- The training data is too small to fit anything meaningful.
- The regularisation is too high.

### Variance is constant everywhere

`Laker.variance(x)` should be close to zero at training points
and rise away from them. If it is constant everywhere the
preconditioner is degenerate; rebuild with `Laker(gamma=0.1,
probes=200, dtype=torch.float64)`.

---

## Distributed / CUDA

### `RuntimeError: CUDA out of memory`

Either reduce `chunk_size` (smaller per-call allocation) or
disable the dense kernel in favour of `kernel="nystrom"`.

### Single-device fallback

`laker.kernel.Distribute` falls back to single-device execution
when fewer than two CUDA devices are available. To verify the
multi-device path is taking effect, set
`LAKER_VERBOSE=1` and look for the `Auto-selected chunk_size`
log line.

---

## Diagnostics commands

Quick scripts that catch the common failure modes:

```python
# 1. Check the kernel matrix is well-conditioned.
import torch
from laker.kernel import Exact

e = torch.randn(50, 6, dtype=torch.float64)
op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
cond = torch.linalg.cond(op.to_dense()).item()
print(f"condition number = {cond:.2e}")

# 2. Check the preconditioner is non-degenerate.
from laker.preconditioner import CCCP
prec = CCCP(probes=80, gamma=0.1, max_iter=200, verbose=False)
prec.build(op.matvec, 50, diagonal=op.diagonal())
print(f"isotropic coef = {prec.isotropic_coef}")

# 3. Check variance is well-behaved.
from laker import Laker
m = Laker(embedding_dim=8, regularization=1e-2)
m.fit(torch.rand(40, 2), torch.randn(40))
print(f"variance range = [{m.variance(torch.rand(5, 2)).min():.4f}, "
      f"{m.variance(torch.rand(5, 2)).max():.4f}]")
```

---

## Reporting a bug

If the steps above don't resolve the issue, open an issue with:

- A minimal reproducible snippet.
- The full stack trace and `LAKER_VERBOSE=1` log output.
- The `torch`, `numpy`, and `python` versions.
- The on-disk file size and `format_version` (if the issue is about
  persistence).
