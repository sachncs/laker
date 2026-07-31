# Streaming updates

After the initial fit, three methods let you add new measurements
without re-fitting from scratch:

- `Laker.update(x_new, y_new)` — append new samples and re-solve.
- `Laker.path(x, y, grid=[...])` — fit a sequence of `lam` values.
- `Laker.continuation(x, y, lo, hi, stages)` — auto-generated path.

## `update`

```python
m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
m.fit(x, y)  # initial fit on n samples
m.update(x_new, y_new)  # append m_new samples; now fitted on n + m_new
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `forget` | 1.0 | Scale the previous alpha before warm-starting the new solve |
| `threshold` | 100 | Max cumulative new points before forcing a refit |
| `seed` | None | Seed for the preconditioner probes |

The new samples are appended to `m._x_train` and `m._y_train`; the
embeddings matrix, kernel operator, and preconditioner are rebuilt; PCG
is warm-started from `(forget * old_alpha, zeros)`.

When the cumulative number of new points across all `update` calls
exceeds `threshold`, `update` raises `RuntimeError("update threshold
exceeded")` to signal that a full `fit()` should be performed. This
prevents the embedding matrix from growing without bound.

Validation:
- `update` rejects un-fitted models (`RuntimeError`).
- `update` rejects `x_new` that is not 2-D (`ValueError`).

## `path`

```python
path = m.path(x, y, [0.1, 0.01, 0.001], reuse=True)
print("lam values:", path["lam"])
print("iters per lam:", path["iters"])
print("residuals:", path["rel"])
```

Fits a regularisation path: solves the system for each `lam` in
`grid`, sorted from largest to smallest. Each solve is warm-started
from the previous one. If `reuse=True`, the preconditioner is built
once and reused for every `lam`.

Returns a dict with:
- `"lam"` — sorted list of lambda values
- `"coef"` — list of solution vectors
- `"iters"` — list of PCG iteration counts
- `"rel"` — list of final relative residuals

After `path`, the model's `m.coef_` is the solution for the smallest
`lam` in the grid. Use this to explore how the solution changes as a
function of `lam` without re-embedding or re-preconditioning.

## `continuation`

```python
m.continuation(x, y, lo=1e-3, hi=1.0, stages=5)
```

Automatically builds a geometric schedule from `hi` down to `lo` over
`stages` points, then delegates to `path`. Useful when you don't know
the right `lam` grid in advance.

After `continuation`, `m.lam == lo`.

## When to use which

- Use `update` for online / streaming data where new samples arrive
  incrementally.
- Use `path` when you want to inspect the solution as a function of
  `lam` (e.g. for cross-validation or to pick the right `lam`).
- Use `continuation` when you only have a rough idea of the right
  `lam` range.

All three share the underlying `Laker._stream` helper; the public
methods are thin wrappers.
