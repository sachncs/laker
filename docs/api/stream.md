# `laker.stream` — streaming updates, path fitting, continuation

`Stream` is the underlying helper used by `Laker.update`, `Laker.path`,
and `Laker.continuation`. The public methods on `Laker` are thin
wrappers.

## Class

### `Stream`

```python
from laker import Laker
m = Laker(embed_dim=4, dtype=torch.float64)
m._stream.update(m, x_new, y_new, forget=1.0, threshold=100, seed=0)
m._stream.path(m, x, y, [0.1, 0.01, 0.001], reuse=True)
m._stream.continuation(m, x, y, lo=1e-3, hi=1.0, stages=5)
```

The constructor takes a `laker.core.Core` instance:

```python
Stream(core)
```

## Methods

### `Stream.update(model, x_new, y_new, forget, threshold, seed)`

Append new samples and re-solve with a warm start.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | required | The `Laker` instance |
| `x_new` | required | New inputs `(m, d)` |
| `y_new` | required | New targets `(m,)` |
| `forget` | 1.0 | Scale the previous alpha before warm-starting the new solve |
| `threshold` | 100 | Max cumulative new points before forcing a refit |
| `seed` | None | Seed for the preconditioner probes |

Validates: `x_new` is 2-D, `y_new` is 1-D, model is fitted.

When the cumulative new points exceed `threshold`, raises
`RuntimeError("update threshold exceeded")` to signal that a full
`fit` is required.

### `Stream.path(model, x, y, grid, reuse)`

Fit a regularisation path over a sequence of `lam` values.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | required | The `Laker` instance |
| `x`, `y` | required | Full training data |
| `grid` | required | List of `lam` values to fit |
| `reuse` | True | Reuse the preconditioner across all `lam` values |

Returns a dict with keys `"lam"`, `"coef"`, `"iters"`, `"rel"` (each
a list of length `len(grid)`).

After `path`, the model's `coef_` is the solution for the smallest
`lam` in the grid.

### `Stream.continuation(model, x, y, lo, hi, stages, reuse)`

Auto-generated geometric path from `hi` down to `lo` over `stages`
points. Delegates to `path`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lo` | None | Smallest `lam` (defaults to `model.lam`) |
| `hi` | None | Largest `lam` (defaults to `10 * model.lam`) |
| `stages` | 5 | Number of geometric points |

After `continuation`, `model.lam == lo`.

## When to use the public methods

The public API on `Laker` (`update`, `path`, `continuation`) is
just a thin wrapper around the same calls. Use the public methods
unless you have a custom model wrapper.

## See also

- [Streaming updates](../guides/streaming.md) — practical examples
- [algorithms/low_rank.md](../algorithms/low_rank.md) — kernel choices
  for large-scale fits
