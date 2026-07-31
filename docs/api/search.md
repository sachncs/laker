# `laker.search` — hyperparameter search

`Search` provides grid and Bayesian hyperparameter search over the
key LAKER parameters (`lam`, `gamma`, `num`). Used internally by
`Laker.search` and `Laker.bayes`.

## Class

### `Search`

```python
from laker import Laker
m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
m._search.grid(
    m, x, y,
    lam_grid=[1e-3, 1e-2, 1e-1],
    gamma_grid=[0.0, 0.1, 1.0],
    num_grid=[20, 50, 100],
    val=0.2,
    warm=True,
    seed=0,
)
m._search.bayes(
    m, x, y,
    val=0.2,
    n_calls=15,
    n_init=5,
    lam_bounds=(1e-4, 1.0),
    gamma_bounds=(0.0, 2.0),
    num_bounds=(20, 300),
    seed=0,
)
```

The public API on `Laker` (`m.search(...)`, `m.bayes(...)`) wraps
these and re-fits the model on the full data with the best
hyperparameters.

## Methods

### `Search.grid(model, x, y, lam_grid, gamma_grid, num_grid, val, warm, seed)`

Exhaustive grid search over the cartesian product of `lam_grid`,
`gamma_grid`, `num_grid`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | required | The `Laker` instance (mutated in place) |
| `x`, `y` | required | Full training data |
| `lam_grid` | required | List of `lam` candidates |
| `gamma_grid` | required | List of `gamma` candidates |
| `num_grid` | required | List of `num` (probe count) candidates |
| `val` | required | Validation fraction in `(0, 1)` |
| `warm` | required | Warm-start PCG with the previous best alpha |
| `seed` | required | Seed for the train/val split |

Each trial builds the kernel + preconditioner + PCG solve on the
training split, evaluates RMSE on the validation split, and tracks
the best. If `warm=True`, the next trial's PCG is warm-started from
the previous best alpha.

After the grid is exhausted, sets `model.lam`, `model.gamma`,
`model.num` to the best values. Raises `RuntimeError` if every trial
failed (diverged or raised).

### `Search.bayes(model, x, y, val, n_calls, n_init, lam_bounds, gamma_bounds, num_bounds, seed)`

Bayesian optimisation with a lightweight GP surrogate.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | required | The `Laker` instance |
| `x`, `y` | required | Full training data |
| `val` | required | Validation fraction |
| `n_calls` | 15 | Total BO evaluations (including `n_init`) |
| `n_init` | 5 | Latin-hypercube seed points before GP fit |
| `lam_bounds` | `(1e-4, 1.0)` | log-transformed search range |
| `gamma_bounds` | `(0.0, 2.0)` | log-transformed search range |
| `num_bounds` | `(20, 300)` | linear search range |
| `seed` | required | Seed for LHS init and the train/val split |

The GP surrogate is `laker.math.GP` (RBF kernel, log-transform on
`lam` and `gamma`). Acquisition is Expected Improvement with
`xi=0.01`.

## When to use which

- Use `grid` when the search space is small (≤ 27 configs) and you
  want every config evaluated.
- Use `bayes` when the search space is large or you want to minimise
  total fits.
- The wrapper-side `m.search()` and `m.bayes()` also refit the model
  on the full data with the best hyperparameters.

## Reproducibility

The `seed` parameter controls both the train/val split and (for
`bayes`) the Latin-hypercube initial design. With the same seed,
the same `x`, `y`, and search space produce the same result.

## Limitations

- The validation metric is hard-coded to RMSE. If you care about a
  different metric, write a custom search loop.
- `bayes` does not cache the GP across calls; each `bayes` call
  starts fresh.
- The LHS init and the GP fit use the global numpy RNG; for
  deterministic behaviour across processes, set
  `np.random.default_rng(seed)` before the call.
