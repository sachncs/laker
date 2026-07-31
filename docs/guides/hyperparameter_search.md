# Hyperparameter search

LAKER exposes two validation-based search methods:

- `Laker.search(...)` — exhaustive grid search over
  `(lam, gamma, num)`.
- `Laker.bayes(...)` — Bayesian optimisation using a Gaussian-process
  surrogate and Expected Improvement acquisition.

Both split the data into train/validation, evaluate each candidate on
the validation set, and refit the model with the best configuration on
the full data.

## Grid search

```python
from laker import Laker
m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
m.search(
    x, y,
    lam_grid=[1e-3, 1e-2, 1e-1],
    gamma_grid=[0.0, 0.1, 1.0],
    num_grid=[20, 50, 100],
    val=0.2,
    warm=True,
    seed=0,
)
print("best lam:", m.lam)
print("best gamma:", m.gamma)
print("best num:", m.num)
```

Parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lam_grid` | `[1e-3, 1e-2, 1e-1]` | Ridge-weight candidates |
| `gamma_grid` | `[0.0, 0.1, 1.0]` | CCCP-shrinkage candidates |
| `num_grid` | `[50, 100, 200]` | CCCP-probe-count candidates |
| `val` | 0.2 | Validation fraction in `(0, 1)` |
| `warm` | True | Warm-start PCG with the previous best alpha |
| `seed` | None | Seed for the deterministic train/val split |

After `search`, the model is fully refit on all of `x, y` with the
best hyperparameters and the original `embed_dim` is preserved.

The split is deterministic given `seed` (uses a `torch.Generator`
seeded at `int(seed)`). Without a `seed`, the split uses the model's
current torch global RNG state.

## Bayesian search

```python
m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
m.bayes(
    x, y,
    val=0.2,
    n_calls=15,
    n_init=5,
    lam_bounds=(1e-4, 1.0),
    gamma_bounds=(0.0, 2.0),
    num_bounds=(20, 300),
    seed=0,
)
```

Parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `val` | 0.2 | Validation fraction |
| `n_calls` | 15 | Total BO evaluations (including `n_init`) |
| `n_init` | 5 | Latin-hypercube seed points before fitting the GP |
| `lam_bounds` | (1e-4, 1.0) | log-transformed search range |
| `gamma_bounds` | (0.0, 2.0) | log-transformed search range |
| `num_bounds` | (20, 300) | linear search range |
| `seed` | None | Seed for LHS init and split |

The GP is a lightweight RBF kernel surrogate implemented in
`laker.math.GP`. It does not depend on scikit-learn or GPyTorch.

The internal `_search.grid` and `_search.bayes` methods are also
exposed for programmatic use without the wrapper-side refit:

```python
m._search.grid(
    m, x, y,
    lam_grid=[1e-3, 1e-2],
    gamma_grid=[0.1, 0.5],
    num_grid=[50, 100],
    val=0.2,
    warm=True,
    seed=0,
)
```

## When to use which

- Use `search` when the search space is small (≤ 27 configs).
- Use `bayes` when the search space is large or you want to minimise
  total fits.
- The `Laker.search` dispatcher raises `RuntimeError("all parameter
  combinations diverged")` if every trial fails; in that case, widen
  `lam_grid`, increase `pcg_max`, or switch to `dtype=float64`.
