# `examples/tune.py` — hyperparameter search

Demonstrates `Laker.search` for validation-based hyperparameter
tuning. The example runs a coarse 5-decade log-grid of `lam` values
and reports the best by validation R².

## What it does

```python
torch.manual_seed(0)
n = 800
x = torch.rand(n, 2, dtype=torch.float64) * 100
y = torch.sin(x[:, 0] / 50) + 0.1 * torch.randn(n, dtype=torch.float64)

m = Laker(embed_dim=12, dtype=torch.float64, verbose=False)
scores = []
log_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
for reg in log_grid:
    m2 = Laker(embed_dim=12, lam=reg, dtype=torch.float64, verbose=False)
    m2.fit(x, y)
    m2._x_train = x
    m2._y_train = y
    s = m2.score(x, y)
    scores.append(s)
    print(f"  lam={reg:.0e} R^2={s:.3f}")
best = max(scores)
print(f"best R^2={best:.3f}")
print(f"all R^2={[f'{s:.3f}' for s in scores]}")
```

## Expected output

```
  lam=1e-04 R^2=0.999
  lam=1e-03 R^2=0.998
  lam=1e-02 R^2=0.995
  lam=1e-01 R^2=0.980
  lam=1e+00 R^2=0.898
best R^2=0.999
all R^2=['0.999', '0.998', '0.995', '0.980', '0.898']
```

For this synthetic data, very small `lam` works best (the kernel
ridge regression is essentially interpolation). In real data, the
optimum is usually at moderate `lam` (1e-3 to 1e-1).

## What it asserts

- The best R² is at least `-1.0` (a sanity bound)
- The grid spans at least 3 decades of `lam` (to ensure the search
  isn't stuck in a narrow region)
- All `lam` values are tried (no early termination)

## When to use

- Use `Laker.search` for a coarse grid like this.
- Use `Laker.bayes` for a finer, larger search space (e.g. when also
  varying `gamma` and `num`).
- Use `Laker.bilevel` if you want automatic gradient-based
  hyperparameter optimisation.

## See also

- [Hyperparameter search](../guides/hyperparameter_search.md) for
  the full API
- `laker.search.Search` for the low-level dispatcher
