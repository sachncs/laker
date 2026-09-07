# `examples/scale.py` — large-scale fit + regularisation path

Larger `n` (5000) and grid (30×30 = 900). Shows that LAKER handles
thousands of training points and dense evaluation grids in a few
seconds, and demonstrates the regularisation-path API.

## What it does

```python
torch.manual_seed(0)
n = 5000
tx = torch.tensor([[40.0, 40.0], [70.0, 60.0]])
powers = torch.tensor([-30.0, -35.0])
area = 100.0
grid_size = 30

# Train on n=5000 random sensors.
locs = torch.rand(n, 2, dtype=torch.float64) * area
_, targets = Data.field(locs, tx, powers, loss=2.7)
m = Laker(embed_dim=10, dtype=torch.float64, verbose=False)
m.fit(locs, targets)
print(f"fit seconds: {elapsed:.2f}")
print(f"embed shape: {m.embed_.shape}")

# Predict on a 30x30 grid.
grid = Data.grid((0, area, 0, area), grid_size, dtype=torch.float64)
preds = m.predict(grid)
print(f"pred shape: {preds.shape}")
```

## What it asserts

- `model.embed_.shape == (n, embed_dim)` (5000 × 10)
- `preds.shape == (grid_size²,)` (900)

## When to use

- The default `exact` kernel is fine for `n = 5000`. For larger `n`,
  switch to `nystrom` with `landmarks=200` or `fourier` with
  `features=128`.
- For dense prediction grids, `chunk` controls the per-call memory
  use of `Laker.predict`.
- For exploration of `lam`, use `Laker.path` to fit a sequence of
  `lam` values without rebuilding the kernel or preconditioner.

## See also

- [Choosing a kernel](../guides/choosing_kernel.md) for `n > 5000`
- [Streaming updates](../guides/streaming.md) for online updates
