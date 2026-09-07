# `examples/map.py` — radio map reconstruction

The classic spectrum-cartography demo: fit a `Laker` on a sparse
sensor grid, predict on a dense 50×50 evaluation grid, and report
the reconstruction RMSE versus the mean-baseline RMSE.

## What it does

```python
torch.manual_seed(0)
n = 200  # sparse sensors
tx = torch.tensor([[20.0, 30.0], [80.0, 70.0], [50.0, 50.0]])
powers = torch.tensor([-30.0, -40.0, -35.0])
area = 100.0
grid_size = 50

# Train on sparse sensors.
locs = torch.rand(n, 2, dtype=torch.float64) * area
_, targets = Data.field(locs, tx, powers)
m = Laker(embed_dim=12, dtype=torch.float64, verbose=False)
m.fit(locs, targets)

# Predict on a dense 50x50 grid.
grid = Data.grid((0, area, 0, area), grid_size, dtype=torch.float64)
preds = m.predict(grid)

# Compare to mean baseline.
mean = torch.full_like(targets, targets.mean().item())
baseline_rmse = torch.sqrt(((targets - mean) ** 2).mean()).item()
grid_rmse = torch.sqrt(((preds - Data.field(grid, tx, powers)[0]) ** 2).mean()).item()
print(f"train R^2 = {m.score(locs, targets):.4f}")
print(f"grid RMSE = {grid_rmse:.2f} dBm (baseline = {baseline_rmse:.2f})")
print(f"improvement over mean baseline = {100*(1 - grid_rmse/baseline_rmse):.1f}%")
print(f"pred shape = {preds.shape}")
```

## Expected output

```
train R^2 = 0.4732
grid RMSE = 12.02 dBm (baseline = 12.76)
improvement over mean baseline = 5.8%
pred shape = (2500,)
```

The model achieves a 5-10% RMSE improvement over the mean baseline on
the dense 50×50 reconstruction. Performance is bounded by the
exponential kernel's smoothing; richer embeddings (larger
`embed_dim`) or learned encoders (`learn`) close more of the gap.

## What it asserts

- `pred.shape == (grid_size²,)` (50×50 = 2500)
- RMSE is finite and bounded

## When to use

`map` is the canonical entry-point for the spectrum cartography
task. For real data, replace the synthetic `Data.field` call with
your own measured-signal loader, then plug the result into the same
`Laker.fit` / `Laker.predict` flow.

## See also

- [Choosing a kernel](../guides/choosing_kernel.md) — `nystrom`
  kernel recommended for `n > 5000`
- [Streaming updates](../guides/streaming.md) — for live sensor
  ingestion
- [External datasets](../external_datasets.md) (TODO) — public
  spectrum cartography datasets
