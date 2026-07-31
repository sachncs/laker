# `examples/flow.py` — streaming updates

Demonstrates `Laker.update` for online learning. After the initial
fit, new batches of measurements arrive and are appended to the
training set without a full refit. R² is monitored at every step
and must remain finite.

## What it does

```python
# Initial fit on 200 sensors.
m = Laker(embed_dim=10, dtype=torch.float64, verbose=False)
m.fit(x, y)
assert m.coef_.shape[0] == 200

# 4 batches of 20 sensors each.
for batch_idx in range(4):
    new_loc = ...
    new_tgt = ...
    m.update(new_loc, new_tgt, threshold=100, seed=0)
    assert m.coef_.shape[0] == 200 + 20 * (batch_idx + 1)
    assert m.embed_.shape[0] == 200 + 20 * (batch_idx + 1)
    score = m.score(locations, targets)
    assert score == score, f"score became NaN"  # not NaN
    preds = m.predict(new_loc)
    assert preds.std() > 0.1, "predictions became constant"
```

After 4 batches, the model is fitted on `200 + 4·20 = 280` samples.

## What it asserts

- `m.coef_.shape[0]` grows by exactly 20 per batch.
- `m.embed_.shape[0]` matches the cumulative sample count.
- `m.score(...)` is finite at every step (no NaN).
- New-batch predictions are non-constant.

If the cumulative new points exceed the threshold (100 in this
example), `update` raises `RuntimeError("update threshold exceeded")`
to force a full refit. The 4 × 20 = 80 total is below the threshold.

## Why this matters

Streaming is the recommended pattern for online / production systems
where data arrives continuously and full refits are too expensive.
The warm-start on the previous alpha gives near-constant
per-update cost regardless of how many samples have been seen.

See [Streaming updates](../guides/streaming.md) for the full
mechanics and configuration options.

## Source

[`examples/flow.py`](../../examples/flow.py)
