# `examples/learn.py` — end-to-end encoder training

Demonstrates `Laker.learn`, which backpropagates the regression loss
through the kernel operator into the encoder MLP weights. After
training, the encoder is saved with the model and round-trips through
load with bit-identical predictions.

## What it does

```python
m = Laker(embed_dim=10, dtype=torch.float64, verbose=False)
m.fit(x, y)  # initial fit
m.learn(x, y, lr=1e-2, epochs=10, rebuild=1, patience=5)
print(f"R^2 after learn: {m.score(x, y):.4f}")

# Save / load round-trip.
m.save("model.pt")
m2 = Laker.load("model.pt")
pred1 = m.predict(x)
pred2 = m2.predict(x)
assert torch.equal(pred1, pred2)
```

## What it asserts

- `model.embed_.shape == (n, embed_dim)` after learn
- `model.coef_.shape == (n,)` after learn
- Save and load produce bit-identical predictions
- The encoder state dict is saved and restored

## When to use `learn`

Use `learn` when the default `Position` encoder's Fourier features +
Tanh MLP don't capture the structure of the data. After `learn`:
- The encoder MLP weights are tuned to minimise training MSE.
- The kernel, preconditioner, and PCG solve are re-run with the new
  embeddings.
- A final refit is performed to make the saved model self-contained.

If `learn` doesn't improve the fit, the data is likely better suited
to a custom encoder architecture. `learn` only adjusts the
`Position` MLP; it cannot change the number of Fourier features
or the layer architecture.

See [End-to-end training](../guides/training.md) for the full
training toolchain.

## Source

[`examples/learn.py`](../../examples/learn.py)
