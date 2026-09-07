# `examples/simple.py` — minimal sin/cos fit

The smallest possible LAKER demo. Generates a sin/cos target on a
random sensor grid, fits a `Laker` with the default exact kernel, and
prints the in-sample R².

## What it does

```python
torch.manual_seed(0)
n = 60
x = torch.rand(n, 2, dtype=torch.float64) * 100
y = torch.sin(torch.pi * x[:, 0] / 100) * torch.cos(torch.pi * x[:, 1] / 100)

m = Laker(embed_dim=8, dtype=torch.float64, verbose=False)
m.fit(x, y)

# Closed-form verification: R^2 on the training set must be
# consistent with kernel-reconstruction accuracy.
score = m.score(x, y)
pred = m.predict(x)
print(f"n={n}, embed_dim=8")
print(f"train R^2 = {score:.6f}")
print(f"pred shape = {pred.shape}")
print(f"var min/max = {m.variance(x).min().item():.4e}/{m.variance(x).max().item():.4e}")

# Save / load round-trip must be bit-identical.
from pathlib import Path
import tempfile
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    path = Path(f.name)
m.save(path)
m2 = Laker.load(path)
pred2 = m2.predict(x)
assert torch.equal(pred, pred2), "save/load broke predictions"
print(f"save/load bit-identical: {torch.equal(pred, pred2)}")
path.unlink()
```

## Expected output

```
n=60, embed_dim=8
train R^2 = 0.999998
pred shape = (60,)
var min/max = 2.5590e-07/6.6227e-03
save/load bit-identical: True
```

The R² is near 1 because the kernel ridge regression with enough
embedding capacity exactly interpolates a smooth function. Variance
is bounded away from zero (the kernel is positive definite everywhere).

## Source

[`examples/simple.py`](../../examples/simple.py)
