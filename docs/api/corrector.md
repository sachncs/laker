# `laker.corrector` — residual corrector MLP

`Corrector` is a single class. A small MLP that adds a learned
correction to the base LAKER kernel prediction, capturing local
misspecification that the smooth kernel cannot represent.

## Class

### `Corrector`

```python
from laker.corrector import Corrector
c = Corrector(input_dim=2, output_dim=1, hidden_dim=32, dropout=0.1)
out = c(x)  # (n, output_dim)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `input_dim` | required | Input feature dimension (spatial coordinates) |
| `output_dim` | 1 | Output dimension (typically 1 for scalar regression) |
| `hidden_dim` | 32 | Hidden layer width |
| `dropout` | 0.1 | Dropout probability on the hidden layer |

Architecture:

```
Linear(input_dim, hidden_dim) → Tanh → Dropout(dropout) → Linear(hidden_dim, output_dim)
```

The hidden activation is `Tanh` (smooth, bounded) and the dropout
regularises the residual estimate. Total parameters:
`input_dim · hidden_dim + hidden_dim + hidden_dim · output_dim + output_dim`.

## Usage in LAKER

`Laker.correct` trains a `Corrector` on `y - y_hat_laker`:

```python
m = Laker(embed_dim=4, dtype=torch.float64)
m.fit(x, y)
m.correct(x, y, val=0.2, epochs=100, patience=10, lr=1e-3, seed=0)
print(m.corrector)  # <Corrector>
```

After training, `Laker.predict` automatically adds the corrector's
output to the kernel prediction:

```
predict(x) = kernel_prediction(x) + corrector(x)
```

This is invisible to the caller; the corrector is fully transparent.

## Why a small MLP

The corrector is intentionally tiny (default 32 hidden units) for
two reasons:

1. **LAKER already captures the dominant structure.** The kernel
   operator handles the global non-linear map; the corrector only
   needs to model the local residual.
2. **Avoid memorisation.** A large corrector on small training data
   can memorise the residual noise and destroy calibration. The small
   size plus dropout is a strong inductive bias against this.

If you need more capacity, increase `hidden_dim` or stack multiple
`Corrector`s in your own subclass.

## Saving and loading

The corrector is included in the saved file. The round-trip is
verified in `tests/test_store.py::TestCorrectorPersistence`. The
corrector's `state_dict` is saved and reloaded; no retraining is
required.

## Removing the corrector

Set `model.corrector = None` to remove the corrector. Subsequent
`predict` calls will return the kernel-only prediction.

## See also

- [guides/training.md](../guides/training.md) — practical examples
- [api/laker.md](laker.md) — the `Laker` facade that calls into this
