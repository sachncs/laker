# `laker.train` — training routines

`Trainer` orchestrates the four end-to-end training methods:
`learn`, `correct`, `calibrate`, `bilevel`. The `Laker` facade
exposes these as `Laker.learn`, `Laker.correct`, `Laker.calibrate`,
`Laker.bilevel`.

## Class

### `Trainer`

```python
from laker import Laker
m = Laker(embed_dim=4, dtype=torch.float64)
m._train.learn(m, x, y, lr=1e-2, epochs=5, rebuild=1, patience=5)
m._train.correct(m, x, y, val=0.2, epochs=3, patience=5, lr=1e-2, seed=0)
m._train.calibrate(m, x, y, lr=1e-2, epochs=3, beta=0.1, subset=0.5, patience=5, seed=0)
m._train.bilevel(m, x_tr, y_tr, x_va, y_va, lr=1e-2, epochs=3, patience=5)
```

Constructor takes a `laker.core.Core` instance.

## Methods

### `Trainer.learn(model, x, y, lr, epochs, rebuild, patience)`

Optimise the encoder weights end-to-end on the regression loss.

Backpropagates `(0.5 · ‖K(E) α − y‖²)` through the kernel operator
into the encoder MLP. Every `rebuild` epochs the preconditioner is
rebuilt and the system is re-solved for the current embeddings.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | required | The `Laker` instance |
| `x`, `y` | required | Training data |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 50 | Max epochs |
| `rebuild` | 10 | Rebuild preconditioner every `rebuild` epochs |
| `patience` | 5 | Early-stopping patience |

After convergence, restores the best-snapshot state (embeddings,
alpha, preconditioner, kernel) and refits.

### `Trainer.correct(model, x, y, val, epochs, patience, weight_decay, lr, seed)`

Train a residual corrector MLP on `y - y_hat_laker`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | required | The `Laker` instance |
| `x`, `y` | required | Training data |
| `val` | 0.2 | Validation fraction (deterministic split) |
| `epochs` | 200 | Max training epochs |
| `patience` | 10 | Early-stopping patience on validation loss |
| `weight_decay` | 1e-2 | L2 regularisation on the corrector |
| `lr` | 1e-3 | Adam learning rate for the corrector |
| `seed` | None | Seed for the train/val split |

After training, `model.corrector` is populated and `model.predict`
automatically adds the corrector's output to the kernel prediction.

### `Trainer.calibrate(model, x, y, lr, epochs, beta, subset, patience, seed)`

Uncertainty-aware training with NLL + calibration penalty.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 50 | Max epochs |
| `beta` | 0.1 | Weight of the calibration penalty |
| `subset` | 0.2 | Fraction of training points used for variance each epoch |
| `patience` | 5 | Early-stopping patience on combined loss |
| `seed` | None | Seed for the variance-subset permutation |

### `Trainer.bilevel(model, x_train, y_train, x_val, y_val, lr, epochs, patience)`

Delegate to `laker.bilevel.Bilevel.bilevel`. See
[api/bilevel.md](bilevel.md) and [algorithms/bilevel.md](../algorithms/bilevel.md).

## When to use the public methods

The public `Laker.learn`, `Laker.correct`, `Laker.calibrate`,
`Laker.bilevel` are thin wrappers that call into `Trainer`. Use
the public methods unless you have a custom `Core` instance.

## See also

- [End-to-end training](../guides/training.md) — practical examples
