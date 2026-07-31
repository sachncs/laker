# End-to-end training

Beyond vanilla `fit` (which trains the encoder from scratch and solves
once), LAKER provides four training methods:

- `learn` — backprop through the kernel to train the encoder.
- `correct` — train a residual corrector MLP on `y - y_hat`.
- `calibrate` — uncertainty-aware training (NLL + calibration).
- `bilevel` — jointly optimise `lam` and the encoder.

All four share the underlying `laker.train.Trainer`.

## `learn` — end-to-end encoder optimisation

```python
m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
m.fit(x, y)  # initial fit
m.learn(x, y, lr=1e-2, epochs=5, rebuild=1, patience=5)
```

`learn` backpropagates the regression loss through the kernel
operator into the encoder MLP weights. Every `rebuild` epochs the
preconditioner is rebuilt and the system is re-solved for the current
embeddings; intermediate epochs use the most recent alpha.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 50 | Max epochs |
| `rebuild` | 10 | Rebuild preconditioner every `rebuild` epochs |
| `patience` | 5 | Early-stopping patience (no improvement) |

Validation: raises `RuntimeError` if the model has no encoder (call
`fit` first).

## `correct` — residual corrector MLP

```python
m.correct(x, y, val=0.2, epochs=3, patience=5, lr=1e-2, seed=0)
```

`correct` trains a small MLP on the residual `y - y_hat_laker`. The
MLP is a 2-layer network with `Tanh` activation and dropout, defined
in `laker.corrector.Corrector`. After training, `m.corrector` is
populated and `m.predict` automatically adds the corrector's output
to the kernel prediction.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `val` | 0.2 | Validation fraction (deterministic split) |
| `epochs` | 200 | Max training epochs |
| `patience` | 10 | Early-stopping patience on validation loss |
| `weight_decay` | 1e-2 | L2 regularisation on the corrector |
| `lr` | 1e-3 | Adam learning rate for the corrector |
| `seed` | None | Seed for the deterministic train/val split |

## `calibrate` — uncertainty-aware training

```python
m.calibrate(x, y, lr=1e-2, epochs=3, beta=0.1, subset=0.5, patience=5, seed=0)
```

`calibrate` optimises the embeddings on a combined negative
log-likelihood (NLL) and calibration penalty:

```
L = NLL(y | μ, σ²) + β · (E[r²] - E[σ²])²
```

The variance uses the exact closed-form for the Fourier kernel and a
differentiable distance-to-manifold proxy for other kernels. A random
subset of size `subset * n` is used each epoch to bound variance
estimation cost.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 50 | Max epochs |
| `beta` | 0.1 | Weight of the calibration penalty |
| `subset` | 0.2 | Fraction of training points used for variance each epoch |
| `patience` | 5 | Early-stopping patience on combined loss |
| `seed` | None | Seed for the variance-subset permutation |

## `bilevel` — joint `lam` and encoder optimisation

```python
m.bilevel(x_train, y_train, x_val, y_val, lr=1e-2, epochs=3, patience=5)
```

`bilevel` optimises a learnable log-`lam` against validation MSE.
The inner solve is via the trained `laker.bilevel.Bilevel` optimiser,
which uses the adjoint method (`laker.implicit.hypergradient`) to
compute the gradient of the validation loss through the PCG
fixed-point. A final `fit` is performed with the optimised `lam`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-3 | Outer Adam learning rate on the `lam` logit |
| `epochs` | 20 | Max outer epochs |
| `patience` | 5 | Early-stopping patience on validation loss |

Validation: all four training methods check input shapes and finite
values; `learn` requires a fitted encoder.

## When to use which

- Use `learn` if the initial fit is too smooth or too sharp.
- Use `correct` if the fit has small but systematic residuals that
  the kernel can't capture.
- Use `calibrate` if `m.variance` is poorly calibrated (compare
  `variance` against squared residuals).
- Use `bilevel` if `lam` is hard to pick by hand.
