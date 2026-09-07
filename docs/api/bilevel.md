# `laker.bilevel` — bilevel hyperparameter learning

`Bilevel` jointly optimises the ridge weight `λ` against validation
loss via implicit differentiation. Used by `Laker.bilevel`.

## Class

### `Bilevel`

```python
from laker import Laker
m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
m._core.lam = 0.1
m.bilevel(
    x_train, y_train,
    x_val, y_val,
    lr=1e-2,
    epochs=20,
    patience=5,
)
```

## Constructor

```python
Bilevel(
    core,
    lr=1e-3,        # outer Adam learning rate on log(λ)
    epochs=20,
    patience=5,     # early-stopping patience
    tol=1e-6,       # adjoint PCG tolerance
    max_iter=500,   # adjoint PCG max iterations
    verbose=True,
)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `core` | required | The `laker.core.Core` instance |
| `lr` | 1e-3 | Outer Adam learning rate on the `log(λ)` parameter |
| `epochs` | 20 | Max outer iterations |
| `patience` | 5 | Early-stopping patience on validation loss |
| `tol` | 1e-6 | Adjoint PCG tolerance |
| `max_iter` | 500 | Adjoint PCG max iterations |
| `verbose` | True | Log per-epoch progress |

## Methods

### `Bilevel.bilevel(model, x_train, y_train, x_val, y_val, params=None)`

Run the bilevel optimisation.

| Argument | Default | Meaning |
|----------|---------|---------|
| `model` | required | The `Laker` instance (mutated in place) |
| `x_train`, `y_train` | required | Training data |
| `x_val`, `y_val` | required | Validation data |
| `params` | `None` | Hyperparameters to optimise. Defaults to a single learnable logit for `lam` |

Algorithm:

1. Set up the default parameter: a single learnable
   `logit_λ = log(λ)` (shape `(1,)`) with `requires_grad=True`.
2. Optimise with Adam (learning rate `self.lr`).
3. Each epoch:
   a. Convert `logit_λ` to `λ = exp(logit_λ).clamp(min=1e-8)`.
   b. Compute embeddings, kernel, preconditioner, solve.
   c. Compute validation MSE: `MSE(k_val @ α, y_val)`.
   d. Compute the gradient of validation MSE through `α` via
      `laker.implicit.hypergradient`.
   e. Apply the gradient manually to `logit_λ`.
   f. Adam step on `logit_λ`.
   g. Track best validation loss; early-stop if no improvement.
4. After convergence (or max epochs), refit the model on
   `(x_train, y_train)` with the optimised `λ`.

## What is being optimised

The default parameter is `logit_λ = log(λ)`, parameterised as
`λ = exp(logit_λ)`. This guarantees `λ > 0` without constraints.
The initial value is `log(model.lam)` so the bilevel loop starts at
the current value of `lam` on the `Laker` instance.

## When to use

Use `Bilevel` directly when:
- You have a custom `Core` and want to optimise `λ` without the
  `Laker` wrapper.
- You want to optimise multiple hyperparameters jointly (pass a
  custom `params` list).
- You're building a meta-learning or AutoML pipeline.

For most users, `Laker.bilevel` is the right entry point.

## Limitations

- The hypergradient path is currently approximate: the inner solve
  treats `α` as a fixed constant, so the gradient does not flow
  through the in-place PCG updates. The fix is `PCG(autograd=True)`
  but it's slower.
- The bilevel objective uses validation MSE; a custom objective
  requires writing a custom outer loop.
- Multiple hyperparameters can be optimised by passing a custom
  `params` list, but only the isotropic `λ` is supported out of the
  box.

## References

- Franceschi et al. (2017), *"Bilevel Programming for Hyperparameter
  Optimization and Meta-Learning"*. The bilevel framework.
- See `algorithms/bilevel.md` for the full mathematical treatment.
