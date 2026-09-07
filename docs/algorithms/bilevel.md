# Bilevel learning

`Laker.bilevel` jointly optimises the ridge weight `λ` and the encoder
embeddings against validation loss. The implementation is in
`laker.bilevel.Bilevel`.

## Why bilevel

`Laker.fit(x, y)` requires choosing `lam` upfront. Hand-picking
`lam` requires either a held-out validation set or cross-validation,
which is expensive when the PCG solve is the bottleneck.

`Laker.bilevel` automates this by treating `lam` as a learnable
hyperparameter. The outer loop minimises validation MSE by
backpropagating through the inner PCG solve. After convergence, the
model is refit on the full training data with the optimised `lam`.

## The bilevel problem

```
Outer:  min_λ  L_val( α*(λ) )
Inner:  α*(λ) = argmin_α  L_train(α; λ)  = (K(λ) + λ I)^{-1} y
```

The inner problem has a closed-form solution
`α = (K + λ I)^{-1} y`, so the outer problem is a smooth function
of `λ`. Differentiating it requires the adjoint method
([algorithms/implicit.md](implicit.md)).

## Implementation in `laker.bilevel.Bilevel`

```python
from laker import Laker
m = Laker(embed_dim=4, dtype=torch.float64, verbose=False)
m.bilevel(
    x_train, y_train,
    x_val, y_val,
    lr=1e-2,
    epochs=20,
    patience=5,
)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 1e-3 | Outer Adam learning rate on the `log(λ)` parameter |
| `epochs` | 20 | Max outer iterations |
| `patience` | 5 | Early-stopping patience on validation loss |
| `tol` | 1e-6 | Adjoint PCG tolerance |
| `max_iter` | 500 | Adjoint PCG max iterations |

After convergence, the model is refit on `(x_train, y_train)` with the
optimised `λ` and the fitted state is updated.

## The implementation step-by-step

In `Bilevel.bilevel`:

1. **Set up.** The default hyperparameter is a single learnable logit
   `logit_λ = log(λ)`. Adam is the outer optimiser.
2. **Forward.** Compute `embed = encoder(x_train)`, the kernel
   operator, and the preconditioner. Solve `α` for the current `λ`.
3. **Outer loss.** Compute the validation loss `MSE(k_val @ α, y_val)`
   (no grad through α).
4. **Backward.** Compute the gradient of validation loss through α
   via the adjoint method (`laker.implicit.hypergradient`).
5. **Step.** Adam updates `logit_λ`. Set `m.lam = exp(logit_λ).clamp(1e-8)`.
6. **Refit.** After convergence (or max epochs), call `model.fit(x_train,
   y_train)` with the optimised `lam` to refresh the fitted state.

## When to use

Use `bilevel` if:
- You have a clear train/val split and a clear validation metric.
- `Laker.search` over `lam_grid` is too coarse.
- You want to avoid manual hyperparameter tuning.

Don't use it for very large problems where each fit is expensive —
the bilevel loop refits the preconditioner and PCG solve every outer
epoch.

## Limitations

- The bilevel objective uses validation MSE; if you care about a
  different metric (R², RMSE, calibration), `search` with that
  metric is more direct.
- The hypergradient path is currently approximate: the inner solve
  treats `α` as a fixed constant, so the gradient does not flow
  through the in-place PCG updates. The fix is `PCG(autograd=True)`
  but it's slower.
- Only `lam` is jointly optimised. Joint encoder + `lam` optimisation
  is not yet exposed (use `learn` then `bilevel`).

## References

- Franceschi et al. (2017), *"Bilevel Programming for Hyperparameter
  Optimization and Meta-Learning"*. The bilevel framework.
- Bengio (2000), *"Gradient-Based Optimization of Hyperparameters"*.
  The implicit differentiation approach.
