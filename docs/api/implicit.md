# `laker.implicit` — adjoint-method hypergradient

A single function: `hypergradient`.

## `hypergradient(op, prec, alpha, dl, params, tol, max_iter, verbose)`

Compute hypergradients of a scalar loss `L` through a PCG fixed-point
`α = A^{-1} y` via the adjoint method.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `op` | required | Callable applying `A(θ) @ v` |
| `prec` | required | Callable applying the preconditioner `P^{-1} @ v` |
| `alpha` | required | Fixed-point solution `α*` (detached) |
| `dl` | required | `dL / dα` (same shape as `alpha`) |
| `params` | required | List of parameters `θ_k` to differentiate with respect to |
| `tol` | 1e-6 | Adjoint PCG relative residual tolerance |
| `max_iter` | 500 | Adjoint PCG max iterations |
| `verbose` | False | Log adjoint solve progress |

Returns a list of hypergradient tensors, one per parameter, each with
the same shape as the corresponding parameter.

## Algorithm

1. **Adjoint solve.** Solve `A v = dL / dα` via PCG (the same operator
   and preconditioner). This costs one PCG solve.
2. **Per-parameter gradient.** For each `θ_k` in `params`:
   - If `θ_k.requires_grad`, compute
     `s = v^T · A(α*)` in a `torch.enable_grad()` block by re-evaluating
     `A(α*)` with `α*` cloned and `requires_grad=True`.
   - Return `-d s / dθ_k` via `torch.autograd.grad(s, θ_k)`.
   - If `θ_k.requires_grad` is False, return a zero tensor of the
     same shape.

The reason for cloning `α*` and recomputing `A(α*)` inside an enabled-grad
block: we want to differentiate through the operator's action on `α*`,
not through any of the in-place PCG bookkeeping.

## Use in bilevel learning

`laker.bilevel.Bilevel.bilevel` uses `hypergradient` to compute the
gradient of validation loss through the PCG-fixed-point `α(λ)`. The
outer Adam optimiser then updates `λ`.

## Limitations

- `op` and `prec` are each called once for the adjoint solve and once
  inside the `enable_grad` block. Total cost is `O(2 · cost(op, n))`.
- Parameters without `requires_grad` return zero gradients (they're
  considered fixed for the bilevel objective).
- This function requires `A(α*)` to be differentiable through
  `torch.autograd`. The PCG solver itself uses in-place updates
  which `torch.autograd` cannot backprop through, so the gradient is
  computed only through the final `A(α*)` call.

## References

- Christianson (1994), *"Reverse Accumulation and Implicit
  Equations"*. The original reverse-mode implicit differentiation.
- Bollt et al. (2020), *"On the Convergence of the Learning with
  Kernels Algorithm"*. Adjoint-based hyperparameter learning.
- See `algorithms/implicit.md` for more.
