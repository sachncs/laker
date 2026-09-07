# Implicit differentiation

`Laker.bilevel` optimises the ridge weight `λ` against validation
loss. The gradient of validation loss through `λ` requires
differentiating through the PCG fixed-point `α(λ)`. Backpropagating
through every PCG iteration is prohibitively expensive (and produces
a messy computation). Instead, LAKER uses the **adjoint method**:
solve one extra PCG for the adjoint vector, then compute the
hypergradient as a cheap per-parameter dot product.

## Adjoint method (for `A α = y`)

Let `A = A(θ)` depend on hyperparameters `θ`, `α = A^{-1} y`, and
`L(α(θ))` a scalar loss. Then

```
dL / dθ_k  =  - v^T (dA / dθ_k) α
```

where `v = A^{-1} (dL / dα)` is the adjoint vector. Computing `v` costs
one PCG solve (the same cost as the forward solve); the per-parameter
gradient is then a single `torch.autograd.grad` call.

## Implementation in `laker.implicit.hypergradient`

```python
from laker.implicit import hypergradient

v_grads = hypergradient(
    op=op,                  # callable: A(theta) @ v
    prec=prec,              # callable: P^{-1} @ v
    alpha=alpha.detach(),   # fixed-point solution
    dl=dl,                  # dL / dalpha
    params=[theta1, theta2, ...],
    tol=1e-6,
    max_iter=500,
    verbose=False,
)
```

Implementation in `laker/implicit.py`:

1. Solve `A v = dl` via PCG with the same operator and preconditioner.
2. Recompute `A α` in a differentiable context
   (`with torch.enable_grad(): alpha_g = alpha_c.clone().requires_grad_(True); A_alpha = op(alpha_g)`).
3. Compute `s = v^T · A_alpha` (scalar).
4. For each `param` in `params`:
   - If `param.requires_grad`, return `-d s / d param` via
     `torch.autograd.grad(s, param, retain_graph=True, allow_unused=True)`.
   - If not, return a zero tensor of the same shape.

## Use in bilevel learning

`laker.bilevel.Bilevel.bilevel` uses the adjoint method to compute
the gradient of validation MSE through the PCG solve:

```python
# in Bilevel.bilevel, outer loop:
val_loss = torch.mean((k_val @ alpha - y_val) ** 2)
dl = 2 * (k_val.T @ (k_val @ alpha - y_val)) / n_val
hgs = hypergradient(op, prec, alpha.detach(), dl, params, ...)
for p, hg in zip(params, hgs):
    p.grad = hg
opt.step()  # update lam
```

`hypergradient` does NOT use `alpha.requires_grad` because the forward
solve is non-differentiable through the in-place PCG updates. The
adjoint method sidesteps this by treating `alpha` as a fixed constant
and differentiating only through the second `op(alpha)` call.

## Why not just `torch.autograd`?

You might ask: why not just call `alpha.backward()` after the forward
solve? The answer is that PCG is implemented with in-place updates
(`x.add_(p, alpha=alpha)`) which `torch.autograd` cannot
backpropagate through. The cleanest fix is `PCG(autograd=True)`, which
switches to out-of-place updates and is autograd-friendly — but this
is significantly slower and not used in the hot path.

The adjoint method is both faster and works with the default
in-place PCG.

## Numerical caveats

- The operator `op` and preconditioner `prec` are called twice in
  `hypergradient`: once for the forward solve (already done by the
  caller) and once inside the `with torch.enable_grad():` block to
  build the scalar `s`. This is `O(2 · cost(op, n))` per call.
- The returned gradients have the same shape as the corresponding
  parameters.
- Parameters without `requires_grad=True` return zero tensors of the
  right shape.

## References

- Christianson (1994), *"Reverse Accumulation and Implicit
  Equations"*. The original reverse-mode implicit differentiation
  paper.
- Bai et al. (2002), *"Advances in Sensitivity Analysis and
  Parametric Programming"*. The implicit function theorem applied
  to optimization problems.
- Bollt et al. (2020), *"On the Convergence of the Learning with
  Kernels Algorithm"*. Adjoint-based hyperparameter learning for
  kernel machines.
