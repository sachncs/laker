# `laker.bench` — benchmarking harness

Public types: `Bench` (single result), `SolveBench` (single-solver
benchmark), `BaseBench` (head-to-head comparison), plus the
free-function `bench` and `bench_all`.

## Dataclasses

### `Bench`

```python
from laker.bench import Bench
b = Bench(
    name="LAKER",
    n=500,
    time=1.23,
    iterations=42,
    residual=1e-6,
    cond=12.5,        # optional condition number
    gap=1e-4,         # optional objective gap
)
```

| Field | Type | Meaning |
|-------|------|---------|
| `name` | str | Solver name |
| `n` | int | Problem dimension |
| `time` | float | Wall-clock time in seconds |
| `iterations` | int | PCG iterations executed |
| `residual` | float | Final relative residual |
| `cond` | float \| None | Estimated condition number |
| `gap` | float \| None | Objective gap (relative to reference solution) |

## Classes

### `SolveBench`

```python
from laker.bench import SolveBench
sb = SolveBench(
    name="LAKER",
    op=lambda v: A @ v,
    prec=lambda v: P_inv @ v,
    rhs=b,
    reference=alpha_ref,   # optional
    tol=1e-10,
    max_iter=200,
    lam=1e-2,
)
result = sb.run()
print(result.iterations, result.residual)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `name` | required | Solver label |
| `op` | required | Operator callable |
| `prec` | required | Preconditioner callable (or `None` for no preconditioner) |
| `rhs` | required | Right-hand side |
| `reference` | None | Optional reference solution for objective-gap computation |
| `tol` | 1e-10 | PCG tolerance |
| `max_iter` | 1000 | PCG max iterations |
| `lam` | 1e-2 | Regularisation (used only for objective-gap computation) |

`SolveBench.run()` returns a `Bench`.

### `BaseBench`

```python
from laker.bench import BaseBench
bb = BaseBench(
    embed=embeddings,        # (n, d)
    measurements=y,           # (n,)
    lam=1e-2,
    reference=alpha_ref,     # optional
    tol=1e-10,
    max_iter=200,
)
results = bb.run()  # list of 4 Bench objects
```

Runs a head-to-head comparison of four solvers on the same
`(embeddings, y)` problem:

1. LAKER (CCCP preconditioner + PCG)
2. Jacobi PCG (diagonal preconditioner)
3. Unpreconditioned CG
4. Gradient Descent (slow reference, no preconditioner)

The CCCP preconditioner is built once and reused for all four
solvers. Each solver is timed separately.

## Free functions

### `bench(name, op, prec, rhs, ...) -> Bench`

Wrapper around `SolveBench` for quick one-shot benchmarking.

### `bench_all(embed, measurements, ...) -> List[Bench]`

Wrapper around `BaseBench` for the standard head-to-head comparison.

## When to use

- Use `SolveBench` when you want to benchmark a single solver
  configuration against a reference.
- Use `BaseBench` for the standard LAKER-vs-baselines comparison.
- Use the `bench` / `bench_all` free functions for quick scripts.

The actual timing and statistical aggregation is delegated to
`benchmarks.executor.BenchmarkExecutor` (the concrete
`Executor` implementation).

## See also

- `benchmarks/reproducible.py` — full reproducible benchmark suite
- `benchmarks/approximations.py` — kernel approximation benchmarks
- `benchmarks/baseline.py` — baseline comparison with speedup ratios
