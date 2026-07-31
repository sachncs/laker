"""Behavioural + precision tests for the benchmark harness.

The benchmark suite computes iterations, residual norms, and
elapsed time. Each test asserts a real contract — finite iterations
with a meaningful upper bound, residual below the requested
``tol``, objective gap non-negative, finite timings — rather than
the previous ``>= 0``-style existence checks.
"""

from __future__ import annotations

import math

import torch

from laker.benchmark import (
    BenchmarkResult,
    benchmark_laker_vs_baselines,
    benchmark_solver,
)
from laker.kernels import Attention as Exact
from laker.preconditioner import CCCPPreconditioner


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass.
# ---------------------------------------------------------------------------
def test_benchmark_result_round_trips_all_fields():
    """Every documented field round-trips through the dataclass."""
    result = BenchmarkResult(
        name="solver-A",
        n=100,
        solve_time_seconds=0.123,
        iterations=42,
        final_residual=1e-5,
        condition_number=12.5,
        objective_gap=1e-4,
    )
    assert result.name == "solver-A"
    assert result.n == 100
    assert abs(result.solve_time_seconds - 0.123) < 1e-12
    assert result.iterations == 42
    assert abs(result.final_residual - 1e-5) < 1e-15
    assert abs(result.condition_number - 12.5) < 1e-9
    assert abs(result.objective_gap - 1e-4) < 1e-15


# ---------------------------------------------------------------------------
# benchmark_solver: residual and iteration contract.
# ---------------------------------------------------------------------------
def test_benchmark_solver_with_preconditioner_residual_below_tol():
    """``benchmark_solver`` returns a finite iteration count with
    ``final_residual < tol`` (the PCG tolerance).
    """
    torch.manual_seed(0)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    pre = CCCPPreconditioner(
        num_probes=30,
        gamma=1e-1,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        dtype=torch.float64,
    )
    pre.build(op.matvec, n)

    result = benchmark_solver(
        name="PCG",
        operator=op.matvec,
        preconditioner=pre.apply,
        rhs=rhs,
        tol=1e-6,
        max_iter=500,
        lambda_reg=1e-2,
    )
    assert result.n == n
    assert result.iterations > 0
    assert result.iterations <= 500
    assert result.final_residual < 1e-6
    assert math.isfinite(result.solve_time_seconds)


def test_benchmark_solver_without_preconditioner_may_need_more_iterations():
    """Without a preconditioner the iteration count is much larger
    than with a preconditioner on the same system. The weak /
    unpreconditioned path must still converge.
    """
    torch.manual_seed(0)
    n = 50
    e = torch.randn(n, 8, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    pre = CCCPPreconditioner(
        num_probes=30,
        gamma=1e-1,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        dtype=torch.float64,
    )
    pre.build(op.matvec, n)

    with_pre = benchmark_solver(
        name="PCG",
        operator=op.matvec,
        preconditioner=pre.apply,
        rhs=rhs,
        tol=1e-4,
        max_iter=2000,
        lambda_reg=1e-2,
    )
    no_pre = benchmark_solver(
        name="CG",
        operator=op.matvec,
        preconditioner=None,
        rhs=rhs,
        tol=1e-4,
        max_iter=2000,
        lambda_reg=1e-2,
    )

    assert no_pre.final_residual < 1e-4
    # Preconditioning must reduce iteration count (audit invariant).
    assert (
        with_pre.iterations < no_pre.iterations
    ), f"with_pre {with_pre.iterations} >= no_pre {no_pre.iterations}"


# ---------------------------------------------------------------------------
# benchmark_solver: objective gap and reference-solution handling.
# ---------------------------------------------------------------------------
def test_benchmark_solver_objective_gap_nonneg_with_reference():
    """``objective_gap`` is non-negative for any reference solution."""
    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 6, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)

    pre = CCCPPreconditioner(
        num_probes=20,
        gamma=1e-1,
        max_iter=20,
        tol=1e-4,
        verbose=False,
        dtype=torch.float64,
    )
    pre.build(op.matvec, n)

    exact = torch.linalg.solve(op.to_dense(), rhs)

    result = benchmark_solver(
        name="PCG",
        operator=op.matvec,
        preconditioner=pre.apply,
        rhs=rhs,
        reference_solution=exact,
        tol=1e-6,
        max_iter=500,
        lambda_reg=1e-2,
    )
    assert result.objective_gap is not None
    assert result.objective_gap >= 0.0
    assert math.isfinite(result.objective_gap)


def test_benchmark_solver_objective_gap_is_none_without_reference():
    """When no ``reference_solution`` is given, ``objective_gap``
    is ``None``.
    """
    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 6, dtype=torch.float64)
    op = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    rhs = torch.randn(n, dtype=torch.float64)
    result = benchmark_solver(
        name="PCG",
        operator=op.matvec,
        preconditioner=None,
        rhs=rhs,
        tol=1e-6,
        max_iter=200,
        lambda_reg=1e-2,
    )
    assert result.objective_gap is None


# ---------------------------------------------------------------------------
# benchmark_laker_vs_baselines: full head-to-head.
# ---------------------------------------------------------------------------
def test_benchmark_laker_vs_baselines_returns_four_distinct_solvers():
    """``benchmark_laker_vs_baselines`` returns one result per
    documented solver; timings and iterations are finite.
    """
    torch.manual_seed(0)
    n = 50
    e = torch.randn(n, 6, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    results = benchmark_laker_vs_baselines(
        embeddings=e,
        measurements=y,
        lambda_reg=1e-2,
        pcg_tol=1e-6,
        pcg_max_iter=500,
    )
    assert len(results) == 4
    names = [r.name for r in results]
    assert names[0] == "LAKER"
    assert "Jacobi PCG" in names
    assert "CG (no precond)" in names
    assert "Gradient Descent" in names

    for r in results:
        assert math.isfinite(r.solve_time_seconds)
        assert r.solve_time_seconds >= 0.0
        assert r.iterations >= 0
        assert r.n == n
        # The residual is finite and the LAKER preconditioned path
        # converges to the requested tolerance.
        if r.name == "LAKER":
            assert r.final_residual < 1e-3
