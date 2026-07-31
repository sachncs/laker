"""Tests for :mod:`laker.bench`."""

import pytest
import torch

from laker.bench import BaseBench, Bench, SolveBench, bench, bench_all


def _sym_pd(n, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float64)
    return a @ a.T + torch.eye(n, dtype=torch.float64)


class TestBench:
    def test_dataclass_construction(self):
        b = Bench(name="x", n=10, time=1.0, iterations=5, residual=1e-6)
        assert b.name == "x"
        assert b.n == 10
        assert b.time == 1.0
        assert b.iterations == 5
        assert b.residual == 1e-6
        assert b.cond is None
        assert b.gap is None


class TestSolveBench:
    def test_run_returns_bench(self):
        n = 10
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        sb = SolveBench(
            name="test",
            op=lambda v: A @ v,
            prec=lambda v: v,
            rhs=rhs,
            tol=1e-10,
            max_iter=200,
            lam=1e-2,
        )
        result = sb.run()
        assert isinstance(result, Bench)
        assert result.name == "test"
        assert result.n == n
        assert result.iterations > 0
        assert result.residual < 1e-6

    def test_run_with_reference(self):
        n = 10
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        ref = torch.linalg.solve(A, rhs)
        sb = SolveBench(
            name="ref",
            op=lambda v: A @ v,
            prec=lambda v: v,
            rhs=rhs,
            reference=ref,
            tol=1e-12,
            max_iter=500,
            lam=1e-2,
        )
        result = sb.run()
        assert result.gap is not None

    def test_no_preconditioner(self):
        n = 10
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        sb = SolveBench(
            name="noprec",
            op=lambda v: A @ v,
            prec=None,
            rhs=rhs,
            tol=1e-12,
            max_iter=500,
        )
        result = sb.run()
        assert result.residual < 1e-6


class TestBaseBench:
    def test_run_returns_four_results(self):
        n = 20
        torch.manual_seed(0)
        e = torch.rand(n, 4, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)
        bb = BaseBench(embed=e, measurements=y, lam=1e-2, tol=1e-10, max_iter=200)
        results = bb.run()
        assert len(results) == 4
        names = {r.name for r in results}
        assert "LAKER" in names
        assert "Jacobi" in names
        assert "CG" in names
        assert "Descent" in names

    def test_laker_better_than_cg(self):
        n = 20
        torch.manual_seed(0)
        e = torch.rand(n, 4, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)
        bb = BaseBench(embed=e, measurements=y, lam=1e-2, tol=1e-10, max_iter=500)
        results = bb.run()
        by_name = {r.name: r for r in results}
        # LAKER should converge in fewer iterations than CG.
        assert by_name["LAKER"].iterations <= by_name["CG"].iterations


class TestFreeFunctions:
    def test_bench_function(self):
        n = 10
        A = _sym_pd(n)
        rhs = torch.randn(n, dtype=torch.float64)
        result = bench("test", lambda v: A @ v, lambda v: v, rhs)
        assert result.name == "test"

    def test_bench_all_function(self):
        n = 15
        torch.manual_seed(0)
        e = torch.rand(n, 4, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)
        results = bench_all(e, y, lam=1e-2, tol=1e-10, max_iter=200)
        assert len(results) == 4