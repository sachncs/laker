"""Benchmarking and comparison utilities for LAKER and baselines.

Public types: :class:`Bench` (single result), :class:`SolveBench`,
:class:`BaseBench`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from laker.kernel import Exact
from laker.prec import CCCP
from laker.solve import PCG, Descent, Jacobi

logger = logging.getLogger(__name__)


@dataclass
class Bench:
    """Container for a single benchmark run."""

    name: str
    n: int
    time: float
    iterations: int
    residual: float
    cond: Optional[float] = None
    gap: Optional[float] = None


class SolveBench:
    """Benchmark a single solver configuration."""

    def __init__(
        self,
        name: str,
        op: Callable[[torch.Tensor], torch.Tensor],
        prec: Optional[Callable[[torch.Tensor], torch.Tensor]],
        rhs: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        tol: float = 1e-10,
        max_iter: int = 1000,
        lam: float = 1e-2,
    ) -> None:
        self.name = name
        self.op = op
        self.prec = prec
        self.rhs = rhs
        self.reference = reference
        self.tol = tol
        self.max_iter = max_iter
        self.lam = lam

    def run(self) -> Bench:
        """Execute the benchmark and return :class:`Bench`."""
        pcg = PCG(tol=self.tol, max_iter=self.max_iter, verbose=False)
        start = time.perf_counter()
        if self.prec is not None:
            sol, _ = pcg.solve(self.op, self.prec, self.rhs)
        else:
            sol, _ = pcg.solve(self.op, lambda x: x, self.rhs)
        elapsed = time.perf_counter() - start

        rhs_norm = torch.linalg.norm(self.rhs).item()
        res = torch.linalg.norm(self.op(sol) - self.rhs).item() / rhs_norm if rhs_norm > 0 else 0.0

        gap = None
        if self.reference is not None:
            sol_op = self.op(sol)
            r = sol_op - self.rhs
            r2 = torch.dot(r, r).item()
            lam_term = self.lam * torch.dot(sol, self.op(sol)).item()
            obj_sol = r2 + lam_term
            obj_ref = self.lam * torch.dot(self.rhs, self.reference).item()
            gap = abs(obj_sol - obj_ref) / abs(obj_ref) if abs(obj_ref) > 1e-12 else None

        return Bench(
            name=self.name,
            n=self.rhs.shape[0],
            time=elapsed,
            iterations=pcg.iterations,
            residual=res,
            gap=gap,
        )


class BaseBench:
    """Run a head-to-head benchmark of LAKER vs baseline solvers."""

    def __init__(
        self,
        embed: torch.Tensor,
        measurements: torch.Tensor,
        lam: float = 1e-2,
        reference: Optional[torch.Tensor] = None,
        tol: float = 1e-10,
        max_iter: int = 1000,
    ) -> None:
        self.embed = embed
        self.measurements = measurements
        self.lam = lam
        self.reference = reference
        self.tol = tol
        self.max_iter = max_iter

    def run(self) -> List[Bench]:
        """Execute all baseline comparisons."""
        n = self.embed.shape[0]
        op_inst = Exact(self.embed, lam=self.lam)
        results = []

        prec = CCCP(num=None, gamma=1e-1, max_iter=100, tol=1e-6, verbose=False)
        start = time.perf_counter()
        prec.build(op_inst.matvec, n)
        pre_time = time.perf_counter() - start

        res = SolveBench(
            name="LAKER",
            op=op_inst.matvec,
            prec=prec.apply,
            rhs=self.measurements,
            reference=self.reference,
            tol=self.tol,
            max_iter=self.max_iter,
            lam=self.lam,
        ).run()
        res.time += pre_time
        results.append(res)

        jac = Jacobi(op_inst.diag())
        results.append(
            SolveBench(
                name="Jacobi",
                op=op_inst.matvec,
                prec=jac.apply,
                rhs=self.measurements,
                reference=self.reference,
                tol=self.tol,
                max_iter=self.max_iter,
                lam=self.lam,
            ).run()
        )

        results.append(
            SolveBench(
                name="CG",
                op=op_inst.matvec,
                prec=None,
                rhs=self.measurements,
                reference=self.reference,
                tol=self.tol,
                max_iter=self.max_iter,
                lam=self.lam,
            ).run()
        )

        gd = Descent(tol=1e-3, max_iter=50000, verbose=False)
        start = time.perf_counter()
        gd.solve(op_inst.matvec, self.measurements)
        gd_time = time.perf_counter() - start
        results.append(
            Bench(
                name="Descent",
                n=n,
                time=gd_time,
                iterations=gd.iterations,
                residual=gd.residual,
            )
        )

        return results


def bench(
    name: str,
    op: Callable[[torch.Tensor], torch.Tensor],
    prec: Optional[Callable[[torch.Tensor], torch.Tensor]],
    rhs: torch.Tensor,
    reference: Optional[torch.Tensor] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    lam: float = 1e-2,
) -> Bench:
    """Run a single solver benchmark."""
    return SolveBench(
        name=name,
        op=op,
        prec=prec,
        rhs=rhs,
        reference=reference,
        tol=tol,
        max_iter=max_iter,
        lam=lam,
    ).run()


def bench_all(
    embed: torch.Tensor,
    measurements: torch.Tensor,
    lam: float = 1e-2,
    reference: Optional[torch.Tensor] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> List[Bench]:
    """Run a head-to-head benchmark."""
    return BaseBench(
        embed=embed,
        measurements=measurements,
        lam=lam,
        reference=reference,
        tol=tol,
        max_iter=max_iter,
    ).run()


__all__ = ["Bench", "SolveBench", "BaseBench", "bench", "bench_all"]
