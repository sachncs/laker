"""Performance benchmarks for LAKER critical paths.

Run this module directly to execute all benchmarks::

    python -m benchmarks.run
"""

import logging
from typing import Optional

import torch

from benchmarks.executor import BenchmarkExecutor
from laker.kernel import Exact
from laker.model import Laker
from laker.prec import CCCP
from laker.solve import PCG

logger = logging.getLogger(__name__)


class PerformanceBenchmarkSuite:
    """Suite of performance benchmarks for LAKER critical paths."""

    def __init__(self, executor: Optional[BenchmarkExecutor] = None) -> None:
        self.executor = executor if executor is not None else BenchmarkExecutor()
        self.dtype = torch.float64
        self.embedding_dim = 10
        self.lambda_reg = 1e-2

    @classmethod
    def benchmark_kernel_matvec(cls, n: int = 5000, chunk_size: Optional[int] = 1024) -> dict:
        suite = cls()
        return suite.kernel_matvec(n, chunk_size)

    @classmethod
    def benchmark_preconditioner_build(cls, n: int = 5000, num_probes: int = 100) -> dict:
        suite = cls()
        return suite.preconditioner_build(n, num_probes)

    @classmethod
    def benchmark_pcg_solve(cls, n: int = 5000, num_probes: int = 100) -> dict:
        suite = cls()
        return suite.pcg_solve(n, num_probes)

    @classmethod
    def benchmark_full_fit(cls, n: int = 500) -> dict:
        suite = cls()
        return suite.full_fit(n)

    def kernel_matvec(self, n: int = 5000, chunk_size: Optional[int] = 1024) -> dict:
        embeddings = torch.randn(n, self.embedding_dim, dtype=self.dtype)
        vector = torch.randn(n, dtype=self.dtype)
        kernel = Exact(embeddings, lam=self.lambda_reg, chunk=chunk_size, dtype=self.dtype)

        result = self.executor.run_repeated(
            f"kernel_matvec_n{n}",
            lambda: kernel.matvec(vector),
            repetitions=20,
        )
        return {
            "n": n,
            "chunk_size": chunk_size,
            "matvec_ms": result["mean_ms"],
        }

    def preconditioner_build(self, n: int = 5000, num_probes: int = 100) -> dict:
        embeddings = torch.randn(n, self.embedding_dim, dtype=self.dtype)
        kernel = Exact(embeddings, lam=self.lambda_reg, dtype=self.dtype)
        preconditioner = CCCP(
            num=num_probes,
            gamma=1e-1,
            max_iter=20,
            tol=1e-4,
            verbose=False,
            dtype=self.dtype,
        )

        result = self.executor.run_once(
            f"preconditioner_build_n{n}",
            lambda: preconditioner.build(kernel.matvec, n),
        )
        return {
            "n": n,
            "num_probes": num_probes,
            "build_ms": result["mean_ms"],
            "iters": preconditioner.max_iter,
        }

    def pcg_solve(self, n: int = 5000, num_probes: int = 100) -> dict:
        embeddings = torch.randn(n, self.embedding_dim, dtype=self.dtype)
        kernel = Exact(embeddings, lam=self.lambda_reg, dtype=self.dtype)
        rhs = torch.randn(n, dtype=self.dtype)

        preconditioner = CCCP(
            num=num_probes,
            gamma=1e-1,
            max_iter=20,
            tol=1e-4,
            verbose=False,
            dtype=self.dtype,
        )
        preconditioner.build(kernel.matvec, n)

        pcg = PCG(tol=1e-8, max_iter=500, verbose=False)

        result = self.executor.run_once(
            f"pcg_solve_n{n}",
            lambda: pcg.solve(kernel.matvec, preconditioner.apply, rhs),
        )
        return {
            "n": n,
            "num_probes": num_probes,
            "solve_ms": result["mean_ms"],
            "pcg_iters": pcg.iterations,
        }

    def full_fit(self, n: int = 500) -> dict:
        x_train = torch.rand(n, 2, dtype=self.dtype) * 100.0
        y_train = torch.randn(n, dtype=self.dtype)

        model = Laker(
            embed_dim=self.embedding_dim,
            lam=self.lambda_reg,
            gamma=1e-1,
            num=50,
            cccp_max=20,
            cccp_tol=1e-4,
            pcg_tol=1e-8,
            pcg_max=500,
            verbose=False,
            dtype=self.dtype,
        )

        result = self.executor.run_once(
            f"full_fit_n{n}",
            lambda: model.fit(x_train, y_train),
        )
        return {
            "n": n,
            "fit_ms": result["mean_ms"],
            "pcg_iters": getattr(model, "iters_", None),
        }

    def run_all(self) -> None:
        logger.info("=" * 60)
        logger.info("LAKER Performance Benchmarks")
        logger.info("=" * 60)

        for n in [1000, 2000, 5000]:
            result = self.kernel_matvec(n=n, chunk_size=1024)
            logger.info("Kernel matvec n=%d: %.2f ms", n, result["matvec_ms"])

        for n in [1000, 2000, 5000]:
            result = self.preconditioner_build(n=n, num_probes=100)
            logger.info("Preconditioner build n=%d: %.2f ms", n, result["build_ms"])

        for n in [1000, 2000, 5000]:
            result = self.pcg_solve(n=n, num_probes=100)
            logger.info(
                "PCG solve n=%d: %.2f ms (iters=%d)",
                n,
                result["solve_ms"],
                result["pcg_iters"],
            )

        for n in [200, 500, 1000]:
            result = self.full_fit(n=n)
            logger.info("Full fit n=%d: %.2f ms", n, result["fit_ms"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    suite = PerformanceBenchmarkSuite()
    suite.run_all()
