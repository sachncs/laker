"""Reproducible benchmarks for LAKER critical paths.

Run with::

    python benchmarks/reproducible.py

Provides deterministic, seed-controlled benchmarks for every major LAKER
component. All random inputs are generated via ``torch.manual_seed(42)``.
"""

import logging
from typing import Optional

import torch

from benchmarks.executor import BenchmarkExecutor
from laker.embed import Position
from laker.kernel import Exact, Fourier, Grid, Neighbors, Nystrom
from laker.model import Laker
from laker.prec import CCCP
from laker.solve import PCG

logger = logging.getLogger(__name__)


class ReproducibleBenchmarkSuite:
    """Suite of reproducible benchmarks for LAKER critical paths."""

    def __init__(self, executor: Optional[BenchmarkExecutor] = None):
        self.executor = executor if executor is not None else BenchmarkExecutor()
        self.dtype = torch.float32
        self.embedding_dim = 10
        self.lambda_reg = 1e-2

    def warmup(self, kernel, vector, count: int = 20) -> None:
        for _ in range(count):
            kernel.matvec(vector)

    def make_embeddings(self, n: int, dim: int = 10) -> torch.Tensor:
        torch.manual_seed(42)
        x = torch.rand(n, 2, dtype=self.dtype) * 100.0
        embed = Position(input_dim=2, dim=dim, dtype=self.dtype)
        with torch.no_grad():
            return embed(x)

    def kernelmatvec(
        self,
        n: int = 5000,
        chunk_size: Optional[int] = 1024,
        trials: int = 50,
        warmup: int = 20,
    ) -> dict:
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        vector = torch.randn(n, dtype=self.dtype)
        kernel = Exact(
            embeddings,
            lam=self.lambda_reg,
            chunk=chunk_size,
            dtype=self.dtype,
        )
        self.warmup(kernel, vector, warmup)

        result = self.executor.run(
            f"kernelmatvec_n{n}",
            lambda: kernel.matvec(vector),
            trials=trials,
            warmup=0,
        )
        return {
            "n": n,
            "chunk_size": chunk_size,
            "matvec_ms_mean": result["mean_ms"],
            "matvec_ms_std": result["std_ms"],
        }

    def preconditioner_build(self, n: int = 5000, num_probes: int = 100) -> dict:
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
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
        }

    def pcg_solve(self, n: int = 5000, num_probes: int = 100) -> dict:
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
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

        pcg = PCG(tol=1e-10, max_iter=1000, verbose=False)

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
        torch.manual_seed(42)
        x_train = torch.rand(n, 2, dtype=self.dtype) * 100.0
        y_train = torch.randn(n, dtype=self.dtype)

        model = Laker(
            embed_dim=self.embedding_dim,
            lam=self.lambda_reg,
            gamma=1e-1,
            num=50,
            cccp_max=20,
            cccp_tol=1e-4,
            pcg_tol=1e-10,
            pcg_max=1000,
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
            "pcg_iters": getattr(model, "iters", None),
        }

    def approx_kernelmatvec(self, n: int = 2000, trials: int = 20) -> dict:
        embeddings = self.make_embeddings(n, dim=self.embedding_dim)
        vector = torch.randn(n, dtype=self.dtype)

        results = {}

        op_exact = Exact(embeddings, lam=self.lambda_reg, dtype=self.dtype)
        result = self.executor.run(
            "exact_matvec",
            lambda: op_exact.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["exact"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        op_nys = Nystrom(embeddings, lam=self.lambda_reg, num=200, dtype=self.dtype)
        result = self.executor.run(
            "nystrom_matvec",
            lambda: op_nys.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["nystrom"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        op_fourier = Fourier(embeddings, lam=self.lambda_reg, num=400, dtype=self.dtype)
        result = self.executor.run(
            "rff_matvec",
            lambda: op_fourier.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["rff"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        op_neighbors = Neighbors(embeddings, lam=self.lambda_reg, k=50, dtype=self.dtype)
        result = self.executor.run(
            "knn_matvec",
            lambda: op_neighbors.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["knn"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        op_grid = Grid(embeddings, lam=self.lambda_reg, grid_size=1024, dtype=self.dtype)
        result = self.executor.run(
            "ski_matvec",
            lambda: op_grid.matvec(vector),
            trials=trials,
            warmup=0,
        )
        results["ski"] = {"mean": result["mean_ms"], "std": result["std_ms"]}

        return {"n": n, **results}

    def generate_report(self) -> str:
        lines = [
            "# LAKER Benchmark Results",
            "",
            "**Date:** 2026-07-31  ",
            "**Platform:** Darwin (macOS)  ",
            f"**PyTorch:** {torch.__version__}  ",
            "**Dtype:** float32 (default)  ",
            "**Seed:** 42  ",
            "",
            "---",
            "",
        ]

        lines.append("## Kernel Matvec")
        lines.append("")
        lines.append("| n | chunk_size | mean (ms) | std (ms) |")
        lines.append("|---|------------|-----------|----------|")
        for n in [1000, 2000, 5000]:
            result = self.kernelmatvec(n=n, chunk_size=1024, trials=50, warmup=20)
            lines.append(
                f"| {result['n']} | {result['chunk_size']} | "
                f"{result['matvec_ms_mean']:.3f} | {result['matvec_ms_std']:.3f} |"
            )
        lines.append("")

        lines.append("## Approximation Matvec Comparison (n=2000)")
        lines.append("")
        lines.append("| method | mean (ms) | std (ms) |")
        lines.append("|--------|-----------|----------|")
        result = self.approx_kernelmatvec(n=2000, trials=20)
        for method in ["exact", "nystrom", "rff", "knn", "ski"]:
            lines.append(
                f"| {method} | {result[method]['mean']:.3f} | {result[method]['std']:.3f} |"
            )
        lines.append("")

        lines.append("## Preconditioner Build")
        lines.append("")
        lines.append("| n | N_r | time (ms) |")
        lines.append("|---|-----|-----------|")
        for n in [1000, 2000, 5000]:
            result = self.preconditioner_build(n=n, num_probes=100)
            lines.append(f"| {result['n']} | {result['num_probes']} | {result['build_ms']:.2f} |")
        lines.append("")

        lines.append("## PCG Solve")
        lines.append("")
        lines.append("| n | N_r | time (ms) | iters |")
        lines.append("|---|-----|-----------|-------|")
        for n in [1000, 2000, 5000]:
            result = self.pcg_solve(n=n, num_probes=100)
            lines.append(
                f"| {result['n']} | {result['num_probes']} | "
                f"{result['solve_ms']:.2f} | {result['pcg_iters']} |"
            )
        lines.append("")

        lines.append("## Full Fit")
        lines.append("")
        lines.append("| n | time (ms) | PCG iters |")
        lines.append("|---|-----------|-----------|")
        for n in [200, 500, 1000]:
            result = self.full_fit(n=n)
            lines.append(f"| {result['n']} | {result['fit_ms']:.2f} | {result['pcg_iters']} |")
        lines.append("")

        return "\n".join(lines)

    def save_report(self, path: str = "benchmarks/README.md") -> None:
        report = self.generate_report()
        with open(path, "w") as file:
            file.write(report)
        logger.info("Report saved to %s", path)

    def run_all(self) -> None:
        report = self.generate_report()
        logger.info("\n%s", report)
        self.save_report()

    @classmethod
    def run_all_default(cls) -> str:
        suite = cls()
        return suite.generate_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    suite = ReproducibleBenchmarkSuite()
    suite.run_all()
