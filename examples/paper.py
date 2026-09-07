"""Reproduce the LAKER paper's numerical experiment (arXiv:2604.25138, Section V).

The paper does not release a dataset: the evaluation scene is generated
in code by superposing transmitters with spatially decaying power
profiles and log-normal shadowing over a ``[0, 100] x [0, 100]`` m
region, drawing ``n in {50..2000}`` sensors uniformly and evaluating on
a dense ``45 x 45`` grid. This example regenerates that exact scene and
reproduces the paper's headline measurements with the library's own
CCCP preconditioner (``laker.prec.CCCP``, Algorithm 1 in the paper):

* conditioning: ``kappa(lambda I + G)`` and ``kappa(P (lambda I + G))``
  for the learned preconditioner ``P`` and the Jacobi ``P_J``;
* solver iterations to an objective-gap target of ``1e-3`` for LAKER-PCG,
  Jacobi-PCG and gradient descent, against a dense direct solve;
* reconstruction RMSE / NMSE of LAKER versus a rational-quadratic
  Gaussian-process baseline (GPRT) on the paper's ``45 x 45`` grid.

Results are written to ``outputs/paper/<run_id>/`` (events, a JSON
report with per-size tables, and a provenance manifest).

Run::

    python -m examples.paper
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Callable, Optional

import numpy as np
import torch
from numpy.random import default_rng

from laker.kernel import Exact, exp_safe
from laker.prec import CCCP
from laker.solve import PCG

SIZES: list[int] = [50, 200, 500, 1000, 2000]
GRID = 45
DOMAIN = 100.0
EMBED_DIM = 10
LAM = 1e-2
NOISE_STD = 1.5
GAMMA = 1e-1
TARGET_GAP = 1e-3
PCG_TOL = 1e-10
PCG_MAX = 800
GD_MAX = 3000
N_TRANSMITTERS = 3
SHADOW_STD = 6.0
SHADOW_DECORR = 25.0
SHADOW_MODES = 512


def rng(seed: int) -> np.random.Generator:
    """Deterministic RNG for a seed."""
    return default_rng(seed)


class Paper:
    """Reproduce the paper's Section V synthetic-scene experiment."""

    @staticmethod
    def provenance() -> dict:
        """Capture a reproducible environment fingerprint."""
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.getcwd()
        )
        git_dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.getcwd()
        )
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git_commit": git.stdout.strip() if git.returncode == 0 else "n/a",
            "git_dirty": bool(git_dirty.stdout.strip()),
        }

    @staticmethod
    def grid() -> np.ndarray:
        """Dense ``GRID x GRID`` evaluation coordinates in the domain."""
        lin = np.linspace(0.0, DOMAIN, GRID)
        xs: np.ndarray
        ys: np.ndarray
        xs, ys = np.meshgrid(lin, lin)
        return np.stack([xs.ravel(), ys.ravel()], axis=1)

    @staticmethod
    def field(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Regenerate the paper's synthetic scene.

        Returns ``(grid, sensors, true_grid, y)`` where ``true_grid`` is
        the ground-truth field on the ``45 x 45`` grid (dBm), ``y`` are
        noisy sensor observations, and ``grid``/``sensors`` are the
        corresponding coordinates.
        """
        r = rng(seed)
        grid = Paper.grid()
        sensors = r.uniform(0.0, DOMAIN, size=(n, 2))

        centers = r.uniform(10.0, DOMAIN - 10.0, size=(N_TRANSMITTERS, 2))
        powers = r.uniform(25.0, 40.0, size=N_TRANSMITTERS)
        exponents = r.uniform(2.5, 4.0, size=N_TRANSMITTERS)

        freqs = r.normal(0.0, 1.0, size=(SHADOW_MODES, 2))
        freqs = freqs / np.maximum(np.linalg.norm(freqs, axis=1, keepdims=True), 1e-9)
        freqs = freqs * (2.0 * math.pi / SHADOW_DECORR)
        amp = r.normal(0.0, SHADOW_STD / math.sqrt(SHADOW_MODES), size=SHADOW_MODES)
        phase = r.normal(0.0, SHADOW_STD / math.sqrt(SHADOW_MODES), size=SHADOW_MODES)

        def shadow(pts: np.ndarray) -> np.ndarray:
            ang = pts @ freqs.T
            return np.sin(ang) @ amp + np.cos(ang) @ phase

        def path_loss(pts: np.ndarray) -> np.ndarray:
            out = np.zeros(pts.shape[0])
            for t in range(N_TRANSMITTERS):
                dist = np.linalg.norm(pts - centers[t], axis=1)
                out = out + powers[t] - 10.0 * exponents[t] * np.log10(1.0 + dist)
            return out

        true_grid = path_loss(grid) + shadow(grid)
        y = path_loss(sensors) + shadow(sensors) + r.normal(0.0, NOISE_STD, size=n)
        return grid, sensors, true_grid, y

    @staticmethod
    def embedding(pts: np.ndarray) -> torch.Tensor:
        """Deterministic position-driven mapping ``x -> e in R^10`` (paper: d_e = 10)."""
        u = np.asarray(pts, dtype=np.float64) / DOMAIN
        u1, u2 = u[:, 0], u[:, 1]
        feats = np.stack(
            [
                np.ones_like(u1),
                u1,
                u2,
                u1 * u2,
                u1**2,
                u2**2,
                np.sin(math.pi * u1),
                np.cos(math.pi * u1),
                np.sin(math.pi * u2),
                np.cos(math.pi * u2),
            ],
            axis=1,
        )
        return torch.as_tensor(feats)

    @staticmethod
    def objective(alpha: torch.Tensor, g: torch.Tensor, y: torch.Tensor) -> float:
        """R(alpha) = ||G alpha - y||^2 + lam alpha^T G alpha (paper eq. 6)."""
        ga = g @ alpha
        return float(torch.dot(ga - y, ga - y) + LAM * torch.dot(alpha, ga))

    @staticmethod
    def reference(a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Direct dense solve of ``(lambda I + G) alpha = y`` (high-accuracy reference)."""
        return torch.linalg.solve(a, y)

    @staticmethod
    def preconditioned_cg(
        op: Callable[[torch.Tensor], torch.Tensor],
        prec: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        alpha_ref: torch.Tensor,
        g: torch.Tensor,
        y: torch.Tensor,
        max_iter: int = PCG_MAX,
    ) -> tuple[torch.Tensor, int]:
        """PCG with preconditioner ``prec``; count iterations to objective gap <= 1e-3."""
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        z = prec(r)
        p = z.clone()
        rho = float(torch.dot(r, z))
        ref_obj = abs(Paper.objective(alpha_ref, g, y))
        converged = PCG_MAX
        for it in range(1, max_iter + 1):
            ap = op(p)
            step = rho / float(torch.dot(p, ap))
            x = x + step * p
            r = r - step * ap
            if ref_obj > 0:
                gap = abs(Paper.objective(x, g, y) - Paper.objective(alpha_ref, g, y)) / ref_obj
                if gap <= TARGET_GAP:
                    converged = it
                    break
            z = prec(r)
            rho_new = float(torch.dot(r, z))
            if rho_new <= 0:
                break
            p = z + (rho_new / rho) * p
            rho = rho_new
        return x, converged

    @staticmethod
    def gradient_descent(
        a: torch.Tensor,
        g: torch.Tensor,
        y: torch.Tensor,
        alpha_ref: torch.Tensor,
        seed: int,
    ) -> tuple[int, float]:
        """Gradient descent on (6); step size selected by grid search.

        GD is expected to stagnate on these ill-conditioned systems
        (the paper reports objective gaps > 1), so work is capped at
        ``GD_MAX`` iterations per step size.
        """
        ref_obj = abs(Paper.objective(alpha_ref, g, y))
        best_iters = GD_MAX
        best_gap = float("inf")
        for eta in np.logspace(-5.0, -0.5, 12):
            alpha = torch.zeros_like(y)
            converged = False
            for it in range(1, GD_MAX + 1):
                alpha = alpha - eta * 2.0 * g @ (a @ alpha - y)
                if ref_obj > 0 and it % 10 == 0:
                    gap = abs(Paper.objective(alpha, g, y) - ref_obj) / ref_obj
                    if gap <= TARGET_GAP:
                        converged = True
                        break
                    best_gap = min(best_gap, gap)
            if converged:
                best_iters = min(best_iters, it)
        return best_iters, float(best_gap)

    @staticmethod
    def run(size: int, seed: int, probes: Optional[int], skip_gprt: bool) -> dict:
        """Run the full protocol for one problem size."""
        t_total = time.perf_counter()
        grid, sensors, true_grid, y = Paper.field(seed, size)
        y_t = torch.as_tensor(y)
        true_t = torch.as_tensor(true_grid)

        embed = Paper.embedding(sensors)
        kernel = Exact(embeddings=embed, lam=LAM)
        a = kernel.dense()
        g = a.clone()
        g.diagonal().add_(-LAM)
        alpha_ref = Paper.reference(a, y_t)
        ref_obj = abs(Paper.objective(alpha_ref, g, y_t))

        # condition numbers: kappa(A), kappa(P A), kappa(P_J A)
        kappa_raw = float(torch.linalg.svdvals(a)[0] / torch.linalg.svdvals(a)[-1])
        cccp = CCCP(
            num=probes, gamma=GAMMA, max_iter=200, tol=1e-6, verbose=False, dtype=torch.float64
        )
        t0 = time.perf_counter()
        cccp.build(kernel.matvec, size, seed=seed)
        prec_s = time.perf_counter() - t0
        p_mat = cccp.dense()
        pa = p_mat @ a
        kappa_laker = float(torch.linalg.svdvals(pa)[0] / torch.linalg.svdvals(pa)[-1])
        diag = kernel.diag()
        pj = diag.reciprocal()
        kappa_jacobi = float(
            torch.linalg.svdvals(pj.unsqueeze(1) * a)[0]
            / torch.linalg.svdvals(pj.unsqueeze(1) * a)[-1]
        )

        def jacobi_prec(v: torch.Tensor) -> torch.Tensor:
            return pj * v

        # iterations to objective-gap target (paper TAR.TOL = 1e-3)
        _, iters_laker = Paper.preconditioned_cg(
            kernel.matvec, cccp.apply, y_t, alpha_ref, g, y_t
        )
        _, iters_jacobi = Paper.preconditioned_cg(
            kernel.matvec, jacobi_prec, y_t, alpha_ref, g, y_t
        )
        t0 = time.perf_counter()
        iters_gd, gap_gd = Paper.gradient_descent(a, g, y_t, alpha_ref, seed)
        gd_s = time.perf_counter() - t0

        # tight LAKER solve (paper PCG.TOL ~ 1e-10) -> reconstruction coefficients
        pcg = PCG(tol=PCG_TOL, max_iter=PCG_MAX, verbose=False)
        alpha_laker, _ = pcg.solve(op=kernel.matvec, prec=cccp.apply, rhs=y_t)
        resid = float(
            torch.linalg.norm(kernel.matvec(alpha_laker) - y_t) / torch.linalg.norm(y_t)
        )
        obj_gap_laker = abs(Paper.objective(alpha_laker, g, y_t) - ref_obj) / ref_obj

        # reconstruction on the 45x45 grid (paper eq. 8)
        kq = kernel.eval(Paper.embedding(grid), embed)
        pred_laker = kq @ alpha_laker
        pred_ref = kq @ alpha_ref

        def rmse_nmse(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
            err = pred - truth
            return (
                float(torch.sqrt(torch.mean(err**2))),
                float(torch.dot(err, err) / torch.dot(truth, truth)),
            )

        rmse_laker, nmse_laker = rmse_nmse(pred_laker, true_t)
        rmse_ref, nmse_ref = rmse_nmse(pred_ref, true_t)

        gprt = None
        if not skip_gprt:
            t0 = time.perf_counter()
            gprt = Paper.gprt(sensors, y, grid, true_grid, seed)
            gprt_s = time.perf_counter() - t0
        else:
            gprt_s = None

        # loud-failure reproduction checks (see docs/examples/paper.md)
        for name, val in (
            ("kappa_raw", kappa_raw),
            ("kappa_laker", kappa_laker),
            ("kappa_jacobi", kappa_jacobi),
        ):
            assert math.isfinite(val), f"non-finite {name} at n={size}: {val}"
        assert kappa_laker <= kappa_raw / 10.0, (
            f"preconditioner not effective at n={size}: kappa(P A)={kappa_laker:.3e} "
            f"vs kappa(A)={kappa_raw:.3e}"
        )
        assert iters_laker < iters_jacobi, (
            f"LAKER-PCG not faster than Jacobi-PCG at n={size}: "
            f"{iters_laker} vs {iters_jacobi} iterations"
        )
        assert rmse_laker <= rmse_ref * 1.01 + 1e-9, (
            f"LAKER diverged from exact solve at n={size}: RMSE {rmse_laker} vs {rmse_ref}"
        )

        return {
            "n": size,
            "kappa_raw": kappa_raw,
            "kappa_laker": kappa_laker,
            "kappa_jacobi": kappa_jacobi,
            "iters_laker": iters_laker,
            "iters_jacobi": iters_jacobi,
            "iters_gd": iters_gd,
            "gap_gd": gap_gd,
            "residual_laker": resid,
            "obj_gap_laker": obj_gap_laker,
            "preconditioner_s": prec_s,
            "gd_s": gd_s,
            "gprt_s": gprt_s,
            "rmse_laker": rmse_laker,
            "rmse_ref": rmse_ref,
            "rmse_gprt": gprt["rmse"] if gprt else None,
            "nmse_laker": nmse_laker,
            "nmse_ref": nmse_ref,
            "nmse_gprt": gprt["nmse"] if gprt else None,
            "total_s": time.perf_counter() - t_total,
        }

    @staticmethod
    def gprt(
        sensors: np.ndarray, y: np.ndarray, grid: np.ndarray, truth: np.ndarray, seed: int
    ) -> dict:
        """GPRT baseline: Gaussian process regression, rational-quadratic kernel."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RationalQuadratic

        kernel = RationalQuadratic(
            length_scale=10.0, alpha=0.5, length_scale_bounds=(1e-2, 1e3)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=NOISE_STD**2,
            random_state=int(seed) % (2**31),
            n_restarts_optimizer=1,
        )
        gp.fit(sensors, y)
        pred = gp.predict(grid)
        err = pred - truth
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "nmse": float(np.dot(err, err) / np.dot(truth, truth)),
        }

    @staticmethod
    def print_tables(results: list[dict]) -> None:
        """Render the paper-style tables."""
        print("\n=== LAKER paper reproduction (synthetic scene, seed 0) ===")
        print(
            "\nTable I/II  conditioning and solver iterations "
            "(kappa(P(lambda I + G)) reduction, iterations to objective gap <= 1e-3)"
        )
        print(
            f"{'n':>5} | {'k(A)':>11} | {'k(P_A)':>11} {'k(P_JA)':>11} "
            f"| {'LAKER':>6} {'Jacobi':>7} {'GD':>6}"
        )
        for r in results:
            print(
                f"{r['n']:>5} | {r['kappa_raw']:11.3e} | {r['kappa_laker']:11.3e} "
                f"{r['kappa_jacobi']:11.3e} | {r['iters_laker']:>6} {r['iters_jacobi']:>7} "
                f"{r['iters_gd']:>6}"
            )
        print("\nTable III  reconstruction RMSE / NMSE on the 45x45 grid (dBm field)")
        print(
            f"{'n':>5} | {'RMSE LAKER':>11} {'RMSE ref':>10} {'RMSE GPRT':>11} "
            f"| {'NMSE LAKER':>12} {'NMSE GPRT':>11}"
        )
        for r in results:
            gprt = "   n/a" if r["rmse_gprt"] is None else f"{r['rmse_gprt']:10.4f}"
            nm = "    n/a" if r["nmse_gprt"] is None else f"{r['nmse_gprt']:11.3e}"
            print(
                f"{r['n']:>5} | {r['rmse_laker']:11.4f} {r['rmse_ref']:10.4f} {gprt} "
                f"| {r['nmse_laker']:12.3e} {nm}"
            )

    @staticmethod
    def main(argv: Optional[list[str]] = None) -> int:
        global GRID, LAM

        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
        parser.add_argument("--grid", type=int, default=GRID)
        parser.add_argument("--lam", type=float, default=LAM)
        parser.add_argument("--probes", type=int, default=None, metavar="Nr")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--skip-gprt", action="store_true")
        parser.add_argument("--out-dir", default="outputs/paper")
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args(argv)

        GRID, LAM = args.grid, args.lam
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = os.path.join(args.out_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        events = open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8")
        events.write(json.dumps({"event": "start", "argv": list(sys.argv), "ts": run_id}) + "\n")

        results = []
        for size in args.sizes:
            row = Paper.run(size, args.seed, probes=args.probes, skip_gprt=args.skip_gprt)
            results.append(row)
            events.write(json.dumps({"event": "size_done", "n": size, "row": row}) + "\n")
            print(f"size {size:>5}: done in {row['total_s']:.1f}s", flush=True)

        report = {
            "seed": args.seed,
            "grid": args.grid,
            "lam": args.lam,
            "sizes": list(args.sizes),
            "results": results,
        }
        report_path = os.path.join(run_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        manifest = {
            "run_id": run_id,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "argv": list(sys.argv),
            "provenance": Paper.provenance(),
            "files": {"events": events.name, "report": report_path},
        }
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        events.close()

        Paper.print_tables(results)
        print(f"run artifacts: {run_dir}")
        return 0


if __name__ == "__main__":
    raise SystemExit(Paper.main())
