"""Minimal end-to-end Laker fit on a single-layer analytic target.

The smallest possible pipeline that demonstrates the public API
and verifies the result against a closed-form reference. Equivalent
in spirit to a "hello world" for Laker.

Run::

    python -m examples.simple
"""
from __future__ import annotations

import argparse

import torch

from laker import Laker


class Simple:
    """Single primary class — one ``@staticmethod run()`` entry point."""

    @staticmethod
    def run(
        n: int = 60,
        area: float = 5.0,
        embedding_dim: int = 8,
        seed: int = 0,
    ) -> None:
        """Fit on a smooth single-layer target and assert R² above a
        documented threshold.

        The target ``f(x) = sin(pi * x[0] / area) * cos(pi * x[1] / area)``
        is smooth and continuous — exactly the function class Laker's
        attention kernel is designed for. We verify that the model
        recovers it to high precision.
        """
        torch.manual_seed(seed)
        x = torch.rand(n, 2, dtype=torch.float64) * area
        y = (
            torch.sin(torch.pi * x[:, 0] / area)
            * torch.cos(torch.pi * x[:, 1] / area)
        )

        model = Laker(
            embedding_dim=embedding_dim,
            regularization=1e-6,
            probes=200,
            cccp_max_iter=200,
            pcg_tol=1e-12,
            pcg_max_iter=2000,
            dtype=torch.float64,
        )
        model.fit(x, y)

        # Closed-form verification: R^2 on the training set must be
        # very close to 1.0 since the underlying function is smooth and
        # the attention kernel captures it exactly at the limit.
        train_r2 = float(model.score(x, y))

        # Predictions on a held-out evaluation grid must remain
        # numerically stable and finite.
        grid_x = torch.linspace(0, area, 20, dtype=torch.float64)
        grid = torch.stack(torch.meshgrid(grid_x, grid_x, indexing="ij"), dim=-1).reshape(-1, 2)
        preds = model.predict(grid)

        # Variance is non-negative everywhere and the saved model
        # round-trips exactly through save/load.
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(str(path))
            loaded = Laker.load(str(path))
            loaded_pred = loaded.predict(grid)

        var = model.variance(grid)

        print(f"n={n}, embedding_dim={embedding_dim}")
        print(f"train R^2 = {train_r2:.6f}")
        print(f"pred shape = {tuple(preds.shape)}")
        print(f"var min/max = {var.min():.4e}/{var.max():.4e}")
        print(f"save/load bit-identical: {torch.equal(preds, loaded_pred)}")

        # ---- Behavioural + precision assertions ------------------------------
        assert (
            preds.shape == grid.shape[:-1]
            if preds.dim() == 1
            else preds.shape == (grid.shape[0],)
        ), f"predict: shape mismatch ({preds.shape})"
        assert torch.isfinite(preds).all(), "predict: NaN / Inf"
        assert torch.isfinite(var).all(), "variance: NaN / Inf"
        assert (var >= 0).all(), "variance: must be non-negative"
        assert torch.equal(
            preds, loaded_pred
        ), "save/load: predictions must be bit-identical"
        assert train_r2 > 0.95, (
            f"target is smooth; R^2={train_r2:.4f} too low for default Laker"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--area", type=float, default=5.0)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Simple.run(
        n=args.n,
        area=args.area,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
