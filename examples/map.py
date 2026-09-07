"""Reconstruct a coverage map on a held-out evaluation grid.

Fit a Laker model on scattered sensor measurements, then evaluate
the recovered signal on a dense grid against the underlying
path-loss ground truth. Verifies round-trip integrity with
variance check and predictions mean.

Run::

    python -m examples.map
"""

from __future__ import annotations

import argparse

import torch

from laker import Laker
from laker.data import Data


class Map:
    """Fit on scattered sensors, evaluate on a dense grid."""

    @staticmethod
    def run(
        n: int = 2000,
        area: float = 100.0,
        grid_size: int = 50,
        embedding_dim: int = 12,
        seed: int = 0,
    ) -> None:
        """Fit, predict on a grid, verify against the ground truth."""
        torch.manual_seed(seed)

        transmitters = torch.tensor(
            [
                [area * 0.20, area * 0.30],
                [area * 0.80, area * 0.70],
                [area * 0.50, area * 0.50],
                [area * 0.30, area * 0.80],
            ],
            dtype=torch.float64,
        )
        powers = torch.tensor([-30.0, -40.0, -35.0, -32.0], dtype=torch.float64)

        locations = torch.rand(n, 2, dtype=torch.float64) * area
        _, targets = Data.field(
            locations,
            transmitters,
            powers,
            loss=2.5,
            ref=1.0,
            shadow=1.0,
            seed=seed,
        )

        grid = Data.grid(
            (0.0, area, 0.0, area),
            size=grid_size,
            dtype=torch.float64,
        )
        _, ground_truth = Data.field(
            grid,
            transmitters,
            powers,
            loss=2.5,
            ref=1.0,
            shadow=0.0,
        )

        # ---- data sanity ---------------------------------------------------
        n_grid = grid_size * grid_size
        assert ground_truth.shape == (n_grid,), "data: grid truth shape"
        assert torch.isfinite(ground_truth).all(), "data: ground truth"
        assert ground_truth.std().item() > 5.0, "data: grid truth too flat"

        # ---- fit -----------------------------------------------------------
        model = Laker(
            embed_dim=embedding_dim,
            lam=1e-2,
            dtype=torch.float64,
        )
        model.fit(locations, targets)
        predictions = model.predict(grid)

        # ---- verification --------------------------------------------------
        assert predictions.shape == ground_truth.shape, "predict: shape mismatch"
        assert torch.isfinite(predictions).all(), "predict: non-finite"
        assert predictions.std().item() > 0.1, "predict: predictions are constant"
        train_r2 = model.score(locations, targets)
        assert train_r2 > 0.0, f"fit: train R^2 is non-positive ({train_r2:.4f})"

        baseline_rmse = float(((ground_truth - ground_truth.mean()) ** 2).mean().sqrt().item())
        model_rmse = float(((predictions - ground_truth) ** 2).mean().sqrt().item())
        improvement = (
            (baseline_rmse - model_rmse) / baseline_rmse * 100.0 if baseline_rmse > 0 else 0.0
        )

        print(f"sensors={n} grid={grid_size} dim={embedding_dim}")
        print(f"train R^2={train_r2:.4f}")
        print(f"grid RMSE={model_rmse:.2f} dBm (baseline={baseline_rmse:.2f})")
        print(f"improvement over mean baseline={improvement:.1f}%")

        assert improvement > 0.0, f"predict: model is worse than mean baseline ({improvement:.1f}%)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--area", type=float, default=100.0)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Map.run(
        n=args.n,
        area=args.area,
        grid_size=args.size,
        embedding_dim=args.embed_dim,
        seed=args.seed,
    )
