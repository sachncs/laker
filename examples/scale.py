"""Scale: fit a large sensor network, predict on a held-out grid.

Real-world deployment scale: thousands of sensors, dense evaluation
grid, mass-prediction over the domain. Verifies that large
matrices are produced and the model scales without errors.

Run::

    python -m examples.scale
"""

from __future__ import annotations

import argparse
import time

import torch

from laker import Laker
from laker.data import Data


class Scale:
    """Large-scale fit and predict with timing."""

    @staticmethod
    def run(
        n: int = 5000,
        area: float = 200.0,
        grid_size: int = 30,
        embedding_dim: int = 10,
        seed: int = 0,
    ) -> None:
        """Fit on ``n`` sensors and predict on a ``grid_size`` × ``grid_size`` grid."""
        torch.manual_seed(seed)

        transmitters = torch.tensor(
            [
                [area * 0.15, area * 0.15],
                [area * 0.85, area * 0.85],
                [area * 0.50, area * 0.50],
                [area * 0.20, area * 0.85],
                [area * 0.85, area * 0.20],
            ],
            dtype=torch.float64,
        )
        powers = torch.tensor([-30.0, -40.0, -35.0, -32.0, -38.0], dtype=torch.float64)

        locations = torch.rand(n, 2, dtype=torch.float64) * area
        _, targets = Data.field(
            locations,
            transmitters,
            powers,
            loss=2.7,
            ref=1.0,
            shadow=1.0,
            seed=seed,
        )

        n_grid = grid_size * grid_size
        grid = Data.grid(
            (0.0, area, 0.0, area),
            size=grid_size,
            dtype=torch.float64,
        )

        # ---- data sanity ---------------------------------------------------
        assert locations.shape == (n, 2), "data: location shape"
        assert targets.shape == (n,), "data: target shape"
        assert grid.shape == (n_grid, 2), "data: grid shape"
        assert torch.isfinite(targets).all(), "data: targets non-finite"

        # ---- fit (timed) ---------------------------------------------------
        model = Laker(
            embed_dim=embedding_dim,
            lam=1e-3,
            dtype=torch.float64,
        )
        t0 = time.perf_counter()
        model.fit(locations, targets)
        fit_seconds = time.perf_counter() - t0

        assert model.coef.shape == (n,), "fit: coef shape mismatch"
        assert model.embed.shape == (n, embedding_dim), "fit: embeddings shape mismatch"

        # ---- predict (timed) ------------------------------------------------
        t0 = time.perf_counter()
        predictions = model.predict(grid)
        predict_seconds = time.perf_counter() - t0

        assert predictions.shape == (n_grid,), "predict: shape mismatch"
        assert torch.isfinite(predictions).all(), "predict: non-finite"
        assert predictions.std().item() > 0.1, "predict: constant output"

        print(f"sensors={n} grid={grid_size} dim={embedding_dim}")
        print(f"fit seconds={fit_seconds:.2f}")
        print(f"predict seconds={predict_seconds:.2f}")
        print(f"pred shape={tuple(predictions.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--area", type=float, default=200.0)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Scale.run(
        n=args.n,
        area=args.area,
        grid_size=args.size,
        embedding_dim=args.embed_dim,
        seed=args.seed,
    )
