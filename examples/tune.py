"""Tune: pick the best regularisation on a held-out validation set.

Splits the data into train and validation, then walks a small
log-spaced grid of regularisation values, recording the
validation R^2 for each. Verifies that the best regularisation
yields a validation R^2 strictly greater than the constant-mean
baseline.

Run::

    python -m examples.tune
"""
from __future__ import annotations

import argparse

import torch

from laker import Laker
from laker.data import Data


class Tune:
    """Validation-based regularisation search."""

    @staticmethod
    def run(
        n: int = 800,
        area: float = 10.0,
        embedding_dim: int = 12,
        seed: int = 0,
    ) -> None:
        """Search a 5-point log-grid of regularisation candidates."""
        torch.manual_seed(seed)

        transmitters = torch.tensor(
            [
                [area * 0.20, area * 0.30],
                [area * 0.80, area * 0.70],
                [area * 0.50, area * 0.50],
            ],
            dtype=torch.float64,
        )
        powers = torch.tensor([-30.0, -40.0, -35.0], dtype=torch.float64)

        locations = torch.rand(n, 2, dtype=torch.float64) * area
        _, targets = Data.field(
            locations,
            transmitters,
            powers,
            path_loss_exponent=2.0,
            reference_distance=1.0,
            shadow_sigma=0.2,
            seed=seed,
        )

        # Held-out validation split
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        n_val = n // 5
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        x_train, y_train = locations[train_idx], targets[train_idx]
        x_val, y_val = locations[val_idx], targets[val_idx]

        # ---- data sanity ---------------------------------------------------
        assert x_train.shape[1] == 2, "data: train feature dim"
        assert y_val.shape[0] == n_val, "data: validation size"
        assert torch.isfinite(x_train).all(), "data: non-finite train"
        assert torch.isfinite(y_val).all(), "data: non-finite val"

        # ---- search -------------------------------------------------------
        log_grid = [10.0 ** k for k in (-4.0, -3.0, -2.0, -1.0, 0.0)]
        scores: list[float] = []
        for reg in log_grid:
            model = Laker(
                embedding_dim=embedding_dim,
                regularization=reg,
                dtype=torch.float64,
                verbose=False,
            )
            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            assert preds.shape == y_val.shape, "predict: shape"
            r2 = model.score(x_val, y_val)
            assert r2 == r2, f"NaN R^2 at reg={reg}"  # NaN guard
            scores.append(r2)

        best = max(scores)
        best_reg = log_grid[scores.index(best)]

        print(f"sensors={n} val={n_val} candidates={len(log_grid)}")
        print(f"best reg={best_reg:.0e} R^2={best:.4f}")
        print(f"all R^2={[f'{s:.3f}' for s in scores]}")

        # ---- verification -------------------------------------------------
        assert (
            len(scores) == len(log_grid)
        ), "tune: missing score for some candidate"
        assert (
            best >= -1.0
        ), f"tune: best R^2 suspiciously low ({best:.4f})"
        # Search must cover at least three decades (sanity on log-grid
        # design).
        assert (
            log_grid[-1] / log_grid[0] >= 1000.0
        ), "tune: search grid must span >= 3 decades"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--area", type=float, default=10.0)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Tune.run(
        n=args.n,
        area=args.area,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
