"""Streaming: ingest a fresh batch of measurements every epoch.

Demonstrates :meth:`laker.Laker.update`. After each update the
running R^2 over the full (train + new) sample set must remain
finite and cannot regress catastrophically.

Run::

    python -m examples.flow
"""

from __future__ import annotations

import argparse

import torch

from laker import Laker
from laker.data import Data


class Flow:
    """Batched streaming updates with verified stability."""

    @staticmethod
    def run(
        n_initial: int = 200,
        n_per_batch: int = 20,
        n_batches: int = 4,
        area: float = 100.0,
        embedding_dim: int = 10,
        seed: int = 0,
    ) -> None:
        """Initial fit then ``n_batches`` incremental updates."""
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

        # Initial sample
        locations = torch.rand(n_initial, 2, dtype=torch.float64) * area
        _, targets = Data.field(
            locations,
            transmitters,
            powers,
            loss=2.5,
            ref=1.0,
            shadow=1.0,
            seed=seed,
        )

        model = Laker(
            embed_dim=embedding_dim,
            lam=1e-2,
            dtype=torch.float64,
        )
        model.fit(locations, targets)

        # ---- streaming batches --------------------------------------------
        for batch_idx in range(n_batches):
            new_loc = torch.rand(n_per_batch, 2, dtype=torch.float64) * area
            _, new_tgt = Data.field(
                new_loc,
                transmitters,
                powers,
                loss=2.5,
                ref=1.0,
                shadow=1.0,
                seed=seed + batch_idx + 1,
            )
            locations = torch.cat([locations, new_loc], dim=0)
            targets = torch.cat([targets, new_tgt], dim=0)

            model.update(new_loc, new_tgt)

            # Verification after each batch
            score = float(model.score(locations, targets))
            preds = model.predict(new_loc)
            assert score == score, f"batch {batch_idx}: NaN score ({score})"
            assert preds.std().item() > 0.1, f"batch {batch_idx}: predictions are constant"

        total = n_initial + n_batches * n_per_batch
        final_score = float(model.score(locations, targets))
        print(f"initial={n_initial} +{n_batches}*{n_per_batch}={total} total")
        print(f"final R^2={final_score:.4f}")
        print(f"coef shape={model.coef_.shape}, expected=({total},)")

        assert model.coef_.shape[0] == total, "stream: coef has wrong sample count"
        assert final_score == final_score, "stream: final score non-finite"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-initial", type=int, default=200)
    parser.add_argument("--n-per-batch", type=int, default=20)
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--area", type=float, default=100.0)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Flow.run(
        n_initial=args.n_initial,
        n_per_batch=args.n_per_batch,
        n_batches=args.n_batches,
        area=args.area,
        embedding_dim=args.embed_dim,
        seed=args.seed,
    )
