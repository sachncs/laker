"""Reconstruct a synthetic radio field and evaluate grid predictions.

Generates ``n`` random samples from a noisy sinusoid, fits
:class:`laker.Laker`, and reports R² and grid prediction shape.

Run:
    python -m examples.radio_field --n 2000 --embedding-dim 10
"""
from __future__ import annotations

import argparse

import torch

from laker import Laker
from laker.data import Data


class RadioField:
    """Generate, fit, and evaluate a 2-D radio-field regression."""

    @staticmethod
    def run(n: int = 2000, embedding_dim: int = 10, regularization: float = 1e-2) -> None:
        """Run the example end-to-end and report scores."""
        torch.manual_seed(0)
        locations = torch.rand(n, 2) * 100.0
        # Smooth analytical target so a good fit is genuinely informative.
        targets = torch.sin(locations.sum(-1) / 30.0)

        model = Laker(
            embedding_dim=embedding_dim,
            regularization=regularization,
        )
        model.fit(locations, targets)

        grid = Data.grid((0.0, 100.0, 0.0, 100.0), grid_size=50)
        predictions = model.predict(grid)
        train_score = model.score(locations, targets)

        print(f"R^2 on training data: {train_score:.4f}")
        print(
            f"n={n}, embedding_dim={embedding_dim}: "
            f"prediction shape {tuple(predictions.shape)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--regularization", type=float, default=1e-2)
    args = parser.parse_args()
    RadioField.run(
        n=args.n,
        embedding_dim=args.embedding_dim,
        regularization=args.regularization,
    )
