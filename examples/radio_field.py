"""Radio-field reconstruction example.

Run:
    python -m examples.radio_field --n 2000 --embedding-dim 10
"""
from __future__ import annotations

import argparse

import torch

from laker import Laker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="sample count")
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--regularization", type=float, default=1e-2)
    args = parser.parse_args()

    torch.manual_seed(0)
    locations = torch.rand(args.n, 2) * 100.0
    targets = torch.sin(locations.sum(-1) / 30.0) + 0.1 * torch.randn(args.n)

    model = Laker(
        embedding_dim=args.embedding_dim,
        regularization=args.regularization,
    )
    model.fit(locations, targets)

    grid = torch.stack(torch.meshgrid(
        torch.linspace(0, 100, 50),
        torch.linspace(0, 100, 50),
        indexing="ij",
    ), dim=-1).reshape(-1, 2)
    predictions = model.predict(grid)
    print(f"R^2 score on training data: {model.score(locations, targets):.4f}")
    print(f"Prediction shape: {tuple(predictions.shape)}")


if __name__ == "__main__":
    main()
