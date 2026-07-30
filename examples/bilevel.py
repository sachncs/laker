"""Bilevel joint regularisation + encoder example.

Demonstrates :meth:`laker.Laker.tune`, which optimises the
regularisation weight and the encoder weights jointly via implicit
differentiation through the PCG fixed point.

Run:
    python -m examples.bilevel --epochs 50 --lr 1e-3
"""
from __future__ import annotations

import argparse

import torch

from laker import Laker


class Bilevel:
    """Bilevel tuning wrapper that reports before/after validation R²."""

    @staticmethod
    def run(epochs: int = 50, lr: float = 1e-3, n: int = 300) -> None:
        """Run the example and report before/after R²."""
        torch.manual_seed(0)
        x = torch.rand(n, 2) * 100.0
        y = torch.sin(x.sum(-1) / 30.0) + 0.1 * torch.randn(n)

        n_val = n // 5
        perm = torch.randperm(n)
        x_train, x_val = x[perm[n_val:]], x[perm[:n_val]]
        y_train, y_val = y[perm[n_val:]], y[perm[:n_val]]

        model = Laker(regularization=1e-2, embedding_dim=8)
        model.fit(x_train, y_train)
        before = model.score(x_val, y_val)
        model.tune(x_train, y_train, x_val, y_val, lr=lr, epochs=epochs)
        after = model.score(x_val, y_val)

        print(f"R^2 before tune: {before:.4f}")
        print(f"R^2 after tune:  {after:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    Bilevel.run(epochs=args.epochs, lr=args.lr, n=args.n)
