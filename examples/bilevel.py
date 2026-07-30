"""Bilevel joint lambda + embedding example.

Run:
    python -m examples.bilevel --epochs 50 --lr 1e-3
"""
from __future__ import annotations

import argparse

import torch

from laker import Laker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(0)
    n = args.n
    x = torch.rand(n, 2) * 100.0
    y = torch.sin(x.sum(-1) / 30.0) + 0.1 * torch.randn(n)

    n_val = n // 5
    perm = torch.randperm(n)
    x_train, x_val = x[perm[n_val:]], x[perm[:n_val]]
    y_train, y_val = y[perm[n_val:]], y[perm[:n_val]]

    model = Laker(regularization=1e-2, embedding_dim=8)
    model.fit(x_train, y_train)
    before = model.score(x_val, y_val)
    model.tune(x_train, y_train, x_val, y_val, lr=args.lr, epochs=args.epochs)
    after = model.score(x_val, y_val)
    print(f"R^2 before tune: {before:.4f}")
    print(f"R^2 after tune:  {after:.4f}")


if __name__ == "__main__":
    main()
