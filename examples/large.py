"""Large-scale fitting with chunked matrix-free evaluation.

Demonstrates :class:`laker.Laker` on ``n=5000`` synthetic measurements
where the attention kernel matrix :math:`K \\in \\mathbb{R}^{n\\times n}` is
never fully materialised. ``Data.field`` produces the radio-field
samples; ``Laker`` fits with ``chunk_size=1024``.

Run:
    python -m examples.large --n 5000
"""
from __future__ import annotations

import argparse

import torch

from laker import Laker
from laker.data import Data


class LargeScale:
    """Fit Laker on a large synthetic radio-field dataset."""

    @staticmethod
    def run(n: int = 5000, embedding_dim: int = 10, chunk_size: int = 1024) -> None:
        """Generate samples, fit, and report summary statistics."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        torch.manual_seed(42)
        locations = torch.rand(n, 2, device=device) * 100.0
        transmitters = torch.tensor(
            [[25.0, 25.0], [75.0, 75.0], [50.0, 80.0]],
            device=device,
        )
        # Power per transmitter (dBm). All use the package's
        # linear-power summation implemented in ``Data.field``.
        powers = torch.tensor([-40.0, -40.0, -40.0], device=device)
        _, observations = Data.field(
            locations,
            transmitters,
            powers,
            reference_distance=1.0,
            shadow_sigma=1.5,
            seed=42,
        )

        model = Laker(
            embedding_dim=embedding_dim,
            regularization=1e-2,
            chunk_size=chunk_size,
            device=device,
            dtype=torch.float64,
        )
        import time

        start = time.perf_counter()
        model.fit(locations, observations)
        elapsed = time.perf_counter() - start

        # Quick prediction sanity check.
        test_x = torch.rand(100, 2, device=device) * 100.0
        predictions = model.predict(test_x)
        print(
            f"n={n}, embedding_dim={embedding_dim}, chunk_size={chunk_size}: "
            f"fit in {elapsed:.2f}s, predictions shape {tuple(predictions.shape)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()
    LargeScale.run(
        n=args.n, embedding_dim=args.embedding_dim, chunk_size=args.chunk_size
    )
