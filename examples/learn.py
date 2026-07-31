"""Learn a regression model from sensor-style measurements.

Single-shot fit on scattered measurements, evaluate on the same
domain, demonstrate save/load round-trip integrity.

Run::

    python -m examples.learn
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from laker import Laker
from laker.data import Data


class Learn:
    """Single-shot fit, evaluate, persist, and reload."""

    @staticmethod
    def run(
        n: int = 200,
        area: float = 100.0,
        embedding_dim: int = 10,
        seed: int = 0,
    ) -> None:
        """Fit ``Laker``, assert outputs are well-formed, save and reload."""
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
            loss=2.5,
            ref=1.0,
            shadow=1.0,
            seed=seed,
        )

        # ---- data integrity ------------------------------------------------
        assert locations.shape == (n, 2), "data: location shape"
        assert targets.shape == (n,), "data: target shape"
        assert torch.isfinite(targets).all(), "data: targets non-finite"
        assert targets.std().item() > 5.0, "data: insufficient dynamic range"

        # ---- fit -----------------------------------------------------------
        model = Laker(
            embed_dim=embedding_dim,
            lam=1e-2,
            dtype=torch.float64,
        )
        model.fit(locations, targets)
        train_r2 = model.score(locations, targets)

        assert model.coef_ is not None, "fit: coef is None after fit"
        assert model.embed_ is not None, "fit: embeddings is None"
        assert model.coef_.shape == (n,), "fit: coef shape mismatch"
        assert model.embed_.shape == (n, embedding_dim), "fit: embeddings shape"
        assert (
            model.embed_.requires_grad is False
        ), "fit: frozen embeddings expected at fit time"

        # ---- predict -------------------------------------------------------
        query = torch.rand(20, 2, dtype=torch.float64) * area
        predictions = model.predict(query)
        assert predictions.shape == (20,), "predict: shape mismatch"
        assert torch.isfinite(predictions).all(), "predict: non-finite"
        assert predictions.std().item() > 0.1, "predict: model produced constant output"

        # ---- save / load round-trip ----------------------------------------
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(str(path))
            assert path.exists(), "save: file not written"
            loaded = Laker.load(str(path))

        loaded_predictions = loaded.predict(query)
        assert torch.allclose(
            predictions, loaded_predictions, atol=1e-5
        ), "round-trip: predictions diverge after reload"

        print(f"n={n} sensors, dim={embedding_dim}")
        print(f"train R^2={train_r2:.4f}")
        print(f"predictions mean={predictions.mean().item():.2f}")
        print("round-trip OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--area", type=float, default=100.0)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Learn.run(
        n=args.n,
        area=args.area,
        embedding_dim=args.embed_dim,
        seed=args.seed,
    )
