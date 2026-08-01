# Examples

Each `examples/*.py` script is a self-contained runnable demo. Run
them directly with `python -m examples.<name>`.

| Script | What it shows |
|--------|---------------|
| [`simple.py`](../../examples/simple.py) | Minimal sin/cos fit. Trains a small `Laker` on synthetic data and reports the in-sample R². |
| [`learn.py`](../../examples/learn.py) | End-to-end encoder optimisation via `learn`, then save and reload with bit-identical predictions. |
| [`map.py`](../../examples/map.py) | Radio map reconstruction: fit on a sparse sensor grid, predict on a dense 50×50 grid, plot a side-by-side comparison. |
| [`scale.py`](../../examples/scale.py) | Large-scale (n=5000) fit, regularisation path via `path`, and full reconstruction on a 30×30 evaluation grid. |
| [`flow.py`](../../examples/flow.py) | Streaming updates: fit on 200 sensors, then add 4 batches of 20 sensors each via `update`, asserting non-regression of R² at every step. |
| [`tune.py`](../../examples/tune.py) | Hyperparameter search: validation-based `search` over a 5-decade `lam` grid, picking the best by validation R². |
| [`scalable_data.py`](../../examples/scalable_data.py) | Reliable ETL for the real-world UCF-50K corpus: download → verify → extract → index → clean → transform → load, with a traceability report. |
| [`scalable.py`](../../examples/scalable.py) | Full reproducible kernel sweep on UCF-50K: all kernel configurations benchmarked per scene, Pareto front, winner validated on the complete 256×256 grid over the whole corpus (resumable). |
| [`paper.py`](../../examples/paper.py) | Reproduces the LAKER paper's numerical experiment (arXiv:2604.25138, Section V) on the paper's synthetic scene: operator conditioning, PCG iterations vs Jacobi/GD baselines, and reconstruction RMSE/NMSE vs a Gaussian-process baseline. |

## How to run

```bash
# From the repo root:
python -m examples.simple
python -m examples.learn
python -m examples.map
python -m examples.scale
python -m examples.flow
python -m examples.tune
# Real-world dataset example (downloads ~10 GB on first run):
python -m examples.scalable_data --prepare --selfcheck 5
python -m examples.scalable --max-maps 25 --validate-maps 25
# Evaluate the winning configuration on the ENTIRE 50,000-map corpus:
python -m examples.scalable --validate-maps 0 --workers 8
# Reproduce the LAKER paper's Section V numerical experiment (synthetic scene, ~2 min):
python -m examples.paper
```

All examples except `scalable`/`scalable_data`/`paper` use synthetic data
via `laker.data.Data.field` and a random `Position` encoder (seed 42 by
default for reproducibility). Runtime is a few seconds on a CPU;
`scalable` downloads ~10 GB of real ray-traced radio maps and the
default sweep takes a couple of minutes.

## What each example asserts

| Script | Assertions |
|--------|-----------|
| `simple.py` | `coef.shape == (n,)`, `pred.shape == (n,)`, `R² > 0.95` |
| `learn.py` | `model.embed_.shape == (n, embed_dim)`, save → load → predict bit-identical |
| `map.py` | `train R² > 0.4`, grid RMSE < baseline RMSE |
| `scale.py` | `embed.shape == (n, embed_dim)`, `pred.shape == (size²,)` |
| `flow.py` | After each `update`: `coef.shape[0] == cumulative_n`, score is finite, predictions are not constant |
| `tune.py` | `R² > -1`, best `lam` spans at least 3 decades of the grid |
| `scalable_data.py` | each parquet is one 256×256 map with a single transmitter pixel; index counts match the corpus manifest (50,000 maps) |
| `scalable.py` | winner RMSE is < 65% of the mean-target baseline; winner is within 5% of `exact`; corpus fingerprint is stable |
| `paper.py` | reported `κ` is finite and the learned preconditioner keeps `κ(P·A)` at least an order of magnitude below `κ(A)` at every `n`; LAKER-PCG reaches the objective gap in fewer iterations than Jacobi-PCG; LAKER reconstruction RMSE is within 1% of the exact reference solve |

## Adding your own example

Each example follows the same pattern:

```python
# examples/my_example.py
"""One-line description."""
from __future__ import annotations
import argparse
import torch
from laker import Laker
# (imports and helpers)


class MyExample:
    @staticmethod
    def run(n=200, embedding_dim=4, seed=0) -> None:
        torch.manual_seed(seed)
        x, y = make_data(n, seed)
        m = Laker(embed_dim=embedding_dim, dtype=torch.float64, verbose=False)
        m.fit(x, y)
        # (your code)
        # assert something concrete
        assert m.score(x, y) > 0.0, "model failed to fit"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    MyExample.run(n=args.n, embedding_dim=args.embedding_dim, seed=args.seed)
```

Add it to `examples/__init__.py` so it's discoverable.
