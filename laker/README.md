# LAKER Package

Learning-based Attention Kernel Regression for large-scale spectrum cartography.

## Package Structure

| Module | Class | Purpose |
|--------|-------|---------|
| `model.py` | `Laker` | High-level sklearn-compatible estimator. |
| `kernel.py` | `Kernel` (+ `Exact`, `Nystrom`, `Fourier`, `Neighbors`, `Grid`, `Hybrid`, `Spectrum`, `Distribute`) | Exact and approximate kernel operators. |
| `preconditioner.py` | `Preconditioner` (+ `CCCP`, `Adaptive`, `Jacobi`) | Learned data-dependent preconditioners. |
| `solve.py` | `Solve` (+ `PCG`, `Descent`) | Preconditioned conjugate gradient and gradient descent. |
| `embed.py` | `Embed` (+ `Position`, `Visual`) | Embedding modules. |
| `search.py` | `Search` | Grid and Bayesian hyperparameter search. |
| `fit.py` | `Fit` | Learned embeddings, correction, calibration, tuning. |
| `stream.py` | `Stream` | Online updates, regularisation paths, continuation. |
| `implicit.py` | `Implicit` | Hypergradient adjoint. |
| `plot.py` | `Plot` | Radio-map and convergence plotting. |
| `data.py` | `Data` | Synthetic radio-field generation and grid creation. |
| `helpers.py` | `Helpers` | Math/RNG helpers. |
| `backend.py` | `Backend` | Device/dtype/env management. |
| `base.py` | `Base` | Validation and tensor coercion. |
| `cli.py` | `CLI` | CLI handlers. |

## Quick Start

```python
import torch
from laker import Laker

locations = torch.rand(200, 2) * 100.0
measurements = torch.randn(200)

model = Laker(embedding_dim=10, regularization=1e-2)
model.fit(locations, measurements)

query = torch.rand(1000, 2) * 100.0
predictions = model.predict(query)
```

## Kernel Approximations

The `kernel` argument controls the operator:

- `"exact"` — exact attention kernel (default).
- `"nystrom"` — Nyström low-rank approximation.
- `"fourier"` — Random Fourier features.
- `"neighbors"` — sparse k-NN approximation.
- `"grid"` — Structured Kernel Interpolation.
- `"spectrum"` — Spectral shaping.
- `"hybrid"` — Two-scale combined approximation.

Example:

```python
model = Laker(kernel="nystrom", landmarks=100)
model.fit(locations, measurements)
```

## Design Principles

- **One primary class per module**: every module exports a single
  public class; helpers exist as `@staticmethod`.
- **Module-qualified names**: secondary classes live behind their
  parent module (`laker.kernel.Nystrom`, `laker.solve.PCG`).
- **No legacy aliases**: clean-break renames between releases;
  see `CONTRIBUTING.md`.
- **No semi-private naming**: no leading-underscore modules in the
  public surface.
- **Logging over prints**: all diagnostic output goes through
  `logging`.
