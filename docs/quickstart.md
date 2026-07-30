# Quick Start

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy >= 1.23

For visualisation examples and tests you will also want `matplotlib`
(`.[viz]` extra) and the dev dependencies (`.[dev]`).

## Installation

For the latest release from PyPI:

```bash
pip install laker
```

From source (editable) with development and visualisation dependencies:

```bash
git clone https://github.com/sachncs/laker.git
cd laker
pip install -e ".[dev,viz]"
```

See [Installation](installation.md) for the full set of options,
including pinning torch versions.

## Basic Usage

The single public class is `Laker`. Fit a model and inspect the
predictions, the predictive variance, and the score:

```python
import torch
from laker import Laker

# 1000 sensor locations in a 100 x 100 m^2 area.
x_train = torch.rand(1000, 2) * 100.0
y_train = torch.sin(x_train[:, 0] / 10.0) + torch.cos(x_train[:, 1] / 7.0)

model = Laker(regularization=1e-2)
model.fit(x_train, y_train)

# Predictive mean / variance on held-out queries.
x_test = torch.rand(50, 2) * 100.0
mean = model.predict(x_test)
var = model.variance(x_test)

# R^2 on the training set (1.0 = perfect, 0.0 = predict-the-mean).
r2 = model.score(x_train, y_train)
```

The estimator trains a data-dependent preconditioner from random
probes of the kernel. With the defaults (`gamma=0.1`,
`probes=None`) the preconditioner converges in a few dozen iterations
of CCCP and PCG completes in tens of iterations. See [Theory](theory.md)
for the mathematics and [Performance](performance.md) for scaling.

## Constructor reference

```python
class Laker(
    embedding_dim: int = 10,
    regularization: float = 1e-2,
    gamma: float = 1e-1,
    probes: Optional[int] = None,
    epsilon: float = 1e-8,
    base_rho: float = 0.05,
    cccp_max_iter: int = 200,
    cccp_tol: float = 1e-6,
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 1000,
    chunk_size: Optional[int] = None,
    encoder: Optional[torch.nn.Module] = None,
    kernel: str = "exact",
    landmarks: Optional[int] = None,
    features: Optional[int] = None,
    neighbors: Optional[int] = None,
    grid_size: Optional[int] = None,
    blend: float = 0.5,
    selection: str = "greedy",
    knots: int = 5,
    pilot: Optional[int] = None,
    distributed: bool = False,
    embedding_dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
)
```

| Parameter | Default | Purpose |
| --- | --- | --- |
| `embedding_dim` | 10 | Width of the embedding MLP. |
| `regularization` | 1e-2 | Tikhonov $\lambda$. |
| `gamma` | 0.1 | CCCP shrinkage. |
| `probes` | adaptive | Random probes for preconditioner; auto = $\max(200, 2\sqrt{n})$. |
| `epsilon` | 1e-8 | Numerical safety floor. |
| `base_rho` | 0.05 | Base shrinkage $\rho_0$. |
| `cccp_max_iter` | 200 | CCCP outer iterations. |
| `cccp_tol` | 1e-6 | CCCP convergence tol. |
| `pcg_tol` | 1e-6 | PCG relative-residual tol. |
| `pcg_max_iter` | 1000 | PCG iteration cap. |
| `chunk_size` | None | Tile size; auto-selected above 5000 samples. |
| `encoder` | None | Replace the position embedding with a custom module. |
| `kernel` | "exact" | `"exact"`, `"nystrom"`, `"fourier"`, `"neighbors"`, `"grid"`, `"spectrum"`, `"hybrid"`. |
| `landmarks` | None | Nyström landmark count. |
| `features` | None | Random Fourier feature count. |
| `neighbors` | None | Sparse k-NN neighbour count. |
| `grid_size` | None | SKI grid resolution. |
| `blend` | 0.5 | Hybrid (two-scale) mix weight. |
| `selection` | "greedy" | Nyström landmark selection rule. |
| `knots` | 5 | Spectral kernel spline knots. |
| `pilot` | None | Nyström pilot sample size. |
| `distributed` | False | Use the multi-device kernel. |
| `embedding_dtype` | None | Dtype for the embedding computation. |
| `device` | None | `torch.device` (auto-detects CUDA). |
| `dtype` | None | Floating dtype for the solver. |
| `verbose` | True | Logging verbosity. |

## Kernel approximations

For very large $n$ use one of the low-rank or sparse strategies by
passing `kernel=...`:

```python
# Nyström low-rank.
model = Laker(kernel="nystrom", landmarks=200)

# Random Fourier features (stationary Gaussian surrogate).
model = Laker(kernel="fourier", features=400)

# Sparse k-NN.
model = Laker(kernel="neighbors", neighbors=50)

# Structured Kernel Interpolation.
model = Laker(kernel="grid", grid_size=1024)

# Spectral-shaped kernel via a monotone spline on the eigenvalues.
model = Laker(kernel="spectrum", knots=5, dtype=torch.float64)

# Two-scale: global Nyström + local sparse k-NN.
model = Laker(kernel="hybrid", landmarks=200, neighbors=30)
```

The audit found that the matvec implementations of Nyström and RFF
differ from `to_dense @ x`. The exact kernel has no such gap. See
[API Reference](api.md) for the operator-by-operator invariants.

## Save / Load

```python
model.save("laker_model.pt")
loaded = Laker.load("laker_model.pt")
assert torch.allclose(loaded.predict(x_test), model.predict(x_test))
```

The on-disk file contains a `format_version` field (currently `2`)
that the loader checks before deserialising.

## CLI

```bash
laker fit      --locations x_train.pt --measurements y_train.pt --output model.pt
laker predict  --model model.pt --locations x_test.pt --output y_pred.pt
```

The CLI accepts `.pt` and `.npy` files. See [CLI](cli.md) for the
full flag list.

## Working with hyperparameters

Search and validation-based regularisation selection:

```python
model = Laker(embedding_dim=10, dtype=torch.float64)
model.search("grid", x_train, y_train, regularizations=[1e-4, 1e-3, 1e-2, 1e-1])
model.search("bayes", x_train, y_train, n_calls=15)
```

Regularisation path:

```python
path = model.path(x_train, y_train, regularizations=[1.0, 0.1, 0.01, 0.001])
```

Tuning on a held-out set:

```python
model.fit(x_train, y_train)
model.tune(x_train, y_train, x_val, y_val, lr=5e-3, epochs=15)
```

## Streaming updates

```python
model.fit(x_train, y_train)
model.update(x_new, y_new, rebuild_threshold=10_000)
```

When the cumulative update count exceeds `rebuild_threshold` the next
update triggers a full refit rather than another incremental update.

## Saving and loading state

```python
state = model.get_params()
m2 = Laker(**state)
```

`set_params(**kwargs)` validates against the closed parameter set
defined on `Laker.PARAMS`. Unknown keys raise `ValueError`.

## Where to go next

- [Theory](theory.md) — the math behind the preconditioner.
- [Examples](examples.md) — real-world-style snippets.
- [Migration](migration.md) — if you're porting from the legacy
  `LAKERRegressor` API.
- [Performance](performance.md) — scaling to large `n`.
- [Troubleshooting](troubleshooting.md) — when something goes wrong.
