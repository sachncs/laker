# Examples

All snippets use the current public API. The single top-level class
is `Laker`; secondary classes are under `laker.<module>`.

For the smallest possible end-to-end pipeline, see
[`examples/simple.py`](../examples/simple.py) — the Laker equivalent of
a "hello world": fit, predict, save/load, with assertions on R² and
save/load bit-identity. All examples use the same single-class-per-file
convention; each module exposes one primary class with a
`@staticmethod run(...)` entry and an `if __name__ == "__main__"`
argparse block.

---

## Minimal fit — `simple.py`

The absolute smallest pipeline: fit on a smooth analytic target,
verify R² > 0.95, and assert save/load is bit-identical.

```python
import torch
from laker import Laker

torch.manual_seed(0)
x = torch.rand(60, 2, dtype=torch.float64) * 5.0
y = torch.sin(torch.pi * x[:, 0] / 5.0) * torch.cos(torch.pi * x[:, 1] / 5.0)

model = Laker(
    embedding_dim=8,
    regularization=1e-6,
    probes=200,
    cccp_max_iter=200,
    pcg_tol=1e-12,
    pcg_max_iter=2000,
    dtype=torch.float64,
)
model.fit(x, y)
print(f"train R^2 = {model.score(x, y):.6f}")
```

---

## Reproduce the n = 3 worked example

A small deterministic example that verifies the kernel matrix
matches `exp(E Eᵀ)` and the Laker solution matches `linalg.solve`.

```python
import torch
from laker import Laker
from laker.kernel import Exact

# Three hand-crafted embeddings (paper Eq. 53).
e = torch.tensor(
    [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
    dtype=torch.float64,
)
y = torch.tensor([-66.14, -65.77, -77.30], dtype=torch.float64)

class FixedEmbedding(torch.nn.Module):
    def forward(self, x):
        return e

model = Laker(
    embedding_dim=2,
    regularization=0.1,
    gamma=0.0,
    cccp_max_iter=10,
    pcg_tol=1e-12,
    pcg_max_iter=10,
    encoder=FixedEmbedding(),
    dtype=torch.float64,
)
# Dummy input rows; the encoder returns e regardless.
model.fit(torch.zeros(3, 2, dtype=torch.float64), y)

# Compare against the direct dense solve.
K = Exact(e, lambda_reg=0.1, dtype=torch.float64).to_dense()
alpha_ref = torch.linalg.solve(K, y)
assert torch.allclose(model.coef_, alpha_ref, atol=1e-6)
```

---

## Reconstruct a radio coverage map

```python
import torch
from laker import Laker
from laker.data import Data

area = 100.0
torch.manual_seed(0)

transmitters = torch.tensor(
    [[area * 0.20, area * 0.30], [area * 0.80, area * 0.70], [area * 0.50, area * 0.50]],
    dtype=torch.float64,
)
powers = torch.tensor([-30.0, -40.0, -35.0], dtype=torch.float64)

x = torch.rand(200, 2, dtype=torch.float64) * area
_, y = Data.field(
    x, transmitters, powers,
    path_loss_exponent=2.5, reference_distance=1.0, shadow_sigma=1.0, seed=0,
)

model = Laker(embedding_dim=12, regularization=1e-2, dtype=torch.float64)
model.fit(x, y)
r2 = model.score(x, y)
print(f"R^2 = {r2:.4f}")

# Predictions on a dense grid.
grid = Data.grid((0.0, area, 0.0, area), grid_size=40, dtype=torch.float64)
predictions = model.predict(grid)
```

---

## Predictive variance and uncertainty

```python
import torch
from laker import Laker
from laker.data import Data

torch.manual_seed(0)
transmitters = torch.tensor([[30.0, 50.0]], dtype=torch.float64)
powers = torch.tensor([-40.0], dtype=torch.float64)
x = torch.rand(80, 2, dtype=torch.float64) * 50.0
_, y = Data.field(x, transmitters, powers, seed=0)

model = Laker(embedding_dim=10, regularization=1e-2, dtype=torch.float64)
model.fit(x, y)
mean = model.predict(x[:10])
var = model.variance(x[:10])
# Variance is non-negative everywhere; in-sample anchored near zero.
assert (var >= 0).all()
```

---

## Tune regularisation on a held-out validation set

```python
import torch
from laker import Laker

torch.manual_seed(0)
n = 200
x = torch.rand(n, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])

model = Laker(embedding_dim=10, regularization=1e-1, dtype=torch.float64)
model.search("grid", x, y, regularizations=[1e-4, 1e-3, 1e-2, 1e-1])
print(f"chosen regularization = {model.regularization}")
```

---

## Bayesian hyperparameter search

```python
import torch
from laker import Laker

torch.manual_seed(0)
n = 200
x = torch.rand(n, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])

model = Laker(embedding_dim=10, dtype=torch.float64)
model.search(
    "bayes",
    x, y,
    n_calls=15,
    n_initial_points=5,
    regularization_bounds=(1e-4, 1.0),
)
```

---

## Learned embeddings via `learn`

`learn` optimises the encoder MLP weights end-to-end against the
regression objective.

```python
import torch
from laker import Laker

torch.manual_seed(0)
n = 200
x = torch.rand(n, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])

model = Laker(embedding_dim=10, dtype=torch.float64)
model.fit(x, y)
model.learn(x, y, lr=1e-3, epochs=50, rebuild_freq=10, patience=5)
```

---

## Residual corrector

A small MLP that fits the residual `y - y_hat_laker` after the base
model has trained.

```python
import torch
from laker import Laker

torch.manual_seed(0)
n = 300
x = torch.rand(n, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0]) + torch.cos(x[:, 1]) + 0.1 * torch.randn(n)

model = Laker(embedding_dim=10, dtype=torch.float64)
model.fit(x, y)
model.correct(x, y, val_fraction=0.2, epochs=200, patience=10)
predictions = model.predict(x)
```

---

## Uncertainty-aware training

`calibrate` minimises NLL plus a calibration penalty so that the
predicted variance tracks the actual residuals.

```python
import torch
from laker import Laker

torch.manual_seed(0)
n = 200
x = torch.rand(n, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0]) + 0.1 * torch.randn(n)

model = Laker(embedding_dim=10, dtype=torch.float64)
model.fit(x, y)
model.calibrate(x, y, lr=1e-3, epochs=50, beta=0.1)
var = model.variance(x)
```

---

## Regularisation path

```python
import torch
from laker import Laker

torch.manual_seed(0)
x = torch.rand(100, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0])

model = Laker(embedding_dim=10, dtype=torch.float64)
result = model.path(
    x, y, regularizations=[1.0, 0.1, 0.01, 1e-3]
)
for stage in result["stages"]:
    print(stage["regularization"], stage["pcg_iterations"])
```

---

## Bilevel joint regularisation + encoder tune

```python
import torch
from laker import Laker

torch.manual_seed(0)
n = 300
x = torch.rand(n, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0]) + 0.05 * torch.randn(n)
perm = torch.randperm(n)
n_val = n // 5
x_train, x_val = x[perm[n_val:]], x[perm[:n_val]]
y_train, y_val = y[perm[n_val:]], y[perm[:n_val]]

model = Laker(regularization=0.1, embedding_dim=8)
model.fit(x_train, y_train)
before = model.score(x_val, y_val)
model.tune(x_train, y_train, x_val, y_val, lr=5e-3, epochs=15, patience=10)
after = model.score(x_val, y_val)
assert before != after
```

---

## Streaming updates

```python
import torch
from laker import Laker

torch.manual_seed(0)
x_init = torch.rand(50, 2, dtype=torch.float64) * 50.0
y_init = torch.sin(x_init[:, 0] / 10.0)

model = Laker(
    embedding_dim=10, regularization=1e-3,
    rebuild_threshold=10_000,  # allow several batches
)
model.fit(x_init, y_init)
for batch_idx in range(4):
    x_new = torch.rand(20, 2, dtype=torch.float64) * 50.0
    y_new = torch.sin(x_new[:, 0] / 10.0)
    model.update(x_new, y_new, rebuild_threshold=10_000)
```

---

## Save, reload, and re-fit

```python
import torch
from laker import Laker

torch.manual_seed(0)
x = torch.rand(50, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0])

model = Laker(embedding_dim=10, dtype=torch.float64)
model.fit(x, y)
model.save("/tmp/laker.pt")
restored = Laker.load("/tmp/laker.pt")
assert torch.allclose(model.predict(x), restored.predict(x))
```

---

## Plot a recovered map

```python
import torch
from laker import Laker
from laker.data import Data
from laker.plot import Plot

torch.manual_seed(0)
transmitters = torch.tensor([[30.0, 50.0]], dtype=torch.float64)
powers = torch.tensor([-40.0], dtype=torch.float64)
x = torch.rand(100, 2, dtype=torch.float64) * 50.0
_, y = Data.field(x, transmitters, powers, seed=0)

model = Laker(embedding_dim=10, dtype=torch.float64)
model.fit(x, y)

grid = Data.grid((0.0, 50.0, 0.0, 50.0), grid_size=30, dtype=torch.float64)
predictions = model.predict(grid)
Plot.field(predictions, grid_size=30, area=50.0, title="Recovered map")
```

---

## Mixing kernel strategies on the same problem

Compare exact / Nyström / RFF predictions on a held-out set:

```python
import torch
from laker import Laker

torch.manual_seed(0)
x = torch.rand(150, 2, dtype=torch.float64) * 10.0
y = torch.sin(x[:, 0]) + torch.cos(x[:, 1])

x_test = torch.rand(50, 2, dtype=torch.float64) * 10.0

exact = Laker(embedding_dim=10, dtype=torch.float64)
exact.fit(x, y)
y_exact = exact.predict(x_test)

nys = Laker(kernel="nystrom", landmarks=80, embedding_dim=10, dtype=torch.float64)
nys.fit(x, y)
y_nys = nys.predict(x_test)

# Compare against the exact baseline.
rmse = float(((y_exact - y_nys) ** 2).mean().sqrt().item())
print(f"Nyström RMSE vs exact: {rmse:.4f}")
```
