# `laker.plot` — radio-map and convergence plots

`Plot` is a single class with static methods. All methods require
`matplotlib` (install the optional `viz` extra: `pip install laker[viz]`).

## Methods

### `Plot.image(predictions, size, x_min=0, x_max=1, y_min=0, y_max=1) -> np.ndarray`

Reshape flat predictions on a regular grid to a 2-D image array.
Returns a numpy array of shape `(size, size)` with `x` varying along
the first axis and `y` along the second.

```python
preds = torch.arange(16.0)  # 16 = 4x4
img = Plot.image(preds, size=4)  # shape (4, 4)
```

### `Plot.field(predictions, size, title="Radio Map Reconstruction", extent=None, label="RSS (dBm)", figsize=(6, 5), vmin=None, vmax=None) -> (figure, axes)`

Plot a 2-D radio-map reconstruction.

| Argument | Default | Meaning |
|----------|---------|---------|
| `predictions` | required | Flat tensor of length `size**2` |
| `size` | required | Points per axis |
| `title` | `"Radio Map Reconstruction"` | Plot title |
| `extent` | None | `(x_min, x_max, y_min, y_max)` for axis labels |
| `label` | `"RSS (dBm)"` | Colorbar label |
| `figsize` | `(6, 5)` | Figure size in inches |
| `vmin`, `vmax` | None | Color limits |

```python
import torch
from laker import Laker
from laker.data import Data

m = Laker(embed_dim=4, dtype=torch.float64)
g = Data.grid((0, 100, 0, 100), 50, dtype=torch.float64)
preds = m.predict(g)
Plot.field(preds, size=50, extent=(0, 100, 0, 100))
```

### `Plot.convergence(gaps, labels=None, title="Convergence Behaviour", xlabel="Iteration", ylabel="Relative Objective Gap", figsize=(6, 4)) -> (figure, axes)`

Plot one or more convergence curves (one per solver).

| Argument | Default | Meaning |
|----------|---------|---------|
| `gaps` | required | Per-solver list of objective gaps |
| `labels` | None | Per-solver labels |
| `title` | `"Convergence Behaviour"` | Plot title |
| `xlabel` | `"Iteration"` | X-axis label |
| `ylabel` | `"Relative Objective Gap"` | Y-axis label |
| `figsize` | `(6, 4)` | Figure size |

```python
Plot.convergence(
    gaps=[[0.1, 0.01, 0.001], [0.2, 0.05, 0.01]],
    labels=["LAKER", "Jacobi PCG"],
)
```

## Backend behaviour

All methods that import `matplotlib` (i.e. `field` and `convergence`)
raise `ImportError` with a clear "install matplotlib" message if the
package is missing. `image` does not import matplotlib so it works
without the `viz` extra.
