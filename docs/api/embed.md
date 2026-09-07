# `laker.embed` — embedding modules

The encoder module maps spatial coordinates to feature vectors that
induce the attention kernel `G = exp(E E^T)`.

## Classes

### `Embed` (abstract base)

```python
from laker.embed import Embed
class MyEncoder(Embed):
    def forward(self, x): ...
```

`Embed` is an `nn.Module` subclass declaring abstract
`input_dim: int` and `dim: int` attributes. Subclasses must implement
`forward(x)` returning a 2-D tensor of shape `(batch, dim)`.

### `Position`

```python
from laker.embed import Position
enc = Position(input_dim=2, dim=8, num=16, sigma=10.0, seed=42)
out = enc(x)  # (n, 8)
```

Position-driven embedding: a fixed bank of `num` random Fourier
features followed by a 2-layer MLP (Tanh activation).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `input_dim` | required | Spatial dimension `d` |
| `dim` | required | Output dimension `D` |
| `num` | `2 * dim` | Number of Fourier frequencies |
| `sigma` | 10.0 | Fourier bandwidth (`freq ~ N(0, 1/σ²)`) |
| `seed` | 42 | Seed for the random Fourier bank and MLP init |
| `device` | `Backend.device` | Target device |
| `dtype` | `Backend.dtype` | Target dtype |

The MLP hidden width is `max(dim, num // 2)`. Weights are initialised
with Kaiming-uniform sampling from a *local* `torch.Generator` so
that construction is thread-safe and deterministic given the seed.

`forward` accepts both `(n, input_dim)` and `(input_dim,)` input
shapes; the latter is unsqueezed internally.

### `Visual`

```python
from laker.embed import Visual
enc = Visual(input_dim=3, dim=10, patch=4)
out = enc(x)  # x: (n, 3, H, W) -> (n, 10)
```

Patch-based visual embedding: a single `Conv2d` (kernel = stride =
`patch`) followed by spatial mean-pooling and a linear projection.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `input_dim` | 3 | Number of input channels |
| `dim` | 10 | Output dimension |
| `patch` | 4 | Patch size (also the conv stride) |
| `seed` | 42 | Seed for the conv weight init |
| `device` | `Backend.device` | Target device |
| `dtype` | `Backend.dtype` | Target dtype |

The conv output is `Conv2d(input_dim, dim, kernel=patch, stride=patch)`,
producing `(n, dim, h/patch, w/patch)`. Spatial mean collapses the last
two dims, then a `Linear(dim, dim)` projection produces the final
embedding.

## Using a custom encoder

Pass any `nn.Module` to `Laker(encoder=my_module)`:

```python
import torch.nn as nn

class MyCNN(Embed):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.conv = nn.Conv1d(input_dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

m = Laker(encoder=MyCNN(input_dim=2, dim=8))
```

`Laker` will use your encoder instead of the default `Position`.

## Why position-driven?

`Position` is the default because it satisfies two properties:

1. **Deterministic.** Given the same `seed`, the same encoder
   produces the same output. No global torch RNG is touched.
2. **Differentiable.** The Fourier features and MLP weights are
   standard `nn.Parameter` and `nn.Module` subclasses, so they
   support gradient flow for `learn` and `calibrate`.

For the synthetic data the package ships with, `Position` produces
embeddings of similar quality to learned encoders. For real data,
consider training a custom encoder via `m.learn(x, y)`.
