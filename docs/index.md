# LAKER Documentation

**LAKER** (Learning-based Attention Kernel Regression) is a production-ready
Python package for large-scale spectrum cartography and radio map
reconstruction.

> **Note:** This repository is an independent implementation of the LAKER
> algorithm. The author of this package is not one of the paper's authors.

It solves the regularised attention kernel regression problem

$$
\min_{\alpha \in \mathbb{R}^n} \; \| G \alpha - y \|_2^2 + \lambda \, \alpha^\top G \alpha
$$

where $G = \exp(E E^\top)$ is an exponential attention kernel induced by
learned embeddings $E \in \mathbb{R}^{n \times d_e}$. The key innovation is a
**learned data-dependent preconditioner** obtained via a
shrinkage-regularised Convex-Concave Procedure (CCCP), which reduces
condition numbers by up to three orders of magnitude and enables near
size-independent Preconditioned Conjugate Gradient (PCG) convergence.

Based on [*Accelerating Regularized Attention Kernel Regression for Spectrum
Cartography*](https://arxiv.org/html/2604.25138v1) (Tao & Tan, 2026).

---

## Quick Start

```python
import torch
from laker import Laker

x = torch.rand(1000, 2) * 100.0
y = torch.sin(x[:, 0] / 10.0) + torch.cos(x[:, 1] / 7.0)

model = Laker(regularization=1e-2)
model.fit(x, y)

preds = model.predict(x[:5])
var = model.variance(x[:5])
score = model.score(x, y)
```

That's the entire public surface. See [Quick Start](quickstart.md) for
the full parameter list and CLI usage.

---

## Documentation Map

| Document | What's in it |
| --- | --- |
| [Quick Start](quickstart.md) | Installation, basic usage, all Laker parameters, CLI |
| [Theory](theory.md) | Problem formulation, CCCP preconditioner, kernel approximations |
| [API Reference](api.md) | All public classes, methods, and the env-var table |
| [Examples](examples.md) | Real-world plug-and-play snippets |
| [Design Patterns](patterns.md) | Module-class convention, naming, logging |
| [Migration](migration.md) | Mapping from old `LAKERRegressor` to the `Laker` facade |
| [CLI](cli.md) | The `laker` command-line subcommands and flags |
| [Performance](performance.md) | Scaling, memory, and dtype guidance |
| [Troubleshooting](troubleshooting.md) | Common errors and diagnostic steps |
| [Installation](installation.md) | PyPI install, source install, dev install, optional extras |
| [Release](release.md) | The release process and version policy |
| [License](license.md) | MIT license summary |

---

## Public surface at a glance

The single top-level class is :class:`laker.Laker`. Secondary classes
are available under their module names:

```python
from laker import Laker
from laker.kernel import Exact, Nystrom, Fourier, Neighbors, Grid, Hybrid, Spectrum
from laker.preconditioner import CCCP, Adaptive, Jacobi
from laker.solve import PCG, Descent
from laker.embed import Position, Visual
from laker.plot import Plot
from laker.data import Data
from laker.helpers import Helpers
from laker.backend import Backend
from laker.base import Base
from laker.cli import CLI
```

Every class is a single primary class per module, with helpers exposed
as `@staticmethod` on that class. There are no module-level public
functions. The convention is enforced by `CONTRIBUTING.md` and verified
by the test suite.

---

## Where to start

1. New to LAKER? Begin with [Quick Start](quickstart.md).
2. Curious about the math? Read [Theory](theory.md).
3. Porting code from the old `LAKERRegressor` API? See
   [Migration](migration.md).
4. Hitting a wall? See [Troubleshooting](troubleshooting.md).
</content>
