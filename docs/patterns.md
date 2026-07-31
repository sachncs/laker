# Design Patterns

This document describes the architectural patterns used throughout
the LAKER project. All new code should follow these conventions.

---

## 1. Module-Class Convention

### Rule: one primary class per module

Every module under `laker/` exports exactly one public class. The
class owns its primary logic as instance or static methods. Helper
functions are exposed only as `@staticmethod` on that class — never
as module-level public functions.

### Naming convention

- The file name and the class name are unprefixed. The module name
  and class name need not match exactly; one-word file names with
  one-word class names are preferred.
- All identifiers are `snake_case`; classes are `PascalCase`.
- No leading-underscore modules (e.g. `_foo.py`) and no leading-underscore
  classes (e.g. `_Foo`). Private helpers inside a class use a single
  underscore (`_helper(self, ...)`).

### Pattern

```python
# laker/helpers.py
class Helpers:
    """Math and RNG helpers."""

    @staticmethod
    def safe_exp(gram, skip_clamp=False):
        ...

    @staticmethod
    def trace_normalize(mat):
        ...
```

### Why we use it

- **Discoverability.** A new user can write `dir(Helpers)` and see
  every public entry point.
- **One decision point per file.** Renaming or deprecating a helper
  is a class-level change with a single migration site.
- **Reduced namespace pollution.** No risk of helper-function name
  collisions between modules.

### Where helpers live

If a helper does not fit the module's primary class, promote it to
its own module. For example, `Backend` owns environment settings;
`Helpers` owns numerical primitives. Helpers never leak as
module-level functions.

---

## 2. The `Laker` facade and module-qualified secondaries

The package's top-level entry point is the `Laker` class. Secondary
classes (`Exact`, `Nystrom`, `CCCP`, ...) are reachable under their
module names. No name is duplicated across the public surface.

```python
from laker import Laker
from laker.kernels import (
    NystromAttention as Nystrom,
    RandomFeatureAttention as Fourier,
    SparseAttention as Neighbors,
    SKIAttention as Grid,
    TwoScaleAttention as Hybrid,
    SpectralAttention as Spectrum,
    Attention as Exact,
)
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

### Why this surface

- **Single primary class.** One type (`Laker`) covers 90% of use
  cases.
- **Discoverable secondaries.** A user who wants to subclass
  `Nystrom` or replace `CCCP` finds them under `laker.kernels.NystromAttention`
  and `laker.preconditioner.CCCP` without grepping.
- **No global functions.** Even `Data.field(...)` is exposed as
  `Data.field(...)`, not `generate_radio_field(...)`.

---

## 3. Logging over printing

Diagnostic output uses Python's standard `logging` module:

```python
import logging
logger = logging.getLogger(__name__)

def my_pipeline(...):
    logger.info("loading data n=%d", n)
```

Documentation code blocks demonstrate `logger.info(...)`, not
`print(...)`. `print()` in library code, examples, and benchmarks is
forbidden; `print()` in CLI scripts and tests is allowed for human
output.

---

## 4. Dtype / device policy

- `torch.float32` is the default floating dtype for the solver.
- `torch.float64` is recommended for very-high-condition-number
  problems ($\kappa \gtrsim 10^{10}$).
- The default device is `cpu`. CUDA auto-detects when available.
- Both dtype and device are coerced at the `Backend.to_tensor`
  boundary; downstream code can rely on type consistency.
- `Laker.predict(x)` returns predictions on the same device as
  `x`; callers are responsible for moving results back if needed.

---

## 5. Test invariants

Tests are written as small, focused assertions. The most important
patterns:

```python
def test_thing():
    # 1. Data sanity (shape / dtype / finiteness).
    assert x.shape == (...)
    assert torch.isfinite(x).all()

    # 2. Mathematical precision against an analytical reference.
    torch.testing.assert_close(actual, expected, atol=..., rtol=...)

    # 3. Behavioural contract.
    assert result.score(x, y) > 0.9
```

Where a kernel cannot satisfy a stronger invariant (the audit
flagged that Nyström `matvec` differs from `to_dense @ x`) the test
explicitly `pytest.skip(...)` with the documented relative error
recorded, rather than fabricating a permissive 100%-error tolerance.

---

## 6. Renaming and naming hygiene

Every rename happens in lock-step with code. There is no
deprecation layer; breaking changes are accepted at the package
boundary. New renames are recorded in `CONTRIBUTING.md` under the
"Module Conventions" section.

The renaming policy:

1. Pick the new name. Prefer one-word names.
2. Add the new public class to its module and re-export from the
   public package surface.
3. Update all callers in the same commit.
4. Update tests, examples, and docs in the same commit.

---

## 7. Where exceptions are raised

- `ValueError` for invalid constructor arguments (range, dtype).
- `ValueError` for invalid input data (shape, dtype, finiteness).
- `RuntimeError` for convergence failures that the user can recover
  from by adjusting tolerances (`pcg_tol`, `cccp_max_iter`).
- `RuntimeError` for documented behaviour that requires user action
  (e.g. `partial_fit` rebuild threshold exceeded).
- `FileNotFoundError` for missing model files on `load`.
- `KeyError` for missing mandatory fields on load (raised explicitly;
  no silent defaults).

Errors carry a message that includes the offending argument name
and its value where appropriate.
