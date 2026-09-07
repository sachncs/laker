---
name: Bug report
about: Report unexpected behaviour in LAKER
title: "[BUG] "
labels: ["bug"]
assignees: ""
---

## Describe the bug

A clear and concise description of what the bug is, including the
operation that triggered it (`fit`, `predict`, `variance`,
`update`, `path`, `learn`, `tune`, ...). If you can tell whether
the bug is in the kernel, the solver, or the preconditioner, say
so.

## Minimal reproduction

A minimal script that fails on `master` with the current release.
Use `Laker` directly (not the legacy `LAKERRegressor`).

```python
import torch
from laker import Laker

torch.manual_seed(0)
x = torch.rand(20, 2, dtype=torch.float64)
y = torch.randn(20, dtype=torch.float64)

model = Laker(...)
# ... call that exhibits the bug ...
```

If the bug is observable from the CLI, also paste the
`laker fit ...` / `laker predict ...` invocation.

## Expected behaviour

What you expected to happen. Include a one-line summary of what you
observed instead (the wrong numerical value, the traceback, the
silent failure, etc.).

## Diagnostic output

If you ran with `LAKER_VERBOSE=1` or `LAKER_LOG_LEVEL=DEBUG`, paste
the relevant log lines. If you have a failing assertion, paste the
assertion output too.

## Environment

- OS: [e.g. macOS 14.4, Ubuntu 22.04, Windows 11]
- Python version: [e.g. 3.12.3]
- PyTorch version: [e.g. 2.1.2]
- CUDA version (if applicable): [e.g. 12.1]
- `laker` version: [run `python -c "import laker; print(laker.__version__)"`]

## Additional context

Anything else relevant: the exact configuration you used
(`cccp_max_iter`, `pcg_tol`, `kernel=...`), a link to a related
issue, or a screenshot of the failure mode.
