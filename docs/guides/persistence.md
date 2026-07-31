# Persistence and reproducibility

LAKER models can be saved to a single `.pt` file and reloaded
verbatim, including the kernel operator and preconditioner state.

## Save and load

```python
m.save("model.pt")
m2 = Laker.load("model.pt")
```

`Laker.save` writes a dictionary with the following structure:

```python
{
    "format": 2,                    # format version
    "embed_dim": 8,                 # model hyperparameters
    "lam": 0.01, "gamma": 0.1,
    "num": None, "eps": 1e-8, ...
    "kernel": "exact",
    "landmarks": None, "features": None, "neighbors": None,
    "grid_size": None, "blend": 0.5, "selection": "greedy",
    "pilot": 1000, "knots": 5,
    "prec_kind": "cccp",
    "cccp_max": 200, "cccp_tol": 1e-6, "pcg_tol": 1e-6, "pcg_max": 1000,
    "chunk": None, "encoder": None, "embed_dtype": None,
    "device": "cpu", "dtype": "torch.float64",
    "verbose": True,
    "embed": ...,                    # training embeddings tensor
    "coef": ...,                     # solution vector
    "x_train": ..., "y_train": ...,  # training data
    "preconditioner_class": "CCCP",  # class name
    "preconditioner_module": "laker.prec",
    "preconditioner_state": {...},  # all tensor attributes
    "encoder_class": "Position",
    "encoder_module": "laker.embed",
    "encoder_state": {...},
    "corrector_state": ...,          # if a corrector was attached
    "corrector_class": "Corrector",
    "corrector_module": "laker.corrector",
}
```

`Laker.load` reconstructs the model by:
1. Reading the hyperparameters and creating a fresh `Laker` with them.
2. Restoring the fitted tensors (`embed`, `coef`) on the right device.
3. Rebuilding the kernel operator from the stored `kernel` parameter
   and the restored embeddings.
4. Restoring the preconditioner via `cls.__new__` + setting tensor
   attributes (no re-running the build).
5. Loading the encoder state dict (with a fallback to `Position` if
   the original class can't be imported).
6. Loading the corrector state dict (if present).

## Round-trip guarantees

Across all 7 kernels and corrector attachments, the model's
predictions and variances are bit-identical (within float tolerance)
to the original before save/load.

Tested in `tests/test_store.py::TestKernelRoundTrip`.

## Format compatibility

The current format version is `2`. A future change to the schema will
bump this. The `load` method will not refuse older files unless
the format is unsupported.

## Reproducibility

The following sources of randomness can be made deterministic:

| Source | How to control |
|--------|----------------|
| Encoder init | `Position(seed=42)` (default) |
| RFF features | `Fourier` uses `seed=42` internally |
| CCCP probes | `CCCP.build(op, n, seed=...)` (passed via `Core.build_prec`) |
| Train/val split | `Laker.search(..., seed=42)`, `Laker.correct(..., seed=42)`, etc. |
| Adam optimisers | Set torch global RNG (`torch.manual_seed(42)`) before any optimiser step |

For a fully reproducible fit, set:
```python
import torch
from laker import Laker

torch.manual_seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)

m = Laker(embed_dim=8, dtype=torch.float64)
m.fit(x, y, seed=42)
```

## What is NOT in the saved file

- The `verbose` flag is saved; environment variables like
  `LAKER_DEVICE` / `LAKER_DTYPE` are not. Reloading with a
  different `Backend.dtype` will preserve the *model's* dtype but the
  `Backend.dtype` global remains whatever it was set to.

- Live Python references to datasets or generators are not saved.
  Save the *result* of `Data.field(...)` to `.npy` if you want to
  reproduce the exact data.
