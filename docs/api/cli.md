# `laker.cli` — command-line interface

`CLI` is a single class with static methods. The console entry
point declared in `pyproject.toml` is `CLI.run`.

## Methods

### `CLI.logging(verbose) -> None`

Configure the root logger level. `True` → DEBUG, `False` → INFO.
Idempotent: subsequent calls reset the root level.

### `CLI.load(path) -> Tensor`

Load a tensor from a `.pt`, `.pth`, or `.npy` file. `.npy` files
are loaded with `torch.from_numpy`; `.pt` / `.pth` files with
`torch.load(weights_only=True)`.

Raises `ValueError` for any other extension.

### `CLI.parser() -> argparse.ArgumentParser`

Build the argument parser with two subcommands:

- `fit --locations X --measurements Y --output MODEL.pt [--regularization λ] [--gamma γ] [--embedding-dim d] [--probes n] [--device cpu|cuda] [--dtype float32|float64] [--kernel exact|nystrom|fourier|neighbors|grid|spectrum|hybrid]`
- `predict --model MODEL.pt --locations X --output PREDS.pt`

The `--regularization` flag has alias `--lam`; the `--probes` flag
has alias `--num`. Both point to the same destination.

### `CLI.fit(args) -> None`

Dispatch the `fit` subcommand: load data, build `Laker`, fit, save.

### `CLI.predict(args) -> None`

Dispatch the `predict` subcommand: load model, load query locations,
predict, save predictions.

### `CLI.run(argv=None) -> int`

Run the CLI end-to-end. `argv` defaults to `sys.argv[1:]`. The
return value is intended for tests; the function calls `sys.exit`
internally with the appropriate status code.

- `0` on success
- `1` when no subcommand is given
- non-zero for argparse errors (unknown kernel name, missing
  required argument, etc.)

## Examples

Fit:

```bash
python -m laker.cli fit \
  --locations x.npy \
  --measurements y.npy \
  --output model.pt \
  --dtype float64 \
  --embed-dim 8 \
  --kernel nystrom
```

Predict:

```bash
python -m laker.cli predict \
  --model model.pt \
  --locations x_test.npy \
  --output preds.pt
```

Programmatic use (e.g. for tests):

```python
import sys
from laker.cli import CLI

with pytest.raises(SystemExit) as exc:
    CLI.run(["fit", "--locations", "x.npy", ...])
assert exc.value.code == 0
```
