# CLI

The `laker` command-line entry point is defined in `pyproject.toml`
under `[project.scripts]` and implemented in `laker.cli.CLI`. Run
`python -m laker <subcommand> [flags]` or, after `pip install laker`,
the installed `laker` script directly.

```text
laker
├── fit           fit a model to training data
└── predict       run inference on a fitted model
```

`laker --help` and `laker <subcommand> --help` print the flag list.

---

## `laker fit`

Fit a Laker model to training data and serialise the result.

```text
laker fit [--locations FILE] --measurements FILE --output FILE
          [--regularization FLOAT] [--gamma FLOAT]
          [--embedding-dim INT] [--probes INT]
          [--cccp-max-iter INT] [--cccp-tol FLOAT]
          [--pcg-tol FLOAT] [--pcg-max-iter INT]
          [--device STR] [--dtype STR]
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--locations` | path | required | Path to a `.pt`/`.pth`/`.npy` tensor of locations `(n, d)`. |
| `--measurements` | path | required | Path to a `.pt`/`.pth`/`.npy` tensor of targets `(n,)`. |
| `--output` | path | required | Output path for the saved model (`.pt`). |
| `--regularization` | float | `1e-2` | Tikhonov $\lambda$. |
| `--gamma` | float | `0.1` | CCCP shrinkage. |
| `--embedding-dim` | int | `10` | Embedding MLP width. |
| `--probes` | int | auto | Random probe count. |
| `--cccp-max-iter` | int | `200` | CCCP outer iterations. |
| `--cccp-tol` | float | `1e-6` | CCCP tolerance. |
| `--pcg-tol` | float | `1e-6` | PCG relative-residual tolerance. |
| `--pcg-max-iter` | int | `1000` | PCG iteration cap. |
| `--device` | string | `cpu` | `cpu`, `cuda`, `cuda:0`, etc. |
| `--dtype` | string | `float64` | `float32` or `float64`. |

The legacy flag names `--lambda-reg` and `--num-probes` are accepted
as aliases and translated automatically.

### Example

```bash
python -m laker fit --locations x_train.pt --measurements y_train.pt --output model.pt
```

---

## `laker predict`

Run inference on a fitted model.

```text
laker predict --model FILE --locations FILE --output FILE
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | path | required | Path to the saved model (`.pt`). |
| `--locations` | path | required | Path to a `.pt`/`.pth`/`.npy` tensor of query locations. |
| `--output` | path | required | Output path for predictions. |

### Example

```bash
python -m laker predict --model model.pt --locations x_test.pt --output y_pred.pt
```

The predictions are saved as a `torch.Tensor` of shape `(n_queries,)`
on CPU. The tensor's dtype matches the model's `dtype`.

---

## File format

Both `fit` and `predict` accept `.pt`/`.pth` (PyTorch) and `.npy`
(NumPy) files. The format is inferred from the file extension. On
load the model checks the `format_version` field (currently `2`).

---

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Bad arguments or `RuntimeError`. |
| `2` | File-not-found or file-format error. |

Use `--verbose` to print extra diagnostic information. Add
`LAKER_VERBOSE=0` to the environment to silence the CLI logger.

---

## Examples

### End-to-end smoke run

```bash
python -c "
import torch
torch.manual_seed(0)
torch.save(torch.rand(50, 2) * 10.0, '/tmp/x.pt')
torch.save(torch.randn(50), '/tmp/y.pt')
"
python -m laker fit --locations /tmp/x.pt --measurements /tmp/y.pt --output /tmp/m.pt
python -c "
import torch
torch.manual_seed(0)
torch.save(torch.rand(5, 2) * 10.0, '/tmp/q.pt')
"
python -m laker predict --model /tmp/m.pt --locations /tmp/q.pt --output /tmp/p.pt
python -c "import torch; print(torch.load('/tmp/p.pt'))"
```

The resulting `Laker` is loaded through `Laker.load` and produces a
1-D tensor of length 5.
