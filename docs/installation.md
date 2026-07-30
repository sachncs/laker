# Installation

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (the wheel is tested against torch 2.0 - 2.x; the
  upper bound in `pyproject.toml` is `<3.0`)
- NumPy >= 1.23 (`<3.0`)

## From PyPI

The simplest installation:

```bash
pip install laker
```

This pulls the CPU wheel. If you have a CUDA-capable system and want
GPU support, install PyTorch first using the index that matches your
CUDA version (see <https://pytorch.org/get-started/locally/>) and
then install LAKER without changing the torch version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install laker
```

## From source (editable)

For local development and tests:

```bash
git clone https://github.com/sachncs/laker.git
cd laker
pip install -e ".[dev,viz]"
```

The `[dev]` extra pulls in `pytest`, `pytest-cov`, `hypothesis`,
`black`, `isort`, `flake8`, `mypy`, `sphinx`, `sphinx-rtd-theme`,
and `myst-parser`. The `[viz]` extra adds `matplotlib` for the
plotting examples and tests.

## Optional extras

| Extra | What it adds |
| --- | --- |
| `viz` | `matplotlib>=3.5,<4.0` for plotting examples and tests. |
| `dev` | Test/lint/docs tooling (pytest, black, isort, flake8, mypy, sphinx, myst-parser, hypothesis). |

## Bumping torch / numpy

LAKER is tested on torch 2.0 and later. To upgrade torch within a
LAKER install, use `pip install --upgrade torch` without re-installing
LAKER. The upper bound in `pyproject.toml` is `<3.0`; installing
torch 3.x without first relaxing that bound will fail.

## Verifying the install

After installation:

```bash
python -c "import torch; from laker import Laker; \
    m = Laker(); x = torch.rand(20, 2); y = torch.randn(20); \
    m.fit(x, y); print(m.predict(x[:5]).shape)"
```

Expected output:

```
torch.Size([5])
```

## Troubleshooting

- **`ImportError: linalg_cuda not found`** — you installed a CPU-only
  torch but CUDA-enabled LAKER. Install the matching CUDA torch first.
- **Version conflicts with `torchvision`/`torchaudio`** — both packages
  pin torch; install them from the same PyTorch index URL.
- **`No module named 'matplotlib'`** when running examples — install
  with `pip install -e ".[viz]"`.

See [Troubleshooting](troubleshooting.md) for more.
