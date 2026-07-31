# `laker.check` — input validation and tensor coercion

`Check` is a single class with static methods. All methods return the
validated tensor (or value) so they can be chained.

## Methods

### `Check.x(x, name="x") -> Tensor`

Validate a 2-D input tensor of shape `(n, d)`. Checks:
- `x.dim() == 2`
- `x.shape[0] > 0`
- all values are finite (no NaN, no Inf)

Raises `ValueError` with a message naming `name`. Returns `x` unchanged.

```python
x = Check.x(torch.randn(10, 3))  # OK
Check.x(torch.randn(10))        # ValueError: "x must be 2-D (n, d)"
Check.x(torch.randn(0, 3))     # ValueError: "x must have at least one row"
```

### `Check.y(y, name="y") -> Tensor`

Coerce and validate `y` to a 1-D tensor. Accepts:
- `(n,)` 1-D tensor
- `(n, 1)` 2-D tensor (last dim squeezed)

Checks:
- `y.dim() in {1, 2}`
- if 2-D, `y.shape[-1] == 1`
- all values are finite

Returns squeezed 1-D tensor.

```python
Check.y(torch.randn(10))       # OK
Check.y(torch.randn(10, 1))    # OK, squeezed to (10,)
Check.y(torch.tensor(1.0))     # ValueError: "y must be 1-D (n,) or 2-D (n, 1)"
Check.y(torch.randn(5, 2))     # ValueError: "(n, 1)"
```

### `Check.embed(out, dim, name="encoder") -> Tensor`

Validate the output of an embedding module. Checks:
- `out.dim() == 2`
- `out.shape[1] == dim`
- all values are finite

```python
out = Check.embed(encoder(x), dim=8)  # OK
```

### `Check.split(n, fraction) -> (n_train, n_val)`

Validate split parameters and return integer row counts.

```python
n_tr, n_va = Check.split(100, 0.2)  # (80, 20)
Check.split(10, 0.99)              # (1, 9) — clamped
Check.split(1, 0.5)                # ValueError: "at least 2"
Check.split(10, 0.0)               # ValueError
```

### `Check.tensor(value, dtype=None, device=None) -> Tensor`

Coerce to a tensor, optionally casting dtype / device. For numpy
arrays and Python lists, uses `torch.as_tensor` with the given kwargs.
For existing tensors, calls `.to(dtype, device)`.

```python
Check.tensor([1.0, 2.0])
Check.tensor([1.0, 2.0], dtype=torch.float64)
Check.tensor(some_torch_tensor, device="cpu")
```

### `Check.device(*tensors) -> torch.device`

Return the device of the first non-None tensor. Defaults to `cpu`.

```python
t = torch.zeros(3, device="cpu")
Check.device(t)         # device(type='cpu')
Check.device()           # device(type='cpu')
Check.device(None, t)    # device(type='cpu') (skips None)
```

## Usage pattern

`Check.x` and `Check.y` are called by every `Laker` entry point that
takes input data (`fit`, `predict`, `variance`, `update`, `learn`,
`correct`, `calibrate`, `bilevel`, `score`). They ensure:

- shapes are correct
- samples are non-empty
- values are finite
- dtype/device are propagated correctly

If a check fails, a `ValueError` is raised with a message that
identifies the offending argument. There is no silent fall-through
to `inf` or `nan` values.
