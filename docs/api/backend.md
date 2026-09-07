# `laker.backend` — device, dtype, compile, seed configuration

`Backend` is a single class with classmethods. State lives at the
class level (it is process-wide global state).

## Constants (class attributes)

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `Backend.device` | `cpu` | Default torch device |
| `Backend.dtype` | `float32` | Default floating-point dtype |
| `Backend.chunk` | 64 MiB | Memory budget per chunked matvec |
| `Backend.chunk_off` | False | If True, disable chunking entirely |
| `Backend.autocast_on` | False | If True, wrap kernel ops in `torch.amp.autocast` |
| `Backend.compile_mode` | `""` | If non-empty, pass to `torch.compile(mode=...)` |

## Methods

### `Backend.load()`

Re-read every `LAKER_*` env var and refresh the cached state. Call
this after mutating the environment at runtime.

```python
os.environ["LAKER_DTYPE"] = "float64"
Backend.load()  # Backend.dtype is now torch.float64
```

### `Backend.device_set(device)`

Set the default device. Pass `None` to auto-select CUDA → MPS → CPU.

```python
Backend.device_set("cpu")
Backend.device_set(torch.device("cuda"))
Backend.device_set(None)  # auto-select
```

### `Backend.dtype_set(dtype)`

Set the default floating-point dtype. Raises `ValueError` for
non-floating dtypes.

```python
Backend.dtype_set(torch.float32)
Backend.dtype_set(torch.float64)
```

### `Backend.chunk_set(megabytes)`

Override the chunk-memory budget in megabytes. Raises `ValueError`
for non-positive values. The default is 64 MiB.

```python
Backend.chunk_set(128)  # 128 MiB
```

### `Backend.tf32() -> bool`

Return `True` if TF32 matmul precision is enabled.

### `Backend.tensor(data, device=None, dtype=None) -> Tensor`

Coerce an array-like to a tensor. Pass `None` to use the current
defaults.

```python
Backend.tensor([1.0, 2.0, 3.0])                # float32, cpu
Backend.tensor([1.0, 2.0], dtype=torch.float64)  # float64
```

### `Backend.compile(func, mode="reduce-overhead")`

If `Backend.compile_mode` is set, return `torch.compile(func, mode=...)`.
Otherwise return `func` unchanged.

### `Backend.autocast()`

If `Backend.autocast_on` is set, return a `torch.amp.autocast` context
on CUDA or CPU. Otherwise return a null context.

### `Backend.seed(value)`

Seed the global torch RNG and set `LAKER_SEED`.

### `Backend.summary()`

Log a one-line description of the current backend configuration.

## Environment variables

| Variable | Effect |
|----------|--------|
| `LAKER_DEVICE` | Initial value of `Backend.device` |
| `LAKER_DTYPE` | If `"float64"`, use `torch.float64`; else `torch.float32` |
| `LAKER_CHUNK_MEMORY_BUDGET` | Initial value of `Backend.chunk` (MiB) |
| `LAKER_DISABLE_CHUNK` | If `"1"`, set `Backend.chunk_off = True` |
| `LAKER_AUTOCAST` | If `"1"`, set `Backend.autocast_on = True` |
| `LAKER_COMPILE_MODE` | If set, the `torch.compile` mode to use |
| `LAKER_TF32` | If `"0"`, disable TF32; else enable (default) |
| `LAKER_NUM_THREADS` | If set, the number of torch threads |
| `LAKER_SEED` | If set, the initial torch seed |

## Examples

```python
from laker.backend import Backend
print(Backend.device, Backend.dtype)  # cpu, torch.float32

# Force float64 globally
Backend.dtype_set(torch.float64)
Backend.seed(42)
```
