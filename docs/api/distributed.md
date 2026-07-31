# `laker.distributed` — multi-GPU wrapper

`Distributed` wraps an `Exact` kernel operator for sharded execution
across multiple CUDA devices. Falls back to single-device execution
when only one GPU is available (or none).

## Usage

```python
from laker.distributed import Distributed
import torch

if torch.cuda.device_count() >= 2:
    op = Distributed(
        embeddings,
        lam=0.01,
        master=torch.device("cuda:0"),
    )
    op.matvec(x)  # shards embeddings across GPUs, gathers results
```

## Constructor

```python
Distributed(
    embeddings: Tensor,
    lam: float = 1e-2,
    master: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `embeddings` | required | Full embedding matrix `(n, embed_dim)` |
| `lam` | 1e-2 | Ridge weight |
| `master` | `embeddings.device` | Device where input/output vectors live |
| `dtype` | `embeddings.dtype` | Floating-point dtype |

When `torch.cuda.is_available()` is false or only one GPU is
detected, the constructor builds a single inner `Exact` operator on
the master device and sets `self.single = True`. Otherwise it shards
the embeddings as evenly as possible across the available CUDA
devices and sets `self.single = False`.

## Attributes

| Attribute | Meaning |
|-----------|---------|
| `self.n` | Operator dimension |
| `self.dim` | Embedding dimension |
| `self.lam` | Ridge weight |
| `self.master` | Master device for input/output |
| `self.device` | Same as `master` |
| `self.dtype` | Floating-point dtype |
| `self.shape` | `(n, n)` |
| `self.skip` | Always `True` (overflow clamp is safe for sharded ops) |
| `self.devices` | List of CUDA devices (or `[master]` if single) |
| `self.single` | Whether running single-device (no sharding) |
| `self.sizes` | Per-shard row counts (only multi-device) |
| `self.ops` | Per-shard `Exact` operators (only multi-device) |

## Methods

### `Distributed.matvec(x)`

Apply the operator to vector(s) `x`. In single-device mode, delegates
to the inner `Exact.matvec`. In multi-device mode:

1. Move `x` to the master device.
2. Gather all per-device embedding shards into a full matrix on
   master.
3. For each device, copy the full embeddings and the input slice to
   that device, then compute the local matvec chunk.
4. Concatenate the per-device output chunks and return on master.

The `chunk` size for the local matvec is hard-coded to 8192 (chosen
empirically to keep per-block memory under `~64 MiB` at `float64`).

### `Distributed.diag()`

Return the diagonal of `λ I + G`. In single-device mode, delegates
to the inner `Exact.diag`. In multi-device mode, each shard computes
its local diagonal and the results are concatenated.

### `Distributed.dense()`

Materialise the full `n × n` matrix on the master device. Single
device delegates; multi-device gathers all embeddings to master and
computes `exp(E E^T) + λ I` in one shot.

### `Distributed.eval(x, y=None, chunk=None)`

Evaluate the kernel between queries and training points. In
single-device mode, delegates. In multi-device mode, gathers
embeddings and uses `torch.exp(x @ gathered.T)`.

## Limitations

- The wrapper does not actually reduce memory or compute; it gathers
  the full embedding matrix to each device for every matvec. This
  is "data parallelism" (parallelise the matvec output chunks), not
  true model parallelism.
- True model parallelism (all-reduce over partial kernel
  contributions) is not implemented. See `README.md` section 8.

For most purposes the CCCP preconditioner makes single-device PCG
fast enough that `Distributed` is not needed.
