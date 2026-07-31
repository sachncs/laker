# `laker.data` — synthetic radio-field generation

`Data` is a single class with static methods. It generates synthetic
radio fields for benchmarks and tests when no real dataset is
available. For real spectrum cartography data, see
[External datasets](../external_datasets.md) (TODO).

## Methods

### `Data.validate(loss, ref, shadow)`

Validate path-loss model parameters. Raises `ValueError` for
non-physical values.

| Parameter | Constraint |
|-----------|-----------|
| `loss` (path-loss exponent) | `≥ 0` |
| `ref` (reference distance) | `> 0` |
| `shadow` (shadowing std) | `≥ 0` |

### `Data.field(locations, transmitters, powers, loss=2.0, ref=1.0, shadow=1.5, seed=None) -> (clean, noisy)`

Generate a synthetic radio field via the standard log-distance
path-loss model with log-normal shadowing.

```
clean(x) = Σ_j  P_j  −  10 · loss · log10( d(x, tx_j) / ref )
noisy(x)  = clean(x) + shadow · ε,  ε ~ N(0, 1)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `locations` | required | Sensor locations `(n, d)` |
| `transmitters` | required | Transmitter positions `(k, d)` |
| `powers` | required | Transmitter powers `(k,)` in dBm |
| `loss` | 2.0 | Path-loss exponent (free space = 2) |
| `ref` | 1.0 | Reference distance in metres |
| `shadow` | 1.5 | Shadowing standard deviation in dB |
| `seed` | None | Seed for the shadowing RNG |

Returns `(clean, noisy)`, each of shape `(n,)`. The shadowing uses a
`torch.Generator` seeded on `locations.device`, so the noise is
reproducible given the same seed and inputs.

The distance `d(x, tx_j)` is clamped to `ref` to avoid `log(0)` when a
sensor is closer than `ref` to a transmitter.

Validates:
- `locations.dim() == 2` and `locations.shape[0] > 0`
- `transmitters.dim() == 2` and `transmitters.shape[0] > 0`
- `powers.dim() == 1` and `powers.shape[0] > 0`
- `transmitters.shape[0] == powers.shape[0]`
- `locations.shape[1] == transmitters.shape[1]`
- `loss`, `ref`, `shadow` are physically valid (see `Data.validate`)
- `transmitters` and `powers` are coerced onto the same device/dtype
  as `locations`.

### `Data.split(n, x, y, val=0.2, seed=None) -> (x_tr, y_tr, x_va, y_va)`

Deterministic train/val split.

| Argument | Default | Meaning |
|----------|---------|---------|
| `n` | required | Total sample count |
| `x` | required | Inputs `(n, d)` |
| `y` | required | Targets `(n,)` |
| `val` | 0.2 | Validation fraction in `(0, 1)` |
| `seed` | None | Seed for the permutation; defaults to `torch.initial_seed()` |

Splits via a single `torch.randperm` seeded on `x.device` (a
`Generator` is constructed on the same device as `x` so the split is
reproducible across CPU/GPU). Returns the four split tensors
without overlap. The split is always `n_train + n_val == n`.

### `Data.grid(bounds, size, device=None, dtype=None) -> Tensor`

Generate a regular 2-D evaluation grid.

| Argument | Default | Meaning |
|----------|---------|---------|
| `bounds` | required | `(x_min, x_max, y_min, y_max)` |
| `size` | required | Points per axis |
| `device` | `Backend.device` | Target torch device |
| `dtype` | `Backend.dtype` | Target torch dtype |

Returns a tensor of shape `(size**2, 2)` with `x` varying in
`bounds[0:2]` and `y` varying in `bounds[2:4]`. Validates:
- `size >= 2`
- `x_min < x_max`
- `y_min < y_max`

## When to use

`Data.field` is the entry point for the existing examples
(`examples/*.py`) and benchmark scripts. For real spectrum
cartography data (e.g. measured radio maps), the recommendation is:
write your own `field`-like function that returns the same `(clean,
noisy)` shape, then plug into `Laker.fit` directly.

The closed-form math in `Data.field` is verified in
`tests/test_data.py::TestField`:

- Single-transmitter no-noise: matches `P − 10·loss·log10(d/ref)` to
  `atol=1e-6`.
- Distance clamping: sensor closer than `ref` is treated as `ref`.
- Shadowing: empirical std of `(noisy − clean)` matches `shadow`
  within 5 % over 20000 samples.
- dBm summation: `clean = Σ_j (clean_j)`, the documented contract
  (sum of dBm contributions, not linear-power summation).
