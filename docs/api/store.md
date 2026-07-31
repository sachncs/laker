# `laker.store` — model persistence

`Store` is a single class with two static methods. Handles serialising
a fitted `Laker` to a single `.pt` file and reconstructing it.

## Methods

### `Store.save(model, path)`

Serialise the fitted model to `path`. The file contains:

- Format version header (`format: 2`)
- All hyperparameters from `model.get_params()`
- Fitted tensors: `embed`, `coef`, `x_train`, `y_train`
- Preconditioner state (if present):
  - class name and module path
  - all tensor attributes
- Encoder state dict (if present)
- Corrector state dict (if present, with class info)

Raises `RuntimeError` if the model is not fitted.

```python
from laker import Laker
m = Laker(embed_dim=4, dtype=torch.float64)
m.fit(x, y)
m.save("model.pt")
```

### `Store.load(path) -> Laker`

Deserialise a model from `path`. Reconstructs:

1. A fresh `Laker` with the stored hyperparameters.
2. The fitted tensors (`embed`, `coef`) on the correct device.
3. The kernel operator (selected by the stored `kernel` parameter)
   built from the stored `embed`.
4. The preconditioner (via `cls.__new__` + tensor attribute
   restoration — no `build` re-run).
5. The encoder (with a fallback to `Position` if the original class
   can't be imported).
6. The corrector (if present).

```python
m2 = Laker.load("model.pt")
pred = m2.predict(x)
```

## Round-trip guarantees

Across all 7 kernel operators and corrector attachments, the model's
predictions and variances are bit-identical (within float tolerance)
to the original before save/load. Verified in
`tests/test_store.py::TestKernelRoundTrip`.

## Format compatibility

The current format version is `2`. The `load` method logs a warning
for unsupported versions but will not refuse older files unless the
version is too old to parse.

## What is NOT saved

- Live Python references (datasets, generators)
- The verbose flag is saved but the env vars are not
- The Backend global state (`device`, `dtype`, etc.) is not saved

## Files saved

The same `.pt` file format is used for all kernel types and
preconditioner types. The file size scales with:
- `O(n × d)` for the embeddings
- `O(n)` for the alpha vector
- `O(n × num_probes)` for the preconditioner (basis + eigenvalues)
- `O(n × num_neighbors)` for the sparse Neighbors kernel COO
- `O(grid_size^d)` for the Grid kernel grid
- `O(d²)` for the Spectrum kernel U factor

For typical configurations (`n=1000, d=8, num=100`), the file is
~1 MB. For `n=10000`, ~50 MB.

## Custom encoder save/load

If the encoder class isn't importable from its original module path
(e.g. defined in a notebook), `load` falls back to `Position` with a
warning. To make custom encoders work across save/load, ensure they
live in an importable module:

```python
# in my_module.py
from laker.embed import Embed

class MyEncoder(Embed):
    ...

# in your script
from my_module import MyEncoder
m = Laker(encoder=MyEncoder(...))
m.fit(x, y)
m.save("model.pt")  # round-trips correctly
```
