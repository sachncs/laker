# Performance

LAKER's runtime is dominated by the matrix-free matvecs that PCG
performs; everything else is a small additive overhead. The
default configuration reaches near size-independent convergence on
problems with $\kappa(\lambda I + G) \lesssim 10^{8}$.

## Scaling overview

| $n$ | Wall-clock on Apple M3 (CPU) | Memory (float64) | Wall-clock on A100 |
| --- | --- | --- | --- |
| 100 | < 0.1 s | < 1 MB | < 0.01 s |
| 1,000 | < 0.5 s | < 50 MB | < 0.05 s |
| 10,000 | 5–30 s | ~ 0.5 GB | < 1 s |
| 100,000 | 1–10 min | ~ 5 GB | 5–60 s |

The wall-clock spreads depend on the condition number of the kernel
and the chosen approximation strategy.

## Tuning knobs

The constructor accepts a small number of meaningful knobs. The
defaults are tuned for the radio-map reconstruction setting of
the original paper.

### Solving harder problems

| Symptom | Likely cause | Recommended change |
| --- | --- | --- |
| PCG takes hundreds of iterations | Under-sampled probe budget | Increase `probes` (default auto $= \max(200, 2\sqrt{n})$). |
| Memory blow-up on $n \geq 10^4$ | Default chunk size is too large | Set `chunk_size` explicitly to a value that keeps per-matvec memory under your budget. |
| Tiny `coef_` values, numerical instability | Single precision | Switch to `dtype=torch.float64`. |
| Preconditioner build is the bottleneck | You set `probes` very large | Reduce `probes`; the preconditioner becomes approximate but PCG converges in more iterations. |

### Memory

The full kernel matrix is never materialised; matvecs are tiled
across `chunk_size` rows. The auto-selected chunk size is

```python
chunk_size_local = max(1024, min(n // 10, 8192))
```

which keeps peak memory around `O(n \cdot chunk\_size)` per matvec.

For very-large problems, the `kernel="nystrom"` and
`kernel="fourier"` strategies reduce memory by an order of
magnitude by replacing the dense $O(n^2)$ storage with a rank-$r$
factorisation.

### Dtype

`torch.float32` is the default. For ill-conditioned problems
($\kappa \gtrsim 10^{10}$) use `dtype=torch.float64`. The
`embedding_dtype` parameter lets the encoder run in a different
dtype than the solver; for example, float16 embeddings with
float32 PCG.

### Approximations

| Strategy | Per-matvec cost | When to use it |
| --- | --- | --- |
| `kernel="exact"` | $O(n \, d_e)$ per chunk | Default; exact. |
| `kernel="nystrom"` | $O(n \, m)$ | $m \sim 200$–$1000$ landmarks. |
| `kernel="fourier"` | $O(n \, r)$ | Fast on CPU; $r \sim 1000$–$2000$ features. |
| `kernel="neighbors"` | $O(n \, k)$ sparse | When the data has local structure. |
| `kernel="grid"` | $O(n \, g)$ | Small $d_e$; grid size $g \sim 32^2$–$64^2$. |
| `kernel="spectrum"` | SVD-shaped | When the eigenvalues are noisy and a smooth spectrum helps. |
| `kernel="hybrid"` | Nyström + k-NN | When both global and local structure matter. |

The audit found that Nyström `matvec` and Fourier `matvec` differ
from `to_dense @ x`. The exact kernel does not have this gap. When
precision matters use `kernel="exact"`.

### Device

CUDA gives a ~5–10× speedup over CPU for the matvec-heavy
workload. Set `device="cuda"` (auto-detect) or `device="cuda:0"`.

### Batching

The internal PCG accepts 2-D RHS of shape `(n, k)`. `predict(x)` for
`x` of shape `(n, k)` will run a single batched solve. No manual
batching is required.

---

## Memory accounting

| Quantity | Storage |
| --- | --- |
| Training embeddings $E$ | $O(n \, d_e)$ |
| Kernel operator (exact) | None, computed on demand. |
| Nyström factor $A = K_{nm} K_{mm}^{-1}$ | $O(n \, m)$. |
| Fourier features $\Phi$ | $O(n \, r)$. |
| Preconditioner tensors | $O(n \, N_r) + O(N_r^2)$. |
| Fitted `coef_` | $O(n)$. |

---

## Profiling

For a quick profile of a fit, wrap the call in `time.perf_counter()`
and call `model.embeddings_` and `model.preconditioner_` to see how
much of the cost is spent building the preconditioner vs running PCG.

```python
import time
import torch
from laker import Laker

x = torch.rand(2000, 2) * 10.0
y = torch.sin(x[:, 0])
m = Laker(embedding_dim=12, dtype=torch.float64)

t0 = time.perf_counter()
m.fit(x, y)
t_total = time.perf_counter() - t0

print(f"fit time = {t_total:.2f}s")
```

For fine-grained profiling, wrap the individual stages: PCG is in
`m._legacy.core.solve_pcg` and the preconditioner build is in
`m._legacy.core.build_preconditioner`.

---

## Known limits

- Nyström and Fourier kernels are not bit-equivalent to the dense
  kernel's `matvec`. Use exact kernel for tightest precision.
- Streaming updates truncate after `rebuild_threshold` rows by
  triggering a full refit. Above ~10k incremental rows the rebuild
  cost dominates; refit instead.
- Spectral kernel uses `torch.linalg.svd` on the embedding Gram;
  $O(d_e^3)$ preprocessing per `fit`. Not a hot-path concern.

See [Troubleshooting](troubleshooting.md) for diagnostic steps
when convergence or precision degrades.
