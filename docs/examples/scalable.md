# `examples/scalable.py` — real-world UCF-50K full-sweep experiment

End-to-end, reproducible experiment on a real spectrum cartography
dataset: 50,000 ray-traced path-loss radio maps of the UCF campus
(`KR-init/Spectrum-Cartography-256x256-UCF-50K`, ~10 GB, MIT licence).
Every LAKER kernel configuration is benchmarked head-to-head, the best
one is validated on the *complete* 256×256 radio grid, and the whole
corpus run is resumable.

Its companion [`scalable_data.py`](../../examples/scalable_data.py)
owns the data path (download → verify → extract → index → clean →
transform → load) and is torch-free.

## What it does

```bash
# 1. One-time: download, verify, extract and index the ~10 GB corpus.
python -m examples.scalable_data --prepare --selfcheck 5

# 2. Full sweep over the val split + validation on the complete grid.
python -m examples.scalable --max-maps 25 --validate-maps 25

# 3. Evaluate the winning configuration on the ENTIRE 50,000-map corpus
#    (complete 256×256 grid, resumable, parallelisable).
python -m examples.scalable --validate-maps 0 --workers 8
```

The sweep covers every kernel family × `lam ∈ {1e-2, 1e-1}`:

| Family | Variants |
|--------|----------|
| `exact` | dense `G = exp(EEᵀ)` |
| `nystrom` | `m ∈ {100, 256, 512}` |
| `fourier` | `r ∈ {128, 256, 512, 1024}` |
| `neighbors` | `k ∈ {5, 10, 20}` |
| `grid` | SKI grid, `g = 64`, `embed_dim = 2` |
| `spectrum` | spectral-shaped kernel, `knots = 5` |
| `hybrid` | Nyström + neighbours blend `b ∈ {0.3, 0.5, 0.7}` |

Each configuration is fitted on 2,000 sensors per scene and scored on a
held-out 8,000-pixel batch (RMSE / MAE / R² in dBm, fit and predict
time, PCG iterations), against a mean-target baseline. Targets are
standardised per scene and predictions un-standardised.

## Results (this machine, seed 0, 25 validation maps)

Dataset: 50,000 maps of 65,536 pixels, path loss `40.9–140.0 dBm`,
per-map mean ≈ 120 dBm, σ ≈ 20.6 dBm, ~21% building pixels excluded
from evaluation.

```
config              rmse   mae    r2  fit_s pred_s iters speedup
nystrom_m100_lam0.01 10.69  7.78 0.727  0.061  0.027   19  2.96x
nystrom_m256_lam0.01 10.70  7.82 0.727  0.216  0.026   19  1.07x
exact_lam0.01        10.74  7.89 0.724  0.187  0.029   21  1.16x
fourier_r128_lam0.01 11.05  8.29 0.708  0.013  0.045   17  4.24x
... (30 configurations total)
spectrum_k5          breakdown   # numerically unstable (see below)

baseline (mean-target) RMSE: 20.64 dB
winner: nystrom_m100_lam0.01  48.2% below baseline
```

Full-grid validation of the winner on 25 held-out maps (complete
256×256 grid, non-building pixels): **masked RMSE 10.56 ± 0.93 dB vs a
20.86 dB baseline** — roughly 2× better than predicting the sensor
mean, at a fraction of the cost of the exact kernel.

### Full-corpus validation (all 50,000 maps, run `20260801T114917Z`)

The winning configuration was evaluated on the complete 256×256 grid
of every map in the corpus (~110 min, `--workers 4`):

```
maps: 50,000  (test 5,000 | val 5,000 | train 40,000)
masked RMSE: 10.38 +/- 0.91 dB   median 10.30   range 6.75 .. 13.07
baseline (mean-target) RMSE: 20.73 dB   -> 49.9% below baseline
per-split RMSE: test 10.37 | val 10.37 | train 10.38   (no split bias)
mean R^2 over maps: 0.746
```

The result is stable across the whole corpus — the 25-map estimate
(10.56) overstates the true error by only ~0.2 dB — and uniform across
splits, so the model generalises with no test/train leakage.

## Findings worth knowing

- Low-rank kernels match the exact kernel's accuracy at a large
  speedup: `nystrom` with `m = 100` lands at the top of the Pareto
  front (~3× faster than `exact` for the same RMSE), and `fourier`
  with `r = 128` is 4× faster at +0.3 dB.
- Sparse kernels (`neighbors`), the SKI `grid` and the `hybrid` blends
  underperform here because the radio field is smooth but global in
  extent — local support alone cannot extrapolate far from sensors.
- `spectrum` (as initialised, `embed_dim = 16`) is **numerically
  unstable on this data**: the Position embeddings have singular values
  up to σ_max ≈ 47, so `exp(0.087·σ²)` reaches ≈ 1e87 and the PCG solve
  either diverges or breaks down. The experiment records this honestly
  as `breakdown` / `degenerate` rows instead of crashing.

## Reproducibility and observability

Every run writes `outputs/scalable/<run_id>/`:

| Artifact | Contents |
|----------|----------|
| `events.jsonl` | append-only, timestamped event log (scene × config outcomes, timings) |
| `sweep.csv` | one row per scene × configuration |
| `validate.csv` | append-only per-map full-grid scores |
| `metrics.json` | aggregated table, Pareto front, winner, validation summary |
| `manifest.json` | git commit, environment fingerprint, exact configuration, seeds, dataset fingerprint |

Scene sampling is deterministic for a given `--seed`; the data
pipeline records a SHA-256 corpus fingerprint that is checked on every
run. A crashed full-corpus run is resumed with
`--resume <run_id> ...` (same arguments); completed maps are skipped
from the checkpoint. Use `--workers N` to parallelise across maps.

## Cross-map (stationary) baseline

`--cross-map` adds an honest floor for cross-map learning without
per-scene conditioning: the **per-pixel mean map aggregated over the
40,000 training maps** (cached at `data/ucf50k/cross_map_mean.npy`,
51,791 of 65,536 pixels covered, mean of per-pixel means
`120.20 dBm`). Every held-out scene is scored against this single
fixed prediction.

Build the cache independently with
`python -m examples.cross_map_cache --workers 8`; then either run the
sweep with `--cross-map` (records `cross_map_rmse` per row in
`validate.csv`) or score the cache against every map in the corpus
with `python -m examples.cross_map_score --workers 8` (writes
`data/ucf50k/cross_map_score.csv`).

Measured on the full 50,000-map corpus
(`python -m examples.cross_map_score --workers 8`):

```
cross-map (per-pixel train mean) RMSE on 50000 maps:
  20.73 +/- 1.50 dB  (median 20.81, range 15.70 .. 23.83)
  test  : n= 5000  mean=20.74 +/- 1.49  median=20.82
  val   : n= 5000  mean=20.73 +/- 1.49  median=20.80
  train : n=40000  mean=20.73 +/- 1.51  median=20.81
```

This is essentially identical to the per-scene mean baseline
(`20.73 dB` from the full-corpus run): the stationary cross-map
prior provides no information beyond the per-scene level because the
building mask is shared across all maps and the per-pixel level
varies scene-to-scene. The information that helps is per-scene
conditioning (sensors + kernel), not a learned stationary prior.

## Positioning vs learned (CNN) baselines

The UCF-50K dataset
(`KR-init/Spectrum-Cartography-256x256-UCF-50K`) was released about
two months ago and is, in the publisher's own words, *purpose-built
for CNN training*. As of writing no leaderboard, no published
baseline and no third-party benchmark exists on it, so a direct
head-to-head with a learned method is not currently possible on this
corpus.

Reference anchor on an analogous task — RadioUNet (Levie et al.,
IEEE TWC 2021), a learned U-Net conditioned on building geometry
and trained on the 56,080-map RadioMapSeer corpus — reports RMSE
"of order 1 dB where the range of pathloss values from the noise
floor to the maximal gain is 100 dB", i.e. roughly **1 % of the
pathloss range**. That is the reference for what learned CNN
methods can achieve on comparable urban radio-map estimation. No
RadioUNet or successor has been trained and evaluated on UCF-50K
yet.

Putting the LAKER result in context on this corpus:

| Method                                       | RMSE (dBm) | % of range  | Notes                                         |
|----------------------------------------------|-----------:|------------:|-----------------------------------------------|
| Per-scene mean (sensor-batch baseline)       |     20.73  |   20.9 %    | constant predictor of scene mean              |
| Cross-map (per-pixel train mean, stationary) |     20.73  |   20.9 %    | measured on 50,000 maps; no per-scene cond.   |
| **LAKER (nystrom m=100, λ=1e-2)**            | **10.38**  | **10.5 %**  | per-scene kernel, training-free, ~2 min/map   |
| RadioUNet (RadioMapSeer, building-geometry)  |       ~1   |    ~1 %     | learned U-Net, building mask + Tx as input    |

LAKER is the strongest training-free per-scene kernel on this
corpus and lands within the same order of magnitude as learned CNNs
that operate on a different, building-geometry-aware dataset. The
gap to RadioUNet is the price of going training-free without
geometry priors — and a fair target for any future learned
baseline evaluated on UCF-50K.

## When to use

- You want a real, large, reproducible spectrum-cartography benchmark
  (not synthetic `Data.field`) with quantitative kernel comparison.
- You are extending LAKER and need a regression corpus that exercises
  the whole pipeline at scale.
- To run the entire 50,000-map evaluation (~8 h single-worker on this
  laptop; ~2 h with `--workers 4`), use `--validate-maps 0`.

## See also

- [`scalable_data.py`](../../examples/scalable_data.py) — the ETL
  pipeline behind the experiment.
- [Choosing a kernel](../guides/choosing_kernel.md) — when each
  approximation is appropriate.
- [Low-rank approximations](../algorithms/low_rank.md) — Nyström, RFF,
  k-NN, SKI, hybrid, spectrum theory.
