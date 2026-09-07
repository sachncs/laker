"""Runnable LAKER examples.

Real-world-style demonstrations of the public surface. Each
example ships as a single primary class in its own module, with
assertions that fail loudly if LAKER regresses. Run any example
as ``python -m examples.<name>``.

- :mod:`.simple` — ``Simple``: minimal end-to-end fit on a single-layer
  analytic target with save/load round-trip; the closest thing to a
  hello world.
- :mod:`.learn` — ``Learn``: single-shot fit on a 200-sensor urban
  radio map with assertions on shapes, dtype, and save/load
  bit-identity.
- :mod:`.map` — ``Map``: 2000-sensor fit, dense grid evaluation
  against ground truth with improvement-over-mean assertion.
- :mod:`.flow` — ``Flow``: streaming updates under the documented
  rebuild threshold with per-batch stability verification.
- :mod:`.scale` — ``Scale``: 5000-sensor fit, 900-point grid predict,
  with timed fit and predict.
- :mod:`.tune` — ``Tune``: validation-driven regularisation search
  covering a log-spaced candidate grid.
- :mod:`.scalable_data` — ``ScalableData``: download, verify, extract,
  index, clean, transform and load the real-world UCF-50K spectrum
  cartography corpus (50,000 ray-traced radio maps, ~10 GB).
- :mod:`.scalable` — ``Scalable``: end-to-end, reproducible full-sweep
  experiment on UCF-50K — every kernel configuration is benchmarked,
  the best is validated on the complete 256×256 grid, and the whole
  corpus run is resumable.
- :mod:`.paper` — ``Paper``: reproduces the LAKER paper's numerical
  experiment (arXiv:2604.25138, Section V) on the paper's synthetic
  scene — conditioning, PCG iterations vs baselines, and reconstruction
  RMSE/NMSE against a Gaussian-process baseline.
- :mod:`.cross_map_cache` — ``cross_map_cache``: one-shot build of the
  per-pixel train mean map cached under ``data/ucf50k/cross_map_mean.npy``,
  used by the cross-map (stationary cross-map prior) baseline.
- :mod:`.cross_map_score` — ``cross_map_score``: score the cached
  cross-map against every map in the corpus and write
  ``data/ucf50k/cross_map_score.csv`` (per-split RMSE summary).
"""

__all__ = [
    "simple",
    "learn",
    "map",
    "flow",
    "scale",
    "tune",
    "scalable_data",
    "scalable",
    "paper",
    "cross_map_cache",
    "cross_map_score",
]
