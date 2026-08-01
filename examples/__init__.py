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
- :mod:`.paper` — ``Paper``: reproduces the LAKER paper's numerical
  experiment (arXiv:2604.25138, Section V) on the paper's synthetic
  scene — conditioning, PCG iterations vs baselines, and reconstruction
  RMSE/NMSE against a Gaussian-process baseline.
"""

__all__ = [
    "simple",
    "learn",
    "map",
    "flow",
    "scale",
    "tune",
    "paper",
]
