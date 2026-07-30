"""Runnable LAKER examples.

Real-world-style demonstrations of the public surface. Each
example ships as a single primary class in its own module, with
assertions that fail loudly if LAKER regresses. Run any example
as ``python -m examples.<name>``.

- :mod:`.learn` — ``Learn``: single-shot fit, score, save, reload
  round-trip.
- :mod:`.map` — ``Map``: scattered-sensor fit, grid-level evaluation
  against ground truth.
- :mod:`.flow` — ``Flow``: streaming updates, validated per batch.
- :mod:`.scale` — ``Scale``: thousands of sensors and a dense grid.
- :mod:`.tune` — ``Tune``: validation-driven regularisation search.
"""

__all__ = ["learn", "map", "flow", "scale", "tune"]
