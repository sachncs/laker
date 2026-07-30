"""Runnable LAKER examples.

Each example ships as a single class in its own module:

- :mod:`.basic` — ``PaperExample`` reproduces the Section IV-E n=3 worked
  example; verifies alpha against the direct solve and the kernel
  against ``exp(E E^T)``.
- :mod:`.large` — ``LargeScale`` fits a 5000-sample radio-field dataset
  with chunked matrix-free evaluation.
- :mod:`.radio_field` — ``RadioField`` fits a synthetic 2-D field and
  reports train R² plus grid prediction shape.
- :mod:`.bilevel` — ``Bilevel`` optimises regularisation and the
  encoder jointly via :meth:`laker.Laker.tune`.
- :mod:`.distributed_matvec` — ``Distributed`` compares single-device
  and multi-device kernel operators (CUDA only).

Run any example as ``python -m examples.<name>``.
"""

__all__ = ["basic", "large", "radio_field", "bilevel", "distributed_matvec"]
