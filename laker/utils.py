"""Backward-compat shim. New code should use ``laker.helpers.Helpers``.

This module preserves the legacy free-function names plus the legacy
``GPSurrogate`` class so existing tests continue to work after the
move to :class:`Helpers`.
"""
from __future__ import annotations

# Re-export legacy free-function names from the private implementation.
from laker._utils_impl import (
    _trace_normalize as trace_normalize,
    _adaptive_shrinkage_rho as adaptive_shrinkage_rho,
    _eigh_stable as eigh_stable,
    _scipy_norm_pdf as scipy_norm_pdf,
    _scipy_norm_cdf as scipy_norm_cdf,
    GPSurrogate,
)


__all__ = [
    "trace_normalize",
    "adaptive_shrinkage_rho",
    "eigh_stable",
    "scipy_norm_pdf",
    "scipy_norm_cdf",
    "GPSurrogate",
]
