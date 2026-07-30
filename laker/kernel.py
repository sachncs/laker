"""Kernel operators (canonical public surface).

The canonical operator classes live here as :class:`Exact`,
:class:`Nystrom`, :class:`Fourier`, :class:`Neighbors`,
:class:`Grid`, :class:`Hybrid`, :class:`Spectrum`, and
:class:`Distribute`. They each subclass the legacy implementation in
``laker.kernels`` to avoid duplicating the matrix-free kernels while
exposing the concise module-qualified names that the public surface
promises.

The legacy module names (``AttentionKernelOperator``, ``NystromAttentionKernelOperator``,
...) remain as backward-compat aliases of the canonical classes.
"""

from __future__ import annotations

from laker.kernels import (
    AttentionKernelOperator as Exact,
    NystromAttentionKernelOperator as Nystrom,
    RandomFeatureAttentionKernelOperator as Fourier,
    SKIAttentionKernelOperator as Grid,
    SparseKNNAttentionKernelOperator as Neighbors,
    SpectralAttentionKernelOperator as Spectrum,
    TwoScaleAttentionKernelOperator as Hybrid,
)
from laker.kernels import KernelOperator  # protocol
from laker.distributed import (
    DistributedAttentionKernelOperator as Distribute,
)

__all__ = [
    "Exact",
    "Nystrom",
    "Fourier",
    "Neighbors",
    "Grid",
    "Hybrid",
    "Spectrum",
    "Distribute",
    "KernelOperator",
]
