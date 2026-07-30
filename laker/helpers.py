"""Math and RNG helpers.

Public entry point is :class:`Helpers`; utilities are exposed as
``@staticmethod``. Helpers wrap shared numerical primitives used across
kernel, solver, and pipeline code.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


class Helpers:
    """Numerical and RNG helpers used across the package."""

    @staticmethod
    def safe_exp(gram: torch.Tensor, skip_clamp: bool = False) -> torch.Tensor:
        """Element-wise exp with dtype-aware overflow guard.

        Clamps ``gram`` to a dtype-aware maximum before exponentiation
        to prevent silent overflow to ``inf``. The returned tensor is
        always exponentiated; callers must consume the return value.

        Args:
            gram: Tensor to exponentiate.
            skip_clamp: If ``True``, skip the clamp step (caller must
                guarantee finite, in-range inputs).

        Returns:
            Element-wise ``exp(gram)``, possibly clamped.
        """
        return _safe_exp(gram, skip_clamp=skip_clamp)

    @staticmethod
    def normal_pdf(x):
        """Standard normal PDF using torch ops."""
        return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def normal_cdf(x):
        """Abramowitz/Stegun approximation of the standard normal CDF."""
        return _normal_cdf_approx(x)

    @staticmethod
    def sinh(x):
        """Stable ``sinh`` for large arguments."""
        return torch.sinh(x)

    @staticmethod
    def trace_normalize(mat: torch.Tensor) -> torch.Tensor:
        """Normalise a positive-definite matrix to ``trace = n``."""
        n = mat.shape[0]
        trace = torch.trace(mat)
        if torch.abs(trace) < 1e-30:
            return mat
        return mat * (n / trace)

    @staticmethod
    def lh_sample(n: int, dims: int, seed: Optional[int] = None) -> np.ndarray:
        """Latin-hypercube sample of shape ``(n, dims)`` in ``[0, 1]``."""
        rng = np.random.default_rng(seed)
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.uniform(0.0, 1.0, size=(dims, n))
        return (cut[:-1] + u * (1.0 / n)).T

    @staticmethod
    def power_iteration(operator, n: int, num_iter: int = 10, seed: int = 0) -> float:
        """Estimate the spectral norm of a linear operator."""
        return _power_iteration(operator, n, num_iter=num_iter, seed=seed)

    @staticmethod
    def safe_chol(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Cholesky factor with diagonal jitter fallback."""
        try:
            return torch.linalg.cholesky(matrix)
        except RuntimeError:
            n = matrix.shape[0]
            jitter = eps * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
            return torch.linalg.cholesky(matrix + jitter)

    @staticmethod
    def kernel_dtype_bytes(dtype: torch.dtype) -> int:
        """Bytes per scalar for memory budget calculations."""
        if hasattr(dtype, "itemsize"):
            return dtype.itemsize
        return 4

    @staticmethod
    def safe_inv_diag(diagonal: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Invert a diagonal, clamping small/negative entries to ``eps``."""
        return 1.0 / diagonal.clamp(min=eps)

    @staticmethod
    def default_seed() -> Optional[int]:
        """Return the global LAKER seed if set."""
        import os

        return int(os.environ["LAKER_SEED"]) if os.environ.get("LAKER_SEED") else None

    @staticmethod
    def set_global_seed(seed: int) -> None:
        """Seed torch, numpy, and the ``LAKER_SEED`` environment variable."""
        import os

        torch.manual_seed(int(seed))
        if hasattr(np.random, "default_rng"):
            np.random.default_rng(int(seed))
        os.environ["LAKER_SEED"] = str(int(seed))

    @staticmethod
    def scatter_to_dense(sparse: torch.Tensor, shape) -> torch.Tensor:
        """Materialise a sparse tensor to dense, if needed."""
        if sparse.is_sparse:
            return sparse.to_dense()
        return sparse


# ---------------------------------------------------------------------------
# Module-internal implementations (private; not part of the public API).
# ---------------------------------------------------------------------------
def _safe_exp(gram: torch.Tensor, skip_clamp: bool = False) -> torch.Tensor:
    """Element-wise exp with dtype-aware overflow guard."""
    if not skip_clamp:
        if gram.dtype == torch.float16:
            max_val = 11.0
        elif gram.dtype == torch.float32:
            max_val = 80.0
        elif gram.dtype == torch.bfloat16:
            max_val = 80.0
        else:
            max_val = 700.0
        if gram.requires_grad:
            return torch.exp(gram.clamp(max=max_val))
        out = gram.clone()
        out.clamp_(max=max_val)
        return torch.exp(out, out=out)
    if gram.requires_grad:
        return torch.exp(gram)
    return torch.exp(gram, out=gram)


def _normal_cdf_approx(x):
    """Abramowitz & Stegun 7.1.26 approximation of the standard normal CDF."""
    sign = torch.where(x >= 0, 1.0, -1.0)
    abs_x = torch.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * abs_x)
    poly = (
        t
        * (
            0.319381530
            + t
            * (
                -0.356563782
                + t
                * (
                    1.781477937
                    + t * (-1.821255978 + t * 1.330274429)
                )
            )
        )
    )
    pdf = torch.exp(-0.5 * abs_x**2) / math.sqrt(2.0 * math.pi)
    return 1.0 - sign * pdf * poly


def _power_iteration(operator, n: int, num_iter: int = 10, seed: int = 0) -> float:
    """Estimate the spectral norm of a linear operator via power iteration."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    v = v / v.norm()
    norm = 0.0
    for _ in range(num_iter):
        u = operator(v)
        norm = u.norm().item()
        if norm == 0.0:
            return 0.0
        v = u / norm
    return norm


__all__ = ["Helpers"]
