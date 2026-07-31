"""Validation and tensor coercion primitives.

Single public class :class:`Check` with single-word static methods.
"""

from __future__ import annotations

from typing import Optional

import torch


class Check:
    """Validation and tensor coercion primitives."""

    @staticmethod
    def x(x: torch.Tensor, name: str = "x") -> torch.Tensor:
        """Validate a 2-D input tensor of shape ``(n, d)``.

        Returns:
            The same tensor.

        Raises:
            ValueError: shape is wrong or values are non-finite.
        """
        if x.dim() != 2:
            raise ValueError(f"{name} must be 2-D (n, d), got shape {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError(f"{name} must have at least one row")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        return x

    @staticmethod
    def y(y, name: str = "y") -> torch.Tensor:
        """Coerce ``y`` to 1-D, validating shape, dimensionality, finiteness."""
        if y.dim() == 0:
            raise ValueError(
                f"{name} must be 1-D (n,) or 2-D (n, 1), got scalar shape {tuple(y.shape)}"
            )
        if y.dim() == 2:
            if y.shape[-1] != 1:
                raise ValueError(f"{name} must have shape (n,) or (n, 1), got {tuple(y.shape)}")
            y = y.squeeze(-1)
        if y.dim() != 1:
            raise ValueError(f"{name} must be 1-D, got shape {tuple(y.shape)}")
        if y.shape[0] == 0:
            raise ValueError(f"{name} must have at least one element")
        if not torch.isfinite(y).all():
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        return y

    @staticmethod
    def embed(out: torch.Tensor, dim: int, name: str = "encoder") -> torch.Tensor:
        """Validate the output of an embedding module."""
        if out.dim() != 2:
            raise ValueError(
                f"{name} output must be 2-D (batch, dim), got shape {tuple(out.shape)}"
            )
        if out.shape[1] != dim:
            raise ValueError(f"{name} output has {out.shape[1]} features but model expects {dim}")
        if not torch.isfinite(out).all():
            raise ValueError(f"{name} output contains non-finite values")
        return out

    @staticmethod
    def split(n: int, fraction: float) -> tuple[int, int]:
        """Validate split parameters and return ``(n_train, n_val)``."""
        if n < 2:
            raise ValueError(f"n must be at least 2 for splits, got {n}")
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        n_val = max(1, int(round(n * fraction)))
        n_val = min(n_val, n - 1)
        return n - n_val, n_val

    @staticmethod
    def tensor(
        value, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Coerce ``value`` to a tensor, optionally casting dtype/device."""
        if isinstance(value, torch.Tensor):
            if dtype is None and device is None:
                return value
            return value.to(dtype=dtype, device=device)
        return torch.as_tensor(value, dtype=dtype, device=device)

    @staticmethod
    def device(*tensors: torch.Tensor) -> torch.device:
        """Return the device of the first non-None tensor."""
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        return torch.device("cpu")


__all__ = ["Check"]
