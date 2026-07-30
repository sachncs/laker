"""Shared validation and tensor coercion utilities.

Public entry point is :class:`Base`; helpers are static methods.
"""

from __future__ import annotations

from typing import Optional

import torch


class Base:
    """Validation and tensor coercion primitives."""

    @staticmethod
    def validate_inputs(x: torch.Tensor, name: str = "x") -> torch.Tensor:
        """Validate a 2-D input tensor (``n``, ``d``).

        Args:
            x: Tensor to validate.
            name: Argument name used in error messages.

        Returns:
            The same tensor (for method chaining).

        Raises:
            ValueError: if shape is wrong or values are non-finite.
        """
        if x.dim() != 2:
            raise ValueError(f"{name} must be 2-D (n, d), got shape {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError(f"{name} must have at least one row")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        return x

    @staticmethod
    def validate_target(y, name: str = "y") -> torch.Tensor:
        """Coerce ``y`` to a 1-D tensor of shape ``(n,)``.

        Accepts either ``(n,)`` or ``(n, 1)``. Rejects 0-D scalars and
        multi-dimensional targets.

        Args:
            y: Tensor-like target.
            name: Argument name used in error messages.

        Returns:
            ``y`` reshaped to ``(n,)``.
        """
        if y.dim() == 0:
            raise ValueError(
                f"{name} must be 1-D (n,) or 2-D (n, 1), got scalar " f"shape {tuple(y.shape)}"
            )
        if y.dim() == 2:
            if y.shape[-1] != 1:
                raise ValueError(f"{name} must have shape (n,) or (n, 1), got " f"{tuple(y.shape)}")
            return y.squeeze(-1)
        if y.dim() != 1:
            raise ValueError(f"{name} must be 1-D, got shape {tuple(y.shape)}")
        if not torch.isfinite(y).all():
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        return y

    @staticmethod
    def validate_embedding_output(
        out: torch.Tensor,
        expected_dim: int,
        name: str = "encoder",
    ) -> torch.Tensor:
        """Validate the output of an embedding module.

        Checks for finite, 2-D output whose column count matches
        ``expected_dim``.

        Args:
            out: Output tensor.
            expected_dim: Expected number of columns.
            name: Argument name.

        Returns:
            ``out`` unchanged.
        """
        if out.dim() != 2:
            raise ValueError(
                f"{name} output must be 2-D (batch, embedding_dim), got "
                f"shape {tuple(out.shape)}"
            )
        if out.shape[1] != expected_dim:
            raise ValueError(
                f"{name} output has {out.shape[1]} features but model " f"expects {expected_dim}"
            )
        if not torch.isfinite(out).all():
            raise ValueError(f"{name} output contains non-finite values")
        return out

    @staticmethod
    def validate_split_indices(
        n: int,
        val_fraction: float,
    ) -> tuple[int, int]:
        """Validate split parameters.

        Returns:
            ``(n_train, n_val)`` row counts.
        """
        if n < 2:
            raise ValueError(f"n must be at least 2 for splits, got {n}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)
        return n - n_val, n_val

    @staticmethod
    def coerce_tensor(value, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Coerce ``value`` to a tensor, optionally casting dtype."""
        if isinstance(value, torch.Tensor):
            return value if dtype is None else value.to(dtype)
        return torch.as_tensor(value, dtype=dtype)

    @staticmethod
    def device_of(*tensors: torch.Tensor) -> torch.device:
        """Return the device of the first non-None tensor, or default."""
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        return torch.device("cpu")


__all__ = ["Base"]
