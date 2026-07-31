"""Synthetic data generation for spectrum cartography.

Single public class :class:`Data` with single-word static methods.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from laker.backend import Backend

logger = logging.getLogger(__name__)


class Data:
    """Synthetic data generation for spectrum cartography."""

    @staticmethod
    def validate(
        loss: float,
        ref: float,
        shadow: float,
    ) -> None:
        """Validate path-loss parameters.

        Args:
            loss: path-loss exponent, must be non-negative.
            ref: reference distance, must be positive.
            shadow: shadowing std-dev, must be non-negative.
        """
        if loss < 0:
            raise ValueError(f"loss must be non-negative, got {loss}")
        if ref <= 0:
            raise ValueError(f"ref must be positive, got {ref}")
        if shadow < 0:
            raise ValueError(f"shadow must be non-negative, got {shadow}")

    @staticmethod
    def field(
        locations: torch.Tensor,
        transmitters: torch.Tensor,
        powers: torch.Tensor,
        loss: float = 2.0,
        ref: float = 1.0,
        shadow: float = 1.5,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a synthetic radio field from multiple transmitters.

        Computes received signal strength (RSS) via the log-distance
        path-loss model with optional log-normal shadowing:

        .. math::

            r(x) = \\sum_j \\bigl(P_j - 10 \\eta \\log_{10} d_j/d_0\\bigr)
            + \\sigma_\\epsilon \\epsilon

        Returns:
            ``(rss_clean, rss_noisy)``, each of shape ``(n,)``.
        """
        Data.validate(loss, ref, shadow)
        if locations.dim() != 2:
            raise ValueError(f"locations must be 2-D, got shape {locations.shape}")
        if locations.shape[0] == 0:
            raise ValueError("locations must have at least one row")
        if transmitters.dim() != 2:
            raise ValueError(f"transmitters must be 2-D, got shape {transmitters.shape}")
        if transmitters.shape[0] == 0:
            raise ValueError("transmitters must have at least one row")
        if powers.dim() != 1:
            raise ValueError(f"powers must be 1-D, got shape {powers.shape}")
        if powers.shape[0] == 0:
            raise ValueError("powers must have at least one element")
        if transmitters.shape[0] != powers.shape[0]:
            raise ValueError(
                "transmitters and powers must have same length, "
                f"got {transmitters.shape[0]} and {powers.shape[0]}"
            )
        if locations.shape[1] != transmitters.shape[1]:
            raise ValueError(
                "locations and transmitters must have same spatial dimension, "
                f"got {locations.shape[1]} and {transmitters.shape[1]}"
            )

        transmitters = transmitters.to(device=locations.device, dtype=locations.dtype)
        powers = powers.to(device=locations.device, dtype=locations.dtype)

        if seed is not None:
            gen = torch.Generator(device=locations.device)
            gen.manual_seed(seed)
        else:
            gen = None

        n = locations.shape[0]
        rss_clean = torch.zeros(n, device=locations.device, dtype=locations.dtype)
        for tx_location, tx_power in zip(transmitters, powers):
            distances = torch.norm(locations - tx_location, dim=1)
            distances = distances.clamp(min=ref)
            path_loss = 10.0 * loss * torch.log10(distances / ref)
            rss_clean += tx_power - path_loss

        noise = torch.randn(n, device=locations.device, dtype=locations.dtype, generator=gen)
        rss_noisy = rss_clean + shadow * noise
        logger.info(
            "Generated radio field: n=%d, tx=%d, loss=%.1f, shadow=%.2f",
            n,
            transmitters.shape[0],
            loss,
            shadow,
        )
        return rss_clean, rss_noisy

    @staticmethod
    def split(
        n: int,
        x: torch.Tensor,
        y: torch.Tensor,
        val: float = 0.2,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic train/val split.

        Args:
            n: number of samples.
            x: inputs of shape ``(n, d)``.
            y: targets of shape ``(n,)``.
            val: validation fraction in ``(0, 1)``.
            seed: optional seed (otherwise uses ``Backend.seed``).
        """
        from laker.check import Check

        n_train, n_val = Check.split(n, val)
        gen = torch.Generator(device=x.device)
        seed_val = int(seed) if seed is not None else int(torch.initial_seed())
        gen.manual_seed(seed_val)
        perm = torch.randperm(n, generator=gen, device=x.device)
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

    @staticmethod
    def grid(
        bounds: Tuple[float, float, float, float],
        size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Generate a regular 2-D evaluation grid.

        Args:
            bounds: ``(x_min, x_max, y_min, y_max)``.
            size: number of points per axis.
            device: target device.
            dtype: target dtype.

        Returns:
            Tensor of shape ``(size**2, 2)``.
        """
        if size < 2:
            raise ValueError(f"size must be at least 2, got {size}")
        x_min, x_max, y_min, y_max = bounds
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be strictly less than x_max ({x_max})")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be strictly less than y_max ({y_max})")
        if device is None:
            device = Backend.device
        if dtype is None:
            dtype = Backend.dtype
        x = torch.linspace(x_min, x_max, size, device=device, dtype=dtype)
        y = torch.linspace(y_min, y_max, size, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


__all__ = ["Data"]
