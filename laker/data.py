"""Synthetic data generation.

The single public type is :class:`Data`; helpers are static methods.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from laker.backend import get_default_device, get_default_dtype

logger = logging.getLogger(__name__)


class Data:
    """Synthetic data generation for spectrum cartography."""

    @staticmethod
    def validate_params(
        path_loss_exponent: float,
        reference_distance: float,
        shadow_sigma: float,
    ) -> None:
        """Validate path-loss parameters.

        Args:
            path_loss_exponent: ``eta``, must be non-negative.
            reference_distance: ``d_0``, must be positive.
            shadow_sigma: Shadowing std-dev, must be non-negative.

        Raises:
            ValueError: if any value is out of range.
        """
        if path_loss_exponent < 0:
            raise ValueError(
                f"path_loss_exponent must be non-negative, got " f"{path_loss_exponent}"
            )
        if reference_distance <= 0:
            raise ValueError(f"reference_distance must be positive, got " f"{reference_distance}")
        if shadow_sigma < 0:
            raise ValueError(f"shadow_sigma must be non-negative, got {shadow_sigma}")

    @staticmethod
    def field(
        locations: torch.Tensor,
        transmitters: torch.Tensor,
        powers: torch.Tensor,
        path_loss_exponent: float = 2.0,
        reference_distance: float = 1.0,
        shadow_sigma: float = 1.5,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Generate a synthetic radio field from multiple transmitters.

        Computes a received signal strength (RSS) field via the standard
        log-distance path-loss model with optional log-normal shadowing:

        .. math::

            r(x) = \sum_j \bigl(P_j - 10 \eta \log_{10}
            \frac{d_j}{d_0} \bigr) + \sigma_\epsilon \epsilon

        where :math:`d_j = \|x - \text{tx}_j\|_2`,
        :math:`\epsilon \sim \mathcal{N}(0, 1)`.

        Args:
            locations: Sensor locations of shape ``(n, d)``.
            transmitters: Transmitter coordinates of shape ``(k, d)``.
            powers: Transmitter power levels (dBm) of shape ``(k,)``.
            path_loss_exponent: Path-loss exponent ``eta``.
            reference_distance: Reference distance ``d_0``.
            shadow_sigma: Shadowing standard deviation.
            seed: Optional seed for reproducible shadowing.

        Returns:
            Tuple ``(rss_clean, rss_noisy)``, each of shape ``(n,)``.
        """
        Data.validate_params(path_loss_exponent, reference_distance, shadow_sigma)
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

        # Coerce transmitter and power tensors onto the same device/dtype
        # as locations so cross-tensor arithmetic works without surprise.
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
            distances = distances.clamp(min=reference_distance)
            path_loss = 10.0 * path_loss_exponent * torch.log10(distances / reference_distance)
            rss_clean += tx_power - path_loss

        noise = torch.randn(n, device=locations.device, dtype=locations.dtype, generator=gen)
        rss_noisy = rss_clean + shadow_sigma * noise
        logger.info(
            "Generated radio field: n=%d, tx=%d, path_loss_exp=%.1f, shadow_sigma=%.2f",
            n,
            transmitters.shape[0],
            path_loss_exponent,
            shadow_sigma,
        )
        return rss_clean, rss_noisy

    @staticmethod
    def grid(
        bounds: Tuple[float, float, float, float],
        grid_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Generate a regular 2-D evaluation grid.

        Args:
            bounds: ``(x_min, x_max, y_min, y_max)``.
            grid_size: Number of points per axis.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tensor of shape ``(grid_size**2, 2)``.
        """
        if grid_size < 2:
            raise ValueError(f"grid_size must be at least 2, got {grid_size}")
        x_min, x_max, y_min, y_max = bounds
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be strictly less than x_max ({x_max})")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be strictly less than y_max ({y_max})")
        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()
        x = torch.linspace(x_min, x_max, grid_size, device=device, dtype=dtype)
        y = torch.linspace(y_min, y_max, grid_size, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


__all__ = ["Data"]
