"""Embedding modules.

Public types: :class:`Embed` (abstract base), :class:`Position`,
:class:`Visual`.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from laker.backend import Backend

logger = logging.getLogger(__name__)


class Embed(nn.Module):
    """Abstract base class for all LAKER embedding modules.

    Subclasses must implement :meth:`forward` returning a 2-D tensor
    of shape ``(batch, dim)``.
    """

    dim: int
    input_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Position(Embed):
    """Deterministic position-driven embedding (Fourier features + MLP).

    Drawing Fourier frequencies and MLP initialisation from a dedicated
    local :class:`torch.Generator` keeps the module fully reproducible
    and thread-safe at construction time.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num: Optional[int] = None,
        sigma: float = 10.0,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dim = int(dim)
        self.sigma = float(sigma)
        if num is None:
            num = dim * 2
        self.num = int(num)
        if device is None:
            device = Backend.device
        if dtype is None:
            dtype = Backend.dtype

        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.register_buffer(
            "freq",
            torch.randn(
                self.input_dim,
                self.num,
                generator=gen,
                device=device,
                dtype=dtype,
            )
            / self.sigma,
        )
        self.register_buffer(
            "phase",
            torch.rand(self.num, generator=gen, device=device, dtype=dtype) * 2.0 * math.pi,
        )
        hidden = max(self.dim, self.num // 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.num, hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, self.dim, device=device, dtype=dtype),
        )
        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    bound = math.sqrt(1.0 / layer.weight.shape[1])
                    layer.weight.copy_(
                        torch.rand(
                            layer.weight.shape,
                            generator=gen,
                            device=device,
                            dtype=dtype,
                        )
                        * (2 * bound)
                        - bound
                    )
                    if layer.bias is not None:
                        layer.bias.copy_(
                            torch.rand(
                                layer.bias.shape,
                                generator=gen,
                                device=device,
                                dtype=dtype,
                            )
                            * (2 * bound)
                            - bound
                        )
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = torch.cos(2.0 * math.pi * (x @ self.freq) + self.phase)
        return self.mlp(features)

    def info(self) -> str:
        return (
            f"input_dim={self.input_dim}, dim={self.dim}, "
            f"num={self.num}, sigma={self.sigma}"
        )


class Visual(Embed):
    """Patch-based visual embedding (image-to-features).

    A small :class:`Conv2d` encoder followed by a linear projection to
    ``dim``. ``input_dim`` is the number of colour channels.
    """

    def __init__(
        self,
        input_dim: int = 3,
        dim: int = 10,
        patch: int = 4,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dim = int(dim)
        self.patch = int(patch)
        if device is None:
            device = Backend.device
        if dtype is None:
            dtype = Backend.dtype
        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.encoder = nn.Conv2d(
            input_dim,
            dim,
            kernel_size=patch,
            stride=patch,
            device=device,
            dtype=dtype,
        )
        self.proj = nn.Linear(dim, dim, device=device, dtype=dtype)
        with torch.no_grad():
            bound = math.sqrt(1.0 / self.encoder.weight.shape[1])
            self.encoder.weight.copy_(
                torch.rand(
                    self.encoder.weight.shape,
                    generator=gen,
                    device=device,
                    dtype=dtype,
                )
                * (2 * bound)
                - bound
            )
            if self.encoder.bias is not None:
                self.encoder.bias.zero_()
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.mean(dim=(-2, -1))
        return self.proj(h)

    def info(self) -> str:
        return f"input_dim={self.input_dim}, dim={self.dim}, patch={self.patch}"


__all__ = ["Embed", "Position", "Visual"]