"""Embedding modules.

Public entry point is :class:`Embed` (with the abstract base),
:class:`Position` (default encoder), and :class:`Visual` (patch
embeddings). The ``Position`` class preserves the API of the prior
``laker.embeddings.PositionEmbedding`` so existing code keeps working
while consumers migrate to the new namespace.
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
    of shape ``(batch, embedding_dim)``.
    """

    embedding_dim: int
    input_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class Position(Embed):
    r"""Deterministic position-driven embedding.

    Maps ``x in R^{d_x}`` to a feature vector in ``R^{d_e}`` via a
    random Fourier feature bank followed by a deterministic MLP.
    Drawing Fourier frequencies and MLP initialisation from a
    dedicated local ``torch.Generator`` keeps the module fully
    reproducible and thread-safe at construction time.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_fourier: Optional[int] = None,
        sigma: float = 10.0,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.sigma = float(sigma)
        if num_fourier is None:
            num_fourier = embedding_dim * 2
        self.num_fourier = int(num_fourier)
        if device is None:
            device = Backend.device
        if dtype is None:
            dtype = Backend.dtype

        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.register_buffer(
            "freq",
            torch.randn(
                self.input_dim,
                self.num_fourier,
                generator=gen,
                device=device,
                dtype=dtype,
            )
            / self.sigma,
        )
        self.register_buffer(
            "phase",
            torch.rand(self.num_fourier, generator=gen, device=device, dtype=dtype) * 2.0 * math.pi,
        )
        mlp_hidden = max(self.embedding_dim, self.num_fourier // 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_fourier, mlp_hidden, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(mlp_hidden, self.embedding_dim, device=device, dtype=dtype),
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

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, embedding_dim={self.embedding_dim}, "
            f"num_fourier={self.num_fourier}, sigma={self.sigma}"
        )


class Visual(Embed):
    """Patch-based visual embedding (image-to-features).

    A small ``Conv2d`` encoder followed by a linear projection to
    ``embedding_dim``. ``input_dim`` denotes the number of colour
    channels.
    """

    def __init__(
        self,
        input_dim: int = 3,
        embedding_dim: int = 10,
        patch_size: int = 4,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.patch_size = int(patch_size)
        if device is None:
            device = Backend.device
        if dtype is None:
            dtype = Backend.dtype
        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.encoder = nn.Conv2d(
            input_dim,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            device=device,
            dtype=dtype,
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim, device=device, dtype=dtype)
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
                self.encoder.bias.copy_(
                    torch.zeros_like(self.encoder.bias, device=device, dtype=dtype)
                )
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height, width)
        h = self.encoder(x)  # (batch, embedding_dim, h', w')
        h = h.mean(dim=(-2, -1))  # (batch, embedding_dim)
        return self.proj(h)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, embedding_dim={self.embedding_dim}, "
            f"patch_size={self.patch_size}"
        )


__all__ = ["Embed", "Position", "Visual"]
