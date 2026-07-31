"""Streaming updates, regularisation paths, and continuation schedules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from laker.core import Core
    from laker.model import Laker

logger = logging.getLogger(__name__)


class Stream:
    """Incremental updates and continuation-path fitting for a fitted model."""

    def __init__(self, core: "Core") -> None:
        self.core = core

    def update(
        self,
        model: "Laker",
        x_new: torch.Tensor,
        y_new: torch.Tensor,
        forget: float = 1.0,
        threshold: int = 100,
        seed: Optional[int] = None,
    ) -> "Laker":
        """Append new observations and re-solve with a warm start.

        Args:
            model: Fitted :class:`Laker`.
            x_new: New inputs of shape ``(m, d)``.
            y_new: New targets of shape ``(m,)``.
            forget: Scalar in ``[0, 1]`` scaling the previous ``alpha``.
            threshold: Max cumulative new points before forcing a refit.
            seed: Optional seed for the preconditioner's random probes.
        """
        from laker.check import Check

        if model.coef_ is None or model.embed_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before update().")

        x_new = Check.x(Check.tensor(x_new, device=model.device, dtype=model.dtype), "x_new")
        y_new = Check.y(Check.tensor(y_new, device=model.device, dtype=model.dtype), "y_new")

        m = x_new.shape[0]
        total = getattr(model, "_partial_count", 0) + m

        if total >= threshold:
            model._partial_count = 0
            raise RuntimeError(
                "update threshold exceeded. Concatenate all data and call fit() for a full refit."
            )

        model._partial_count = total

        new_emb = model.encoder_(x_new.to(dtype=self.core.embed_dtype))
        if self.core.embed_dtype != self.core.dtype:
            new_emb = new_emb.to(dtype=self.core.dtype)

        old_n = model.embed_.shape[0]
        model.embed_ = torch.cat([model.embed_, new_emb], dim=0)

        kernel = self.core.build_kernel(model.embed_, lam=model.lam)
        model.kernel_ = kernel

        old_alpha = model.coef_ * forget
        y_old = getattr(model, "_y_train", None)
        if y_old is None:
            y_old = torch.zeros(old_n, device=self.core.device, dtype=self.core.dtype)
        y_ext = torch.cat([y_old, y_new])

        x0 = torch.cat(
            [old_alpha, torch.zeros(m, device=self.core.device, dtype=self.core.dtype)]
        )

        with torch.no_grad():
            prec = self.core.build_prec(
                kernel.matvec,
                model.embed_.shape[0],
                diag=kernel.diag(),
                seed=seed,
            )
            model.prec_ = prec
            model.coef_, model.iters_ = self.core.solve(kernel, prec, y_ext, x0=x0)
        model._y_train = y_ext
        model._x_train = (
            torch.cat([getattr(model, "_x_train", x_new[:0]), x_new], dim=0)
            if getattr(model, "_x_train", None) is not None
            else x_new
        )

        if self.core.verbose:
            logger.info(
                "update: added %d points, total=%d, iters=%d",
                m,
                model.embed_.shape[0],
                model.iters_,
            )
        return model

    def path(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        grid: list[float],
        reuse: bool = True,
    ) -> dict:
        """Fit a regularisation path over a sequence of ``lam`` values.

        Each solve warm-starts from the previous one, sorted from
        largest to smallest ``lam``.

        Returns a dict with ``"lam"``, ``"coef"``, ``"iters"``, ``"rel"``.
        """
        from laker.check import Check

        x = Check.x(Check.tensor(x, device=model.device, dtype=model.dtype))
        y = Check.y(Check.tensor(y, device=model.device, dtype=model.dtype))
        if not grid:
            raise ValueError("grid must not be empty")

        embed, enc = self.core.embed(x)
        n = embed.shape[0]
        sorted_grid = sorted(grid, reverse=True)

        coefs, iters_list, rels = [], [], []
        x0 = None
        prec = None

        for lam_value in sorted_grid:
            kernel = self.core.build_kernel(embed, lam=lam_value)
            if prec is None or not reuse:
                prec = self.core.build_prec(
                    kernel.matvec,
                    n,
                    gamma=self.core.gamma,
                    num=self.core.num,
                    diag=kernel.diag(),
                )
            coef, it = self.core.solve(kernel, prec, y, x0=x0)
            coefs.append(coef)
            iters_list.append(it)
            rels.append(
                torch.linalg.norm(kernel.matvec(coef) - y).item() / torch.linalg.norm(y).item()
            )
            x0 = coef.clone()

        model.embed_ = embed
        model.encoder_ = enc
        kernel = self.core.build_kernel(embed, lam=sorted_grid[-1])
        model.kernel_ = kernel
        model.prec_ = self.core.build_prec(
            kernel.matvec,
            n,
            gamma=self.core.gamma,
            num=self.core.num,
            diag=kernel.diag(),
        )
        model.coef_ = coefs[-1]
        model.iters = iters_list[-1]
        model._y_train = y
        path = {"lam": sorted_grid, "coef": coefs, "iters": iters_list, "rel": rels}
        model._path = path
        return path

    def continuation(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        stages: int = 5,
        reuse: bool = True,
    ) -> "Laker":
        """Geometric schedule of ``lam`` values from ``hi`` down to ``lo``."""
        if hi is None:
            hi = 10.0 * self.core.lam
        if lo is None:
            lo = self.core.lam
        if stages < 1:
            raise ValueError(f"stages must be positive, got {stages}")
        if hi <= 0 or lo <= 0:
            raise ValueError("lo and hi must be positive")

        ratio = (lo / hi) ** (1.0 / max(1, stages - 1))
        schedule = [hi * (ratio**k) for k in range(stages)]
        schedule[-1] = lo

        path = self.path(model, x, y, grid=schedule, reuse=reuse)
        model.lam = float(lo)
        model.iters = path["iters"][-1]
        return model


__all__ = ["Stream"]