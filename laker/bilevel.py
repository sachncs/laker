"""Bilevel hyperparameter learning via implicit differentiation.

Outer-loop Adam optimiser over hyperparameters with an inner PCG solve.
Hypergradients are computed by the adjoint method
(:mod:`laker.implicit`).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

import torch

from laker.implicit import hypergradient

if TYPE_CHECKING:
    from laker.core import Core
    from laker.model import Laker

logger = logging.getLogger(__name__)


class Bilevel:
    """Bilevel optimiser: outer Adam over hyperparameters, inner PCG solve."""

    def __init__(
        self,
        core: "Core",
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
        tol: float = 1e-6,
        max_iter: int = 500,
        verbose: bool = True,
    ) -> None:
        self.core = core
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def bilevel(
        self,
        model: "Laker",
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        params: Optional[List[torch.Tensor]] = None,
    ) -> "Laker":
        """Optimise hyperparameters via bilevel learning.

        Args:
            model: The :class:`Laker` to optimise.
            x_train, y_train: Training data.
            x_val, y_val: Validation data for outer-loss evaluation.
            params: Hyperparameters to optimise. Defaults to a learnable
                logit for ``lam``.
        """
        from laker.check import Check

        x_train = Check.x(Check.tensor(x_train, device=model.device, dtype=model.dtype))
        y_train = Check.y(Check.tensor(y_train, device=model.device, dtype=model.dtype))
        x_val = Check.x(Check.tensor(x_val, device=model.device, dtype=model.dtype))
        y_val = Check.y(Check.tensor(y_val, device=model.device, dtype=model.dtype))

        if params is None:
            lam_logit = torch.tensor(
                [math.log(model.lam)],
                device=self.core.device,
                dtype=self.core.dtype,
                requires_grad=True,
            )
            params = [lam_logit]

        opt = torch.optim.Adam(params, lr=self.lr)
        best_val = float("inf")
        patience_count = 0

        for epoch in range(self.epochs):
            opt.zero_grad()

            if len(params) == 1:
                lam_logit = params[0]
                with torch.no_grad():
                    cand_lam = float(torch.exp(lam_logit).item())
                cand_lam = max(cand_lam, 1e-8)
                self.core.lam = cand_lam

            embed, enc = self.core.embed(x_train)
            model.encoder = enc

            hyper_ids = {id(p) for p in params}
            for p in enc.parameters():
                if id(p) in hyper_ids:
                    p.requires_grad = True

            kernel = self.core.build_kernel(embed)
            prec = self.core.build_prec(
                kernel.matvec,
                embed.shape[0],
                diag=kernel.diag(),
            )
            alpha, _ = self.core.solve(kernel, prec, y_train)
            alpha_d = alpha.detach()

            with torch.no_grad():
                val_embed, _ = self.core.embed(x_val)
            k_val = kernel.eval(val_embed, embed)
            y_pred = k_val @ alpha_d
            val_loss = torch.mean((y_pred - y_val) ** 2)

            n_val = y_val.shape[0]
            resid = y_pred - y_val
            dl = (2.0 / n_val) * (k_val.T @ resid)

            hgs = hypergradient(
                op=kernel.matvec,
                prec=prec.apply,
                alpha=alpha_d,
                dl=dl,
                params=params,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=False,
            )

            for p, hg in zip(params, hgs):
                if p.grad is None:
                    p.grad = hg
                else:
                    p.grad.add_(hg)
            opt.step()

            val_loss_item = val_loss.item()
            if self.verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    "Bilevel epoch %d/%d, val_loss=%.4e",
                    epoch + 1,
                    self.epochs,
                    val_loss_item,
                )
            if val_loss_item < best_val:
                best_val = val_loss_item
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    if self.verbose:
                        logger.info(
                            "Bilevel early stopping at epoch %d", epoch + 1
                        )
                    break

        if self.verbose:
            logger.info("Bilevel complete. Refitting on full training set.")

        if len(params) == 1:
            with torch.no_grad():
                cand = float(torch.exp(params[0]).item())
            self.core.lam = max(min(cand, 1e3), 1e-8)
            model.lam = self.core.lam
        return model.fit(x_train, y_train)


__all__ = ["Bilevel"]