"""End-to-end embedding training, residual correction, and bilevel learning.

Public class :class:`Trainer` orchestrates all training strategies:
embedding optimisation, residual correction, bilevel learning, and
uncertainty-aware training.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

from laker.bilevel import Bilevel
from laker.check import Check
from laker.corrector import Corrector
from laker.data import Data

if TYPE_CHECKING:
    from laker.core import Core
    from laker.model import Laker

logger = logging.getLogger(__name__)


class Trainer:
    """Coordinates embedding and residual-corrector training."""

    def __init__(self, core: "Core") -> None:
        self.core = core

    def learn(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 50,
        rebuild: int = 10,
        patience: int = 5,
    ) -> "Laker":
        """Optimise embedding weights end-to-end on the regression objective."""
        x = Check.x(x)
        y = Check.y(y)
        if model.encoder_ is None:
            raise RuntimeError("learn requires an encoder. Call fit() first or pass encoder.")

        enc = model.encoder_
        trainable = [p for p in enc.parameters() if p.requires_grad]
        if not trainable:
            for p in enc.parameters():
                p.requires_grad = True
            trainable = list(enc.parameters())
            if not trainable:
                raise RuntimeError("Encoder has no trainable parameters.")

        opt = torch.optim.Adam(trainable, lr=lr)
        best = float("inf")
        patience_count = 0
        best_state = None
        best_prec = None
        best_kernel = None

        for epoch in range(epochs):
            opt.zero_grad()
            embed = enc(x.to(dtype=self.core.embed_dtype))
            if self.core.embed_dtype != self.core.dtype:
                embed = embed.to(dtype=self.core.dtype)

            kernel = self.core.build_kernel(embed)

            if epoch % rebuild == 0 or model.coef_ is None:
                with torch.no_grad():
                    kernel_d = self.core.build_kernel(embed.detach())
                    prec = self.core.build_prec(
                        kernel_d.matvec,
                        embed.shape[0],
                        diag=kernel_d.diag(),
                    )
                    alpha, _ = self.core.solve(kernel_d, prec, y)
                model.prec_ = prec
            else:
                alpha = model.coef_.detach()

            residual = kernel.matvec(alpha) - y
            loss = 0.5 * torch.dot(residual, residual)

            loss.backward()
            opt.step()

            loss_v = loss.item()
            if self.core.verbose and (epoch + 1) % 10 == 0:
                logger.info("learn epoch %d/%d loss=%.4e", epoch + 1, epochs, loss_v)

            if loss_v < best:
                best = loss_v
                best_state = (embed.detach().clone(), alpha.detach().clone())
                best_prec = prec
                best_kernel = kernel
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    if self.core.verbose:
                        logger.info("learn early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            embed, alpha = best_state
            prec = best_prec
            kernel = best_kernel
        else:
            prec = model.prec_
            embed = embed.detach()

        model.embed_ = embed
        model.kernel_ = kernel
        model.prec_ = prec
        model.coef_, model.iters_ = self.core.solve(kernel, prec, y)
        return model

    def correct(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        val: float = 0.2,
        epochs: int = 200,
        patience: int = 10,
        weight_decay: float = 1e-2,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> "Laker":
        """Train a residual corrector on ``y - y_hat``."""
        from laker.check import Check

        if model.coef_ is None or model.embed_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before correct().")

        x = Check.x(x)
        y = Check.y(y)

        x_tr, y_tr, x_va, y_va = Data.split(
            x.shape[0],
            x,
            y,
            val=val,
            seed=seed,
        )

        with torch.no_grad():
            y_tr_base = model.predict(x_tr).detach()
            y_va_base = model.predict(x_va).detach()

        res_tr = y_tr - y_tr_base
        res_va = y_va - y_va_base

        input_dim = x.shape[1]
        if model.corrector is None:
            model.corrector = Corrector(
                input_dim=input_dim,
                output_dim=1,
                hidden_dim=32,
                dropout=0.1,
            ).to(device=self.core.device, dtype=self.core.dtype)
        else:
            model.corrector = model.corrector.to(self.core.device)

        opt = torch.optim.Adam(
            model.corrector.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        best_val = float("inf")
        patience_count = 0
        best_state = None

        last_epoch = 0
        for epoch in range(epochs):
            last_epoch = epoch
            model.corrector.train()
            opt.zero_grad()
            pred = model.corrector(x_tr).squeeze()
            loss = torch.mean((pred - res_tr) ** 2)
            loss.backward()
            opt.step()

            with torch.no_grad():
                model.corrector.eval()
                val_pred = model.corrector(x_va).squeeze()
                val_loss = torch.mean((val_pred - res_va) ** 2).item()

            if val_loss < best_val:
                best_val = val_loss
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in model.corrector.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= patience:
                    if self.core.verbose:
                        logger.info(
                            "corrector early stopping at epoch %d (val=%.4e)",
                            epoch + 1,
                            val_loss,
                        )
                    break

        if best_state is not None:
            model.corrector.load_state_dict(best_state)
        if self.core.verbose:
            logger.info("corrector fitted: epochs=%d, val=%.4e", last_epoch + 1, best_val)
        return model

    def bilevel(
        self,
        model: "Laker",
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
    ) -> "Laker":
        """Optimise hyperparameters via bilevel learning."""
        opt = Bilevel(
            core=self.core,
            lr=lr,
            epochs=epochs,
            patience=patience,
            verbose=self.core.verbose,
        )
        return opt.bilevel(model, x_train, y_train, x_val, y_val)

    def calibrate(
        self,
        model: "Laker",
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 50,
        beta: float = 0.1,
        subset: float = 0.2,
        patience: int = 5,
        seed: Optional[int] = None,
    ) -> "Laker":
        """Uncertainty-aware training: NLL + calibration penalty."""
        from laker.check import Check

        x = Check.x(x)
        y = Check.y(y)

        if model.encoder_ is None:
            raise RuntimeError("calibrate requires an encoder. Call fit() first.")
        enc = model.encoder_
        trainable = [p for p in enc.parameters() if p.requires_grad]
        if not trainable:
            for p in enc.parameters():
                p.requires_grad = True
            trainable = list(enc.parameters())
            if not trainable:
                raise RuntimeError("Encoder has no trainable parameters.")

        opt = torch.optim.Adam(trainable, lr=lr)
        n = x.shape[0]
        n_subset = max(1, int(n * subset))
        best = float("inf")
        patience_count = 0

        for epoch in range(epochs):
            opt.zero_grad()
            embed = enc(x.to(dtype=self.core.embed_dtype))
            if self.core.embed_dtype != self.core.dtype:
                embed = embed.to(dtype=self.core.dtype)

            kernel = self.core.build_kernel(embed)
            with torch.no_grad():
                kernel_d = self.core.build_kernel(embed.detach())
                prec = self.core.build_prec(kernel_d.matvec, n, diag=kernel_d.diag())
                alpha, _ = self.core.solve(kernel_d, prec, y)

            mu = self.core.predict_train(x, enc, embed, kernel, alpha, model.corrector)
            gen = torch.Generator(device=x.device)
            if seed is not None:
                gen.manual_seed(int(seed))
            perm = torch.randperm(n, generator=gen, device=x.device)
            sub = perm[:n_subset]
            x_sub = x[sub]
            var = self.core.predict_var_train(x_sub, enc, embed, kernel, prec, alpha, self.core.lam)
            residual = y[sub] - mu[sub]
            nll = 0.5 * torch.mean(torch.log(2.0 * math.pi * var) + (residual**2) / var)
            calibration = (torch.mean(residual**2) - torch.mean(var)) ** 2
            loss = nll + beta * calibration

            loss.backward()
            opt.step()

            loss_v = loss.item()
            if self.core.verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "calibrate epoch %d/%d loss=%.4e nll=%.4e cal=%.4e",
                    epoch + 1,
                    epochs,
                    loss_v,
                    nll.item(),
                    calibration.item(),
                )
            if loss_v < best:
                best = loss_v
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    if self.core.verbose:
                        logger.info("calibrate early stopping at epoch %d", epoch + 1)
                    break

        model.embed_ = embed.detach()
        kernel_op = self.core.build_kernel(model.embed_)
        model.kernel_ = kernel_op
        prec = self.core.build_prec(kernel_op.matvec, n, diag=kernel_op.diag())
        model.prec_ = prec
        model.coef_, model.iters_ = self.core.solve(kernel_op, prec, y)
        return model


__all__ = ["Trainer"]
