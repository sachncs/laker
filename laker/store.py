"""Model persistence: save and load LAKER models.

Public class :class:`Store` with two static methods :meth:`save` and
:meth:`load`. The serialized format is a single dictionary written with
:func:`torch.save` containing all hyperparameters, fitted tensors, and
neural-network state dicts (encoder and corrector).
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from laker.model import Laker

logger = logging.getLogger(__name__)


class Store:
    """Save and load LAKER models."""

    @staticmethod
    def save(model: "Laker", path: str) -> None:
        """Serialise a fitted model to ``path``."""
        if model.coef_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before save().")
        state: dict[str, Any] = {
            "format": 2,
            "embed_dim": model.embed_dim,
            "lam": model.lam,
            "gamma": model.gamma,
            "num": model.num,
            "eps": model.eps,
            "base": model.base,
            "cccp_max": model.cccp_max,
            "cccp_tol": model.cccp_tol,
            "pcg_tol": model.pcg_tol,
            "pcg_max": model.pcg_max,
            "chunk": model.chunk,
            "kernel": model.kernel,
            "landmarks": model.landmarks,
            "features": model.features,
            "neighbors": model.neighbors,
            "grid_size": model.grid_size,
            "distributed": model.distributed,
            "blend": getattr(model, "blend", 0.5),
            "selection": getattr(model, "selection", "greedy"),
            "pilot": getattr(model, "pilot", 1000),
            "knots": getattr(model, "knots", 5),
            "prec_kind": getattr(model, "prec_kind", "cccp"),
            "device": (str(model.device) if model.device is not None else "cpu"),
            "dtype": str(model.dtype),
            "embed_dtype": (
                str(model.embed_dtype) if model.embed_dtype else None
            ),
            "verbose": model.verbose,
            "embed": model.embed_.cpu() if model.embed_ is not None else None,
            "coef": model.coef_.cpu() if model.coef_ is not None else None,
            "x_train": (model._x_train.cpu() if getattr(model, "_x_train", None) is not None else None),
            "y_train": (model._y_train.cpu() if getattr(model, "_y_train", None) is not None else None),
        }
        if model.prec_ is not None:
            prec = model.prec_
            state["prec_class"] = prec.__class__.__name__
            state["prec_module"] = prec.__class__.__module__
            state["prec_state"] = {k: v for k, v in vars(prec).items() if torch.is_tensor(v)}
        if model.encoder_ is not None:
            state["encoder_state"] = model.encoder_.state_dict()
            state["encoder_class"] = model.encoder_.__class__.__name__
            state["encoder_module"] = model.encoder_.__class__.__module__
            if hasattr(model.encoder_, "input_dim"):
                state["input_dim"] = model.encoder_.input_dim
        if model.corrector is not None:
            state["corrector_state"] = model.corrector.state_dict()
            state["corrector_class"] = model.corrector.__class__.__name__
            state["corrector_module"] = model.corrector.__class__.__module__
        torch.save(state, path)

    @staticmethod
    def load(path: str) -> "Laker":
        """Deserialise a model from ``path``."""
        from laker.model import Laker

        state = torch.load(path, weights_only=True)
        dtype = torch.float32 if "float32" in state["dtype"] else torch.float64
        edt = state.get("embed_dtype")
        embed_dtype = (
            torch.float32
            if edt and "float32" in edt
            else torch.float64
            if edt and "float64" in edt
            else None
        )

        model = Laker(
            embed_dim=state["embed_dim"],
            lam=state["lam"],
            gamma=state["gamma"],
            num=state["num"],
            eps=state["eps"],
            base=state["base"],
            cccp_max=state["cccp_max"],
            cccp_tol=state["cccp_tol"],
            pcg_tol=state["pcg_tol"],
            pcg_max=state["pcg_max"],
            chunk=state.get("chunk"),
            kernel=state.get("kernel"),
            landmarks=state.get("landmarks"),
            features=state.get("features"),
            neighbors=state.get("neighbors"),
            grid_size=state.get("grid_size"),
            distributed=state.get("distributed", False),
            blend=state.get("blend", 0.5),
            selection=state.get("selection", "greedy"),
            pilot=state.get("pilot", 1000),
            knots=state.get("knots", 5),
            prec_kind=state.get("prec_kind", "cccp"),
            embed_dtype=embed_dtype,
            device=state["device"],
            dtype=dtype,
            verbose=state["verbose"],
        )

        if state.get("embed") is not None:
            model.embed_ = state["embed"].to(model.device)
        if state.get("coef") is not None:
            model.coef_ = state["coef"].to(model.device)
        if state.get("x_train") is not None:
            model._x_train = state["x_train"].to(model.device)
        if state.get("y_train") is not None:
            model._y_train = state["y_train"].to(model.device)

        if "encoder_state" in state:
            class_name = state["encoder_class"]
            module_name = state.get("encoder_module", "laker.embed")
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not import %s.%s; falling back to embed.Position.",
                    module_name,
                    class_name,
                )
                from laker.embed import Position as cls
                class_name = "Position"

            input_dim = state.get("input_dim", 2)
            embed_dtype_v = embed_dtype if embed_dtype else dtype
            enc_cls: Callable[..., Any] = cls
            if class_name == "Position":
                model.encoder_ = enc_cls(
                    input_dim=input_dim,
                    dim=model.embed_dim,
                    device=model.device,
                    dtype=embed_dtype_v,
                )
            else:
                try:
                    model.encoder_ = enc_cls(
                        input_dim=input_dim,
                        dim=model.embed_dim,
                        device=model.device,
                        dtype=embed_dtype_v,
                    )
                except TypeError:
                    model.encoder_ = cls()
                    model.encoder_.to(device=model.device, dtype=embed_dtype_v)
            model.encoder_.load_state_dict(state["encoder_state"])

        if "corrector_state" in state:
            cn = state.get("corrector_class", "Corrector")
            mn = state.get("corrector_module", "laker.corrector")
            try:
                cm = importlib.import_module(mn)
                cc = getattr(cm, cn)
            except (ImportError, AttributeError):
                logger.warning("Could not import %s.%s; skipping corrector.", mn, cn)
                cc = None
            if cc is not None:
                input_dim = state.get("input_dim", 2)
                model.corrector = cc(
                    input_dim=input_dim, output_dim=1, hidden_dim=32, dropout=0.1
                ).to(device=model.device, dtype=dtype)
                model.corrector.load_state_dict(state["corrector_state"])

        from laker.kernel import (
            Exact,
            Fourier,
            Grid,
            Hybrid,
            Neighbors,
            Nystrom,
            Spectrum,
        )

        if model.kernel is None or model.kernel == "exact":
            model.kernel_ = Exact(
                embeddings=model.embed_,
                lam=model.lam,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel == "nystrom":
            model.kernel_ = Nystrom(
                embeddings=model.embed_,
                lam=model.lam,
                num=model.landmarks,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel == "fourier":
            model.kernel_ = Fourier(
                embeddings=model.embed_,
                lam=model.lam,
                num=model.features,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel == "neighbors":
            model.kernel_ = Neighbors(
                embeddings=model.embed_,
                lam=model.lam,
                k=model.neighbors,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel == "grid":
            model.kernel_ = Grid(
                embeddings=model.embed_,
                lam=model.lam,
                grid_size=model.grid_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel == "spectrum":
            model.kernel_ = Spectrum(
                embeddings=model.embed_,
                lam=model.lam,
                knots=model.knots,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel == "hybrid":
            model.kernel_ = Hybrid(
                embeddings=model.embed_,
                lam=model.lam,
                alpha=model.blend,
                num=model.landmarks,
                k=model.neighbors,
                chunk=model.chunk,
                device=model.device,
                dtype=dtype,
            )

        if (
            "prec_state" in state
            and "prec_class" in state
            and model.kernel_ is not None
        ):
            try:
                pm = importlib.import_module(state["prec_module"])
                pc = getattr(pm, state["prec_class"])
                prec = pc.__new__(pc)
                for key, value in state["prec_state"].items():
                    if torch.is_tensor(value):
                        setattr(prec, key, value.to(model.device))
                    else:
                        setattr(prec, key, value)
                for attr in (
                    "gamma",
                    "eps",
                    "base",
                    "num",
                    "max_iter",
                    "tol",
                    "verbose",
                    "device",
                    "dtype",
                ):
                    if not hasattr(prec, attr) and hasattr(model, attr):
                        setattr(prec, attr, getattr(model, attr))
                if not hasattr(prec, "device"):
                    setattr(prec, "device", model.device)
                if not hasattr(prec, "dtype"):
                    setattr(prec, "dtype", dtype)
                model.prec_ = prec
            except Exception as exc:
                logger.warning(
                    "Could not restore preconditioner (%s); variance() after load will fail until refit.",
                    exc,
                )
        return model


__all__ = ["Store"]