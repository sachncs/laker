"""Model persistence: save and load LAKER models.

This module provides :class:`ModelPersistence`, a utility class that handles
serialization and deserialization of fitted LAKER models to and from disk.

**State dict format.**
The ``save`` method collects all hyperparameters, fitted tensors (embeddings,
alpha), and (optionally) neural network state dicts (embedding model and
residual corrector) into a single dictionary and writes it with
:func:`torch.save`.  The embedding model's class name and module path are
recorded so that ``load`` can reconstruct the correct module type via
:func:`importlib.import_module`.

**Custom module support.**
If the saved embedding model or residual corrector is a custom ``nn.Module``
subclass, the loader will attempt to import it from its original module
path.  If the import fails (e.g. the class is not on ``sys.path``), the
loader falls back to :class:`~laker.embeddings.PositionEmbedding` and logs
a warning.  For reliable round-tripping of custom modules, ensure they are
importable from the same dotted path used during training.

**Kernel operator reconstruction.**
On ``load``, the kernel operator is reconstructed from the stored
``kernel_approx`` string, selecting the appropriate
:class:`~laker.kernels.KernelOperator` subclass (exact, Nyström, RFF,
k-NN, SKI, or two-scale).

Usage::

    from laker.models import LAKERRegressor

    reg = LAKERRegressor()
    reg.fit(x_train, y_train)
    reg.save("my_model.pt")

    loaded = LAKERRegressor.load("my_model.pt")
    loaded.predict(x_test)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from laker.models import LAKERRegressor

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handles serialization and deserialization of LAKER models.

    This class provides static methods for saving a fitted
    :class:`~laker.models.LAKERRegressor` to disk and loading it back,
    preserving all hyperparameters, learned embeddings, solver state, and
    optionally the neural network weights of the embedding model and
    residual corrector.

    The serialized format is a single dictionary written with
    :func:`torch.save` containing:

    * All regressor hyperparameters (``embedding_dim``, ``lambda_reg``, etc.).
    * Fitted tensors: ``embeddings`` and ``alpha``.
    * (Optional) ``embedding_model_state``, ``embedding_model_class``, and
      ``embedding_model_module`` for reconstructing the embedding network.
    * (Optional) ``residual_corrector_state``, ``residual_corrector_class``,
      and ``residual_corrector_module`` for the corrector MLP.
    """

    @staticmethod
    def save(regressor: "LAKERRegressor", path: str) -> None:
        """Serialize the fitted model to disk.

        Collects all hyperparameters, fitted tensors, and (optionally)
        neural network state dicts into a single dictionary and writes it
        to ``path`` using :func:`torch.save`.

        Args:
            regressor: A fitted :class:`~laker.models.LAKERRegressor`
                instance.  Must have ``alpha`` not ``None`` (i.e. must
                have been fitted via :meth:`~laker.models.LAKERRegressor.fit`
                or equivalent).
            path: Filesystem path where the serialized model will be
                written.  Overwrites any existing file at this location.

        Raises:
            RuntimeError: If the regressor has not been fitted (``alpha``
                is ``None``).

        Examples:
            >>> reg = LAKERRegressor()
            >>> reg.fit(x_train, y_train)
            >>> ModelPersistence.save(reg, "model.pt")
        """
        if regressor.alpha is None:
            raise RuntimeError("Model has not been fitted. Call fit() before saving.")
        state: dict[str, Any] = {
            "format_version": 2,
            "embedding_dim": regressor.embedding_dim,
            "lambda_reg": regressor.lambda_reg,
            "gamma": regressor.gamma,
            "num_probes": regressor.num_probes,
            "epsilon": regressor.epsilon,
            "base_rho": regressor.base_rho,
            "cccp_max_iter": regressor.cccp_max_iter,
            "cccp_tol": regressor.cccp_tol,
            "pcg_tol": regressor.pcg_tol,
            "pcg_max_iter": regressor.pcg_max_iter,
            "chunk_size": regressor.chunk_size,
            "kernel_approx": regressor.kernel_approx,
            "num_landmarks": regressor.num_landmarks,
            "num_features": regressor.num_features,
            "k_neighbors": regressor.k_neighbors,
            "grid_size": regressor.grid_size,
            "distributed": regressor.distributed,
            "twoscale_alpha": getattr(regressor, "twoscale_alpha", 0.5),
            "landmark_method": getattr(regressor, "landmark_method", "greedy"),
            "landmark_pilot_size": getattr(regressor, "landmark_pilot_size", 1000),
            "spectral_knots": getattr(regressor, "spectral_knots", 5),
            "preconditioner_strategy": getattr(regressor, "preconditioner_strategy", "cccp"),
            "device": str(regressor.device),
            "dtype": str(regressor.dtype),
            "embedding_dtype": (
                str(regressor.embedding_dtype) if regressor.embedding_dtype else None
            ),
            "verbose": regressor.verbose,
            "embeddings": regressor.embeddings.cpu() if regressor.embeddings is not None else None,
            "alpha": regressor.alpha.cpu() if regressor.alpha is not None else None,
            "x_train": regressor.x_train.cpu() if regressor.x_train is not None else None,
            "y_train": regressor.y_train.cpu() if regressor.y_train is not None else None,
        }
        # Save the preconditioner tensors if available.
        if regressor.preconditioner is not None:
            prec = regressor.preconditioner
            prec_class = prec.__class__.__name__
            prec_module = prec.__class__.__module__
            state["preconditioner_class"] = prec_class
            state["preconditioner_module"] = prec_module
            # Both CCCPPreconditioner and AdaptivePreconditioner expose
            # their state as torch tensors on self (set during build());
            # we extract tensor attributes for the round-trip.
            state["preconditioner_state"] = {
                k: v for k, v in vars(prec).items() if torch.is_tensor(v)
            }
        if regressor.embedding_model is not None:
            state["embedding_model_state"] = regressor.embedding_model.state_dict()
            state["embedding_model_class"] = regressor.embedding_model.__class__.__name__
            state["embedding_model_module"] = regressor.embedding_model.__class__.__module__
            if hasattr(regressor.embedding_model, "input_dim"):
                state["input_dim"] = regressor.embedding_model.input_dim
        if regressor.residual_corrector is not None:
            state["residual_corrector_state"] = regressor.residual_corrector.state_dict()
            state["residual_corrector_class"] = regressor.residual_corrector.__class__.__name__
            state["residual_corrector_module"] = regressor.residual_corrector.__class__.__module__
        torch.save(state, path)

    @staticmethod
    def load(path: str) -> "LAKERRegressor":
        """Deserialize a model from disk.

        Reconstructs a :class:`~laker.models.LAKERRegressor` with all
        hyperparameters, fitted tensors, and neural network weights
        restored from the file written by :meth:`save`.

        The embedding model and residual corrector are reconstructed by
        importing their original class via :func:`importlib.import_module`.
        If the import fails (e.g. the class is not on ``sys.path``), the
        loader falls back to the default
        :class:`~laker.embeddings.PositionEmbedding` for the embedding
        model and skips the residual corrector, logging a warning.

        The kernel operator is rebuilt from the stored ``kernel_approx``
        string, selecting the appropriate
        :class:`~laker.kernels.KernelOperator` subclass.

        Args:
            path: Filesystem path to the serialized model file written by
                :meth:`save`.

        Returns:
            A fully reconstructed :class:`~laker.models.LAKERRegressor`
            instance ready for prediction.

        Examples:
            >>> reg = LAKERRegressor.load("model.pt")
            >>> reg.predict(x_test)
        """
        state = torch.load(path, weights_only=True)
        dtype = torch.float32 if "float32" in state["dtype"] else torch.float64
        embedding_dtype = (
            torch.float32
            if state.get("embedding_dtype") and "float32" in state["embedding_dtype"]
            else (
                torch.float64
                if state.get("embedding_dtype") and "float64" in state["embedding_dtype"]
                else None
            )
        )

        from laker.models import LAKERRegressor

        model = LAKERRegressor(
            embedding_dim=state["embedding_dim"],
            lambda_reg=state["lambda_reg"],
            gamma=state["gamma"],
            num_probes=state["num_probes"],
            epsilon=state["epsilon"],
            base_rho=state["base_rho"],
            cccp_max_iter=state["cccp_max_iter"],
            cccp_tol=state["cccp_tol"],
            pcg_tol=state["pcg_tol"],
            pcg_max_iter=state["pcg_max_iter"],
            chunk_size=state.get("chunk_size"),
            kernel_approx=state.get("kernel_approx"),
            num_landmarks=state.get("num_landmarks"),
            num_features=state.get("num_features"),
            k_neighbors=state.get("k_neighbors"),
            grid_size=state.get("grid_size"),
            distributed=state.get("distributed", False),
            twoscale_alpha=state.get("twoscale_alpha", 0.5),
            landmark_method=state.get("landmark_method", "greedy"),
            landmark_pilot_size=state.get("landmark_pilot_size", 1000),
            spectral_knots=state.get("spectral_knots", 5),
            preconditioner=state.get("preconditioner_strategy", "cccp"),
            embedding_dtype=embedding_dtype,
            device=state["device"],
            dtype=dtype,
            verbose=state["verbose"],
        )
        model.embeddings = state["embeddings"].to(model.device)
        model.alpha = state["alpha"].to(model.device)
        if state.get("x_train") is not None:
            model.x_train = state["x_train"].to(model.device)
        if state.get("y_train") is not None:
            model.y_train = state["y_train"].to(model.device)

        if "embedding_model_state" in state:
            class_name = state["embedding_model_class"]
            module_name = state.get("embedding_model_module", "laker.embeddings")
            try:
                import importlib

                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not import %s.%s for loading; falling back to PositionEmbedding. "
                    "Save/load of custom embedding modules requires the module to be importable.",
                    module_name,
                    class_name,
                )
                from laker.embeddings import PositionEmbedding as cls

                class_name = "PositionEmbedding"

            input_dim = state.get("input_dim", 2)
            embed_dtype = embedding_dtype if embedding_dtype else dtype
            embed_cls: Callable[..., Any] = cls
            if class_name == "PositionEmbedding":
                model.embedding_model = embed_cls(
                    input_dim=input_dim,
                    embedding_dim=model.embedding_dim,
                    device=model.device,
                    dtype=embed_dtype,
                )
            else:
                try:
                    model.embedding_model = embed_cls(
                        input_dim=input_dim,
                        embedding_dim=model.embedding_dim,
                        device=model.device,
                        dtype=embed_dtype,
                    )
                except TypeError:
                    model.embedding_model = cls()
                    model.embedding_model.to(device=model.device, dtype=embed_dtype)
            model.embedding_model.load_state_dict(state["embedding_model_state"])

        if "residual_corrector_state" in state:
            corr_class_name = state.get("residual_corrector_class", "ResidualCorrector")
            corr_module_name = state.get("residual_corrector_module", "laker.correctors")
            try:
                import importlib

                corr_module = importlib.import_module(corr_module_name)
                corr_cls = getattr(corr_module, corr_class_name)
            except (ImportError, AttributeError):
                logger.warning(
                    "Could not import %s.%s for loading residual corrector; skipping.",
                    corr_module_name,
                    corr_class_name,
                )
                corr_cls = None

            if corr_cls is not None:
                input_dim = state.get("input_dim", 2)
                model.residual_corrector = corr_cls(
                    input_dim=input_dim,
                    output_dim=1,
                    hidden_dim=32,
                    dropout=0.1,
                ).to(model.device)
                model.residual_corrector.load_state_dict(state["residual_corrector_state"])

        from laker.kernels import (
            AttentionKernelOperator,
            NystromAttentionKernelOperator,
            RandomFeatureAttentionKernelOperator,
            SKIAttentionKernelOperator,
            SparseKNNAttentionKernelOperator,
            SpectralAttentionKernelOperator,
            TwoScaleAttentionKernelOperator,
        )

        if model.kernel_approx is None or model.kernel_approx == "exact":
            model.kernel_operator = AttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "nystrom":
            model.kernel_operator = NystromAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                num_landmarks=model.num_landmarks,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "rff":
            model.kernel_operator = RandomFeatureAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                num_features=model.num_features,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "knn":
            model.kernel_operator = SparseKNNAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                k_neighbors=model.k_neighbors,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "ski":
            model.kernel_operator = SKIAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                grid_size=model.grid_size,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "spectral":
            model.kernel_operator = SpectralAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                num_knots=model.spectral_knots,
                device=model.device,
                dtype=dtype,
            )
        elif model.kernel_approx == "twoscale":
            model.kernel_operator = TwoScaleAttentionKernelOperator(
                embeddings=model.embeddings,
                lambda_reg=model.lambda_reg,
                alpha=model.twoscale_alpha,
                num_landmarks=model.num_landmarks,
                k_neighbors=model.k_neighbors,
                chunk_size=model.chunk_size,
                device=model.device,
                dtype=dtype,
            )

        # Restore preconditioner state.
        if (
            "preconditioner_state" in state
            and "preconditioner_class" in state
            and model.kernel_operator is not None
        ):
            try:
                import importlib

                prec_module = importlib.import_module(state["preconditioner_module"])
                prec_cls = getattr(prec_module, state["preconditioner_class"])
                prec = prec_cls.__new__(prec_cls)
                # Restore tensor attributes onto the instance.
                for key, value in state["preconditioner_state"].items():
                    if torch.is_tensor(value):
                        setattr(prec, key, value.to(model.device))
                    else:
                        setattr(prec, key, value)
                # Reattach other required non-tensor state (gamma, base_rho).
                for attr in (
                    "gamma",
                    "epsilon",
                    "base_rho",
                    "num_probes",
                    "max_iter",
                    "tol",
                    "verbose",
                    "device",
                    "dtype",
                ):
                    if not hasattr(prec, attr) and hasattr(model, attr):
                        setattr(prec, attr, getattr(model, attr))
                # Ensure device/dtype fields are correct torch types.
                if not hasattr(prec, "device"):
                    setattr(prec, "device", model.device)
                if not hasattr(prec, "dtype"):
                    setattr(prec, "dtype", dtype)
                model.preconditioner = prec
            except Exception as exc:
                logger.warning(
                    "Could not restore preconditioner (%s); variance() "
                    "after load will fail until refit.",
                    exc,
                )
        return model
