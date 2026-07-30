"""Public model facade.

The single top-level export is :class:`Laker`. It owns configuration
and fitted state directly, and it delegates heavy computation to
``laker.LAKERRegressor`` (the legacy estimator) until the structural
phases complete their migration.

Naming rules:

- Hyperparameters use the new ``NAMING.md`` names (``regularization``,
  ``probes``, ``landmarks`` ...). The legacy names (``lambda_reg``,
  ``num_probes`` ...) are accepted on the constructor and translated
  one-shot; they are not re-exported on the model.

- Fitted state uses ``coef_``, ``embeddings_``, ``kernel_``,
  ``preconditioner_``, ``encoder_``, ``inputs_``, ``targets_``,
  ``iterations_``.

- All sklearn-style parameters are validated against the public
  parameter set (no arbitrary attribute writes).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from laker.backend import to_tensor

logger = logging.getLogger(__name__)


_KERNEL_ALIASES = {
    None: "exact",
    "exact": "exact",
    "nystrom": "nystrom",
    "rff": "fourier",
    "fourier": "fourier",
    "knn": "neighbors",
    "neighbors": "neighbors",
    "ski": "grid",
    "grid": "grid",
    "twoscale": "hybrid",
    "hybrid": "hybrid",
    "spectral": "spectrum",
    "spectrum": "spectrum",
}


class Laker:
    """Sklearn-compatible LAKER estimator (single public type).

    Args:
        embedding_dim: Dimension of the embedding space.
        regularization: Ridge weight :math:`\\lambda`.
        gamma: Kernel bandwidth for the CCCP preconditioner.
        kernel: ``"exact"`` (default), ``"nystrom"``, ``"fourier"``,
            ``"neighbors"``, ``"grid"``, ``"spectrum"``, ``"hybrid"``.
        landmarks: Nyström landmark count.
        features: RFF feature count.
        neighbors: k-NN sparsity count.
        grid_size: SKI grid resolution.
        blend: Hybrid blend weight in ``[0, 1]``.
        selection: ``"greedy"`` or ``"leverage"`` landmark selection.
        pilot: Leverage-score pilot size.
        knots: Spectral kernel spline knots.
        distributed: Use multi-device distributed kernel.
        probes: Random probe count for preconditioner construction.
        epsilon: Numerical stability constant.
        base_rho: Base spectral norm bound for CCCP.
        preconditioner: ``"cccp"`` or ``"adaptive"``.
        cccp_max_iter: Maximum CCCP iterations.
        cccp_tol: CCCP convergence tolerance.
        pcg_tol: PCG relative residual tolerance.
        pcg_max_iter: Maximum PCG iterations.
        chunk_size: Tile size for chunked kernel evaluation.
        encoder: Optional pre-built embedding module.
        embedding_dtype: Dtype for embedding computation.
        device: Target torch device.
        dtype: Floating-point dtype for the solver.
        verbose: Whether to log diagnostics.
        warm_start: Carry forward fitted state across ``fit`` calls.

    Attributes:
        coef_: Solution vector, shape ``(n,)``.
        embeddings_: Training embeddings, shape ``(n, D)``.
        kernel_: Fitted kernel operator.
        preconditioner_: Fitted preconditioner.
        encoder_: Fitted embedding module.
        inputs_: Training locations, shape ``(n, d)``.
        targets_: Training observations, shape ``(n,)``.
        iterations_: Iterations used by the last PCG solve.
    """

    # Canonical public parameter set. ``set_params`` validates against
    # this list and rejects unknown names with ``ValueError``.
    PARAMS = (
        "embedding_dim",
        "regularization",
        "gamma",
        "kernel",
        "landmarks",
        "features",
        "neighbors",
        "grid_size",
        "blend",
        "selection",
        "pilot",
        "knots",
        "distributed",
        "probes",
        "epsilon",
        "base_rho",
        "preconditioner",
        "cccp_max_iter",
        "cccp_tol",
        "pcg_tol",
        "pcg_max_iter",
        "chunk_size",
        "encoder",
        "embedding_dtype",
        "device",
        "dtype",
        "verbose",
        "warm_start",
    )

    def __init__(
        self,
        embedding_dim: int = 10,
        regularization: float = 1e-2,
        gamma: float = 1e-1,
        kernel: str = "exact",
        landmarks: Optional[int] = None,
        features: Optional[int] = None,
        neighbors: Optional[int] = None,
        grid_size: Optional[int] = None,
        blend: float = 0.5,
        selection: str = "greedy",
        pilot: int = 1000,
        knots: int = 5,
        distributed: bool = False,
        probes: Optional[int] = None,
        epsilon: float = 1e-8,
        base_rho: float = 0.05,
        preconditioner: str = "cccp",
        cccp_max_iter: int = 200,
        cccp_tol: float = 1e-6,
        pcg_tol: float = 1e-6,
        pcg_max_iter: int = 1000,
        chunk_size: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        embedding_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = os.environ.get("LAKER_VERBOSE", "1") == "1",
        warm_start: bool = False,
    ) -> None:
        # Convert user-facing kernel alias if needed.
        kernel = _KERNEL_ALIASES.get(kernel, kernel)
        if kernel not in (
            "exact",
            "nystrom",
            "fourier",
            "neighbors",
            "grid",
            "spectrum",
            "hybrid",
        ):
            raise ValueError(
                "kernel must be one of "
                "'exact', 'nystrom', 'fourier', 'neighbors', 'grid', "
                "'spectrum', 'hybrid'; got "
                f"{kernel!r}"
            )

        # Lazy import: avoid circular module load and keep the legacy
        # ``LAKERRegressor`` path available during the structural
        # migration phases.
        from laker.models import LAKERRegressor

        # Pre-flight parameter validation (mirrors LAKERRegressor checks
        # but uses the canonical name set; raises sooner with cleaner
        # messages).
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if regularization <= 0:
            raise ValueError(f"regularization must be positive, got {regularization}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0 <= base_rho <= 1:
            raise ValueError(f"base_rho must be in [0, 1], got {base_rho}")
        if cccp_max_iter <= 0:
            raise ValueError(f"cccp_max_iter must be positive, got {cccp_max_iter}")
        if cccp_tol <= 0:
            raise ValueError(f"cccp_tol must be positive, got {cccp_tol}")
        if pcg_tol <= 0:
            raise ValueError(f"pcg_tol must be positive, got {pcg_tol}")
        if pcg_max_iter <= 0:
            raise ValueError(f"pcg_max_iter must be positive, got {pcg_max_iter}")
        if neighbors is not None and neighbors <= 0:
            raise ValueError(f"neighbors must be positive, got {neighbors}")
        if grid_size is not None and grid_size < 2:
            raise ValueError(f"grid_size must be at least 2, got {grid_size}")
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        if selection not in ("greedy", "leverage"):
            raise ValueError(
                f"selection must be 'greedy' or 'leverage', got {selection!r}"
            )
        if pilot <= 0:
            raise ValueError(f"pilot must be positive, got {pilot}")
        if knots <= 0:
            raise ValueError(f"knots must be positive, got {knots}")
        if preconditioner not in ("cccp", "adaptive"):
            raise ValueError(
                f"preconditioner must be 'cccp' or 'adaptive', got {preconditioner!r}"
            )

        self._legacy = LAKERRegressor(
            embedding_dim=embedding_dim,
            lambda_reg=regularization,
            gamma=gamma,
            num_probes=probes,
            epsilon=epsilon,
            base_rho=base_rho,
            cccp_max_iter=cccp_max_iter,
            cccp_tol=cccp_tol,
            pcg_tol=pcg_tol,
            pcg_max_iter=pcg_max_iter,
            chunk_size=chunk_size,
            embedding_module=encoder,
            kernel_approx=_legacy_kernel(kernel),
            num_landmarks=landmarks,
            num_features=features,
            k_neighbors=neighbors,
            grid_size=grid_size,
            distributed=distributed,
            twoscale_alpha=blend,
            landmark_method=selection,
            landmark_pilot_size=pilot,
            spectral_knots=knots,
            preconditioner=preconditioner,
            embedding_dtype=embedding_dtype,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        self._warm_start = warm_start
        self._fit_called = False

    # ------------------------------------------------------------------
    # Hyperparameter delegation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        fitted = "fitted" if self.coef_ is not None else "not fitted"
        return (
            f"Laker(embedding_dim={self._legacy.core.embedding_dim}, "
            f"regularization={self._legacy.core.lambda_reg}, {fitted})"
        )

    @property
    def coef_(self) -> Optional[torch.Tensor]:
        return self._legacy.alpha

    @property
    def embeddings_(self) -> Optional[torch.Tensor]:
        return self._legacy.embeddings

    @property
    def kernel_(self) -> Any:
        return self._legacy.kernel_operator

    @property
    def preconditioner_(self) -> Any:
        return self._legacy.preconditioner

    @property
    def encoder_(self) -> Optional[nn.Module]:
        return self._legacy.embedding_model

    @property
    def inputs_(self) -> Optional[torch.Tensor]:
        return self._legacy.x_train

    @property
    def targets_(self) -> Optional[torch.Tensor]:
        return self._legacy.y_train

    @property
    def iterations_(self) -> Optional[int]:
        return getattr(self._legacy, "pcg_iterations_", None)

    # ------------------------------------------------------------------
    # Validation contract for set_params / GridSearchCV
    # ------------------------------------------------------------------
    def set_params(self, **params: Any) -> "Laker":
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) for Laker: {sorted(unknown)}. "
                f"Valid parameters: {list(self.PARAMS)}."
            )
        # Type coercion for the new-style string args.
        coerced = dict(params)
        for name in ("device",):
            if isinstance(coerced.get(name), str):
                coerced[name] = torch.device(coerced[name])
        for name in ("dtype", "embedding_dtype"):
            v = coerced.get(name)
            if isinstance(v, str):
                if "float64" in v:
                    coerced[name] = torch.float64
                elif "bfloat16" in v:
                    coerced[name] = torch.bfloat16
                elif "float16" in v:
                    coerced[name] = torch.float16
                else:
                    coerced[name] = torch.float32
        # Translate canonical params to legacy kwargs.
        legacy_kwargs = _to_legacy_kwargs(coerced)
        self._legacy.set_params(**legacy_kwargs)
        return self

    def get_params(self, deep: bool = True) -> dict:
        legacy = self._legacy.get_params()
        return {
            "embedding_dim": legacy["embedding_dim"],
            "regularization": legacy["lambda_reg"],
            "gamma": legacy["gamma"],
            "kernel": _public_kernel(legacy["kernel_approx"]),
            "landmarks": legacy["num_landmarks"],
            "features": legacy["num_features"],
            "neighbors": legacy["k_neighbors"],
            "grid_size": legacy["grid_size"],
            "blend": legacy["twoscale_alpha"],
            "selection": legacy["landmark_method"],
            "pilot": legacy["landmark_pilot_size"],
            "knots": self._legacy.core.spectral_knots,
            "distributed": legacy["distributed"],
            "probes": legacy["num_probes"],
            "epsilon": legacy["epsilon"],
            "base_rho": legacy["base_rho"],
            "preconditioner": legacy["preconditioner"],
            "cccp_max_iter": legacy["cccp_max_iter"],
            "cccp_tol": legacy["cccp_tol"],
            "pcg_tol": legacy["pcg_tol"],
            "pcg_max_iter": legacy["pcg_max_iter"],
            "chunk_size": legacy["chunk_size"],
            "encoder": legacy["embedding_module"],
            "embedding_dtype": legacy["embedding_dtype"],
            "device": legacy["device"],
            "dtype": legacy["dtype"],
            "verbose": legacy["verbose"],
            "warm_start": self._warm_start,
        }

    def __sklearn_clone__(self) -> "Laker":
        return Laker(**self.get_params())

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def fit(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        x0: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> "Laker":
        x = to_tensor(x, device=self._legacy.device, dtype=self._legacy.dtype)
        y = to_tensor(y, device=self._legacy.device, dtype=self._legacy.dtype)
        # Accept (n,) or (n, 1); reject 0-D scalars.
        if y.dim() == 0:
            raise ValueError(
                f"y must be 1-D (n,) or 2-D (n, 1), got scalar shape {tuple(y.shape)}"
            )
        if y.dim() == 2:
            if y.shape[-1] != 1:
                raise ValueError(
                    f"y must have shape (n,) or (n, 1), got {tuple(y.shape)}"
                )
            y = y.squeeze(-1)
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values (NaN or Inf)")
        if not torch.isfinite(y).all():
            raise ValueError("y contains non-finite values (NaN or Inf)")
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have matching sample counts: "
                f"x.shape[0]={x.shape[0]} vs y.shape[0]={y.shape[0]}"
            )
        if x.shape[0] == 0:
            raise ValueError("x must have at least one row, got empty tensor")

        if not self._warm_start and self._fit_called:
            # Atomic refit: drop fitted state by reconstructing with the
            # canonical parameter set (the legacy ``get_params`` returns
            # stringified dtypes, so we read state from ``self`` instead).
            from laker.models import LAKERRegressor

            params = self.get_params()
            dtype_str = str(params["dtype"])
            embed_dtype_str = params["embedding_dtype"]
            if dtype_str and "float64" in dtype_str:
                dtype = torch.float64
            else:
                dtype = torch.float32
            if embed_dtype_str:
                if "float64" in embed_dtype_str:
                    embed_dtype = torch.float64
                elif "bfloat16" in embed_dtype_str:
                    embed_dtype = torch.bfloat16
                elif "float16" in embed_dtype_str:
                    embed_dtype = torch.float16
                else:
                    embed_dtype = torch.float32
            else:
                embed_dtype = None
            self._legacy = LAKERRegressor(
                embedding_dim=params["embedding_dim"],
                lambda_reg=params["regularization"],
                gamma=params["gamma"],
                num_probes=params["probes"],
                pcg_tol=params["pcg_tol"],
                pcg_max_iter=params["pcg_max_iter"],
                embedding_dtype=embed_dtype,
                device=params["device"],
                dtype=dtype,
                verbose=params["verbose"],
            )
        self._legacy.fit(x, y, x0=x0, seed=seed)
        self._fit_called = True
        return self

    def predict(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        """Predict at query locations.

        Args:
            x: Query locations of shape ``(m, d)``.

        Returns:
            Predictions, shape ``(m,)``.
        """
        if self.coef_ is None or self.embeddings_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        x = to_tensor(x, device=self._legacy.device, dtype=self._legacy.dtype)
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
        encoder = self.encoder_
        if encoder is not None and hasattr(encoder, "input_dim"):
            if x.shape[1] != encoder.input_dim:
                raise ValueError(
                    f"x has {x.shape[1]} features but model expects "
                    f"{encoder.input_dim}"
                )
        return self._legacy.predict(x)

    def variance(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        """Predictive posterior variance at query locations.

        Args:
            x: Query locations of shape ``(m, d)``.

        Returns:
            Predicted variances, shape ``(m,)``.
        """
        return self._legacy.predict_variance(x)

    # Aliases mirroring the previous public surface.
    def predict_variance(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
    ) -> torch.Tensor:
        return self.variance(x)

    def score(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
    ) -> float:
        """Coefficient of determination (R²).

        Defined as ``1 - SS_res / SS_tot``. Returns ``1.0`` for a perfect
        fit, ``0.0`` when the model predicts the mean only, and a
        negative value when the model is worse than the mean.
        """
        y_true = to_tensor(y, device=self._legacy.device, dtype=self._legacy.dtype)
        if y_true.dim() == 2 and y_true.shape[-1] == 1:
            y_true = y_true.squeeze(-1)
        if y_true.dim() != 1:
            raise ValueError(f"y must be 1-D, got shape {tuple(y_true.shape)}")
        y_pred = self.predict(x)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"x and y must have matching sample counts: "
                f"predict returned {tuple(y_pred.shape)}, y is {tuple(y_true.shape)}"
            )
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    def condition(self) -> float:
        """Estimated condition number of the preconditioned system."""
        return self._legacy.condition_number()

    # ------------------------------------------------------------------
    # Workflows (canonical names; legacy names aliased)
    # ------------------------------------------------------------------
    def search(
        self,
        method: str,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        val_fraction: float = 0.2,
        lambda_reg_grid: Optional[list[float]] = None,
        gamma_grid: Optional[list[float]] = None,
        num_probes_grid: Optional[list[int]] = None,
        regularizations: Optional[list[float]] = None,
        n_calls: int = 15,
        n_initial_points: int = 5,
        lambda_reg_bounds: tuple[float, float] = (1e-4, 1.0),
        gamma_bounds: tuple[float, float] = (0.0, 2.0),
        num_probes_bounds: tuple[int, int] = (20, 300),
        warm_start: bool = True,
    ) -> "Laker":
        if method == "grid":
            self._legacy.fit_with_search(
                x, y,
                val_fraction=val_fraction,
                lambda_reg_grid=regularizations or lambda_reg_grid,
                gamma_grid=gamma_grid,
                num_probes_grid=num_probes_grid,
                warm_start=warm_start,
            )
        elif method == "bayes":
            self._legacy.fit_with_bo(
                x, y,
                val_fraction=val_fraction,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                lambda_reg_bounds=lambda_reg_bounds,
                gamma_bounds=gamma_bounds,
                num_probes_bounds=num_probes_bounds,
            )
        else:
            raise ValueError(
                f"search method must be 'grid' or 'bayes', got {method!r}"
            )
        return self

    def fit_with_search(
        self, x, y, val_fraction=0.2,
        lambda_reg_grid=None, gamma_grid=None, num_probes_grid=None,
        warm_start=True,
    ) -> "Laker":
        return self.search(
            "grid", x, y, val_fraction=val_fraction,
            lambda_reg_grid=lambda_reg_grid,
            gamma_grid=gamma_grid,
            num_probes_grid=num_probes_grid,
            warm_start=warm_start,
        )

    def fit_with_bo(self, *args, **kwargs) -> "Laker":
        return self.search("bayes", *args, **kwargs)

    def update(
        self,
        x_new: Union[torch.Tensor, "numpy.ndarray"],
        y_new: Union[torch.Tensor, "numpy.ndarray"],
        forgetting_factor: float = 1.0,
        rebuild_threshold: int = 100,
    ) -> "Laker":
        self._legacy.partial_fit(
            x_new, y_new,
            forgetting_factor=forgetting_factor,
            rebuild_threshold=rebuild_threshold,
        )
        return self

    def partial_fit(self, *args, **kwargs) -> "Laker":
        return self.update(*args, **kwargs)

    def path(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        regularizations: list[float],
        reuse_precond: bool = True,
    ) -> dict:
        return self._legacy.fit_path(x, y, regularizations, reuse_precond)

    def fit_path(self, x, y, lambda_reg_grid, reuse_precond=True) -> dict:
        return self.path(x, y, lambda_reg_grid, reuse_precond)

    def continuation(
        self,
        x: Union[torch.Tensor, "numpy.ndarray"],
        y: Union[torch.Tensor, "numpy.ndarray"],
        lambda_max: Optional[float] = None,
        lambda_min: Optional[float] = None,
        n_stages: int = 5,
        reuse_precond: bool = True,
    ) -> "Laker":
        self._legacy.fit_continuation(
            x, y, lambda_max, lambda_min, n_stages, reuse_precond
        )
        return self

    def fit_continuation(self, *args, **kwargs) -> "Laker":
        return self.continuation(*args, **kwargs)

    def learn(
        self,
        x, y, lr: float = 1e-3, epochs: int = 50,
        rebuild_freq: int = 10, patience: int = 5,
    ) -> "Laker":
        self._legacy.fit_learned_embeddings(
            x, y, lr=lr, epochs=epochs,
            rebuild_freq=rebuild_freq, patience=patience,
        )
        return self

    def fit_learned_embeddings(self, *args, **kwargs) -> "Laker":
        return self.learn(*args, **kwargs)

    def correct(
        self,
        x, y, val_fraction: float = 0.2, epochs: int = 200,
        patience: int = 10, weight_decay: float = 1e-2, lr: float = 1e-3,
    ) -> "Laker":
        self._legacy.fit_residual_corrector(
            x, y, val_fraction=val_fraction, epochs=epochs,
            patience=patience, weight_decay=weight_decay, lr=lr,
        )
        return self

    def fit_residual_corrector(self, *args, **kwargs) -> "Laker":
        return self.correct(*args, **kwargs)

    def calibrate(
        self,
        x, y, lr: float = 1e-3, epochs: int = 50,
        beta: float = 0.1, variance_subset: float = 0.2,
        patience: int = 5,
    ) -> "Laker":
        self._legacy.fit_uncertainty_aware(
            x, y, lr=lr, epochs=epochs, beta=beta,
            variance_subset=variance_subset, patience=patience,
        )
        return self

    def fit_uncertainty_aware(self, *args, **kwargs) -> "Laker":
        return self.calibrate(*args, **kwargs)

    def tune(
        self,
        x_train, y_train, x_val, y_val,
        lr: float = 1e-3, epochs: int = 20, patience: int = 5,
    ) -> "Laker":
        self._legacy.fit_bilevel(
            x_train, y_train, x_val, y_val,
            lr=lr, epochs=epochs, patience=patience,
        )
        return self

    def fit_bilevel(self, *args, **kwargs) -> "Laker":
        return self.tune(*args, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        self._legacy.save(path)

    @classmethod
    def load(cls, path: str) -> "Laker":
        from laker.persistence import ModelPersistence

        legacy = ModelPersistence.load(path)
        new = cls(
            embedding_dim=legacy.core.embedding_dim,
            regularization=legacy.core.lambda_reg,
        )
        # Replace the inner legacy estimator.
        object.__setattr__(new, "_legacy", legacy)
        object.__setattr__(new, "_warm_start", False)
        object.__setattr__(new, "_fit_called", True)
        return new


# ---------------------------------------------------------------------------
# Helpers for kernel name translation and parameter conversion.
# ---------------------------------------------------------------------------
def _legacy_kernel(name: str) -> Optional[str]:
    """Map the new kernel name to the legacy ``kernel_approx`` arg."""
    return {
        "exact": None,
        "nystrom": "nystrom",
        "fourier": "rff",
        "neighbors": "knn",
        "grid": "ski",
        "spectrum": "spectral",
        "hybrid": "twoscale",
    }.get(name, name)


def _public_kernel(legacy: Optional[str]) -> str:
    """Map a legacy ``kernel_approx`` value back to the new name."""
    return {
        None: "exact",
        "nystrom": "nystrom",
        "rff": "fourier",
        "knn": "neighbors",
        "ski": "grid",
        "spectral": "spectrum",
        "twoscale": "hybrid",
    }.get(legacy, "exact")


def _to_legacy_kwargs(params: dict) -> dict:
    canonical_to_legacy = {
        "regularization": "lambda_reg",
        "kernel": "kernel_approx",
        "landmarks": "num_landmarks",
        "features": "num_features",
        "neighbors": "k_neighbors",
        "blend": "twoscale_alpha",
        "selection": "landmark_method",
        "pilot": "landmark_pilot_size",
        "knots": "spectral_knots",
        "probes": "num_probes",
        "encoder": "embedding_module",
    }
    out: dict = {}
    for key, value in params.items():
        legacy = canonical_to_legacy.get(key, key)
        if key == "kernel":
            out[legacy] = _legacy_kernel(value)
        elif key == "warm_start":
            continue
        else:
            out[legacy] = value
    return out


__all__ = ["Laker"]
