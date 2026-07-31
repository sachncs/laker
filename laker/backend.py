"""Environment, dtype, and device management.

Single public class :class:`Backend` with single-word static methods.
All env-var parsing happens once at import time. Call
:meth:`Backend.load_env` to re-read after mutating the environment.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class Backend:
    """Single class exposing the LAKER backend configuration.

    All configuration values are read from env vars at import time
    (``LAKER_DEVICE``, ``LAKER_DTYPE``, ``LAKER_CHUNK_MEMORY_BUDGET``,
    ``LAKER_DISABLE_CHUNK``, ``LAKER_AUTOCAST``, ``LAKER_COMPILE_MODE``,
    ``LAKER_TF32``, ``LAKER_NUM_THREADS``, ``LAKER_SEED``). Use
    :meth:`load_env` to refresh after mutating the environment.
    """

    # Class-level state initialised from env vars at import time.
    device: torch.device = (
        torch.device(os.environ["LAKER_DEVICE"])
        if "LAKER_DEVICE" in os.environ
        else torch.device("cpu")
    )
    dtype: torch.dtype = (
        torch.float64 if os.environ.get("LAKER_DTYPE", "").lower() == "float64" else torch.float32
    )
    chunk_budget: int = int(os.environ.get("LAKER_CHUNK_MEMORY_BUDGET", "64")) * 1024 * 1024
    chunk_disabled: bool = os.environ.get("LAKER_DISABLE_CHUNK", "") == "1"
    autocast_enabled: bool = os.environ.get("LAKER_AUTOCAST", "") == "1"
    compile_mode: str = os.environ.get("LAKER_COMPILE_MODE", "")

    @classmethod
    def load_env(cls) -> None:
        """Re-read every ``LAKER_*`` env var and refresh cached state.

        Call this after mutating the environment at runtime. Idempotent.
        """
        cls.device = (
            torch.device(os.environ["LAKER_DEVICE"])
            if "LAKER_DEVICE" in os.environ
            else torch.device("cpu")
        )
        cls.dtype = (
            torch.float64
            if os.environ.get("LAKER_DTYPE", "").lower() == "float64"
            else torch.float32
        )
        cls.chunk_budget = int(os.environ.get("LAKER_CHUNK_MEMORY_BUDGET", "64")) * 1024 * 1024
        cls.chunk_disabled = os.environ.get("LAKER_DISABLE_CHUNK", "") == "1"
        cls.autocast_enabled = os.environ.get("LAKER_AUTOCAST", "") == "1"
        cls.compile_mode = os.environ.get("LAKER_COMPILE_MODE", "")
        if os.environ.get("LAKER_TF32", "1") != "0":
            torch.set_float32_matmul_precision("high")
        env_threads = os.environ.get("LAKER_NUM_THREADS")
        if env_threads:
            try:
                torch.set_num_threads(int(env_threads))
            except ValueError:
                logger.warning("Invalid LAKER_NUM_THREADS=%s; ignoring.", env_threads)
        env_seed = os.environ.get("LAKER_SEED")
        if env_seed:
            try:
                torch.manual_seed(int(env_seed))
            except ValueError:
                logger.warning("Invalid LAKER_SEED=%s; ignoring.", env_seed)

    @classmethod
    def set_device(
        cls,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.device:
        """Set the default device.

        Args:
            device: ``None`` auto-selects CUDA → MPS → CPU; otherwise
                the device string or ``torch.device``.

        Returns:
            The newly-set device.
        """
        if device is None:
            if torch.cuda.is_available():
                chosen = torch.device("cuda")
            elif torch.backends.mps.is_available():
                chosen = torch.device("mps")
            else:
                chosen = torch.device("cpu")
        else:
            chosen = torch.device(device)
        cls.device = chosen
        logger.info("Default device set to %s", chosen)
        return chosen

    @classmethod
    def set_dtype(cls, dtype: torch.dtype) -> None:
        """Set the default floating-point dtype.

        Args:
            dtype: A floating-point ``torch.dtype`` (``float16``,
                ``bfloat16``, ``float32``, ``float64``).
        """
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"set_dtype expects a floating dtype, got {dtype}")
        cls.dtype = dtype
        logger.info("Default dtype set to %s", dtype)

    @classmethod
    def set_chunk_budget(cls, megabytes: int) -> None:
        """Override the chunk-memory budget.

        Args:
            megabytes: Positive budget in megabytes.
        """
        if megabytes <= 0:
            raise ValueError(f"chunk budget must be positive megabytes, got {megabytes}")
        cls.chunk_budget = int(megabytes)

    @classmethod
    def tf32(cls) -> bool:
        """Return ``True`` when TF32 matmuls are enabled."""
        return torch.get_float32_matmul_precision() == "high"

    @classmethod
    def to_tensor(
        cls,
        data,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Coerce an array-like to a tensor.

        Args:
            data: Array-like (numpy array, list, or existing tensor).
            device: Target device; defaults to the backend default.
            dtype: Target dtype; defaults to the backend default.

        Returns:
            The coerced tensor on the requested device and dtype.
        """
        if device is None:
            device = cls.device
        if dtype is None:
            dtype = cls.dtype
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        return torch.as_tensor(data, device=device, dtype=dtype)

    @classmethod
    def maybe_compile(cls, func, mode: str = "reduce-overhead"):
        """Optionally compile ``func`` with :func:`torch.compile`.

        Activated only when ``LAKER_COMPILE_MODE`` is set; otherwise
        returns ``func`` unchanged.

        Args:
            func: Callable to compile.
            mode: ``torch.compile`` mode (``"default"``,
                ``"reduce-overhead"``, ``"max-autotune"``).

        Returns:
            The compiled function, or ``func`` unchanged.
        """
        if not cls.compile_mode:
            return func
        if not hasattr(torch, "compile"):
            return func
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return torch.compile(func, mode=cls.compile_mode or mode)

    @classmethod
    def autocast(cls):
        """Return an autocast context when ``LAKER_AUTOCAST=1``.

        On CUDA the context enables float16/bfloat16 matmuls; on
        CPU/MPS it is a no-op. Call :meth:`load_env` after changing
        the ``LAKER_AUTOCAST`` env var.

        Returns:
            Either an autocast context or ``contextlib.nullcontext``.
        """
        if not cls.autocast_enabled:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def seed(cls, value: int) -> None:
        """Seed torch and numpy RNG.

        Args:
            value: Non-negative integer seed.
        """
        torch.manual_seed(int(value))
        os.environ["LAKER_SEED"] = str(int(value))

    @classmethod
    def summary(cls) -> None:
        """Log one line describing the current backend configuration."""
        logger.info(
            "backend: device=%s dtype=%s chunk_mb=%s tf32=%s threads=%s",
            cls.device,
            cls.dtype,
            cls.chunk_budget,
            "high" if cls.tf32() else "default",
            torch.get_num_threads(),
        )


__all__ = ["Backend"]
