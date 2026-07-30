"""Environment, dtype, and device management.

Module exposes the :class:`Backend` static-method API. The legacy
free functions are kept as thin shims for backward compatibility
through the structural migration.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class Backend:
    """Package-wide environment, dtype, and device management."""

    @staticmethod
    def default_device() -> torch.device:
        """Return the current default compute device."""
        return get_default_device()

    @staticmethod
    def set_default_device(
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.device:
        """Set the default device; ``None`` auto-selects CUDA → MPS → CPU."""
        return set_default_device(device)

    @staticmethod
    def default_dtype() -> torch.dtype:
        """Return the current default floating-point dtype."""
        return get_default_dtype()

    @staticmethod
    def set_default_dtype(dtype: torch.dtype) -> None:
        """Set the default floating-point dtype (must be a floating type)."""
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(
                f"set_default_dtype expects a floating dtype, got {dtype}"
            )
        set_default_dtype(dtype)

    @staticmethod
    def get_default_dtype() -> torch.dtype:
        """Return the current default floating-point dtype."""
        return get_default_dtype()

    @staticmethod
    def get_chunk_budget() -> int:
        """Return the chunk-memory budget in bytes (default 64 MB)."""
        return get_chunk_memory_budget()

    @staticmethod
    def set_chunk_budget(megabytes: int) -> None:
        """Override the chunk-memory budget. ``megabytes`` must be positive."""
        global _LAKER_CHUNK_BUDGET_MB
        if megabytes <= 0:
            raise ValueError(
                f"chunk budget must be positive megabytes, got {megabytes}"
            )
        _LAKER_CHUNK_BUDGET_MB = int(megabytes)

    @staticmethod
    def to_tensor(
        data,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Coerce an array-like to a tensor on the requested device/dtype."""
        return to_tensor(data, device=device, dtype=dtype)

    @staticmethod
    def maybe_compile(func, mode: str = "reduce-overhead"):
        """Optionally compile with :func:`torch.compile`."""
        return maybe_compile(func, mode=mode)

    @staticmethod
    def seed(value: int) -> None:
        """Seed torch and numpy RNG."""
        torch.manual_seed(int(value))
        os.environ["LAKER_SEED"] = str(int(value))

    @staticmethod
    def load_env() -> None:
        """Re-read all ``LAKER_*`` env vars."""
        _init_from_env()

    @staticmethod
    def print_summary() -> None:
        """Log one line describing the current backend configuration."""
        logger.info(
            "backend: device=%s dtype=%s chunk_mb=%s tf32=%s threads=%s",
            get_default_device(),
            get_default_dtype(),
            _LAKER_CHUNK_BUDGET_MB,
            "high" if torch.get_float32_matmul_precision() == "high" else "default",
            torch.get_num_threads(),
        )


# ---------------------------------------------------------------------------
# Module-level constants and legacy functions (private; used by shims).
# ---------------------------------------------------------------------------
DEFAULT_DEVICE: torch.device = torch.device("cpu")
DEFAULT_DTYPE: torch.dtype = torch.float32

_LAKER_COMPILE_MODE = os.environ.get("LAKER_COMPILE_MODE", "")
_LAKER_CHUNK_BUDGET_MB = int(os.environ.get("LAKER_CHUNK_MEMORY_BUDGET", "64"))
_LAKER_DISABLE_CHUNK = os.environ.get("LAKER_DISABLE_CHUNK", "") == "1"

_ENV_INIT_DONE = False


def _init_from_env() -> None:
    global _ENV_INIT_DONE
    if _ENV_INIT_DONE:
        return
    _ENV_INIT_DONE = True
    env_device = os.environ.get("LAKER_DEVICE")
    if env_device:
        set_default_device(env_device)
    env_dtype = os.environ.get("LAKER_DTYPE")
    if env_dtype == "float32":
        set_default_dtype(torch.float32)
    elif env_dtype == "float64":
        set_default_dtype(torch.float64)
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


def get_default_device() -> torch.device:
    _init_from_env()
    return DEFAULT_DEVICE


def set_default_device(
    device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    global DEFAULT_DEVICE
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    DEFAULT_DEVICE = device
    logger.info("Default device set to %s", device)
    return device


def get_default_dtype() -> torch.dtype:
    _init_from_env()
    return DEFAULT_DTYPE


def set_default_dtype(dtype: torch.dtype) -> None:
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype
    logger.info("Default dtype set to %s", dtype)


def get_chunk_memory_budget() -> int:
    return _LAKER_CHUNK_BUDGET_MB * 1024 * 1024


def get_chunk_disabled() -> bool:
    return _LAKER_DISABLE_CHUNK


def maybe_compile(func, mode: str = "reduce-overhead"):
    if "LAKER_COMPILE_MODE" not in os.environ:
        return func
    if not hasattr(torch, "compile"):
        return func
    compile_mode = _LAKER_COMPILE_MODE or mode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return torch.compile(func, mode=compile_mode)


def to_tensor(
    data,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if device is None:
        device = get_default_device()
    if dtype is None:
        dtype = get_default_dtype()
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.as_tensor(data, device=device, dtype=dtype)


_init_from_env()


__all__ = [
    "Backend",
    "get_default_device",
    "set_default_device",
    "get_default_dtype",
    "set_default_dtype",
    "get_chunk_memory_budget",
    "maybe_compile",
    "to_tensor",
]
