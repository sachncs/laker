"""Device and dtype management utilities.

This module is the single source of truth for the package-wide default
:class:`torch.device` and :class:`torch.dtype` used by every other module
when an explicit device/dtype is not passed in. It also exposes a thin
:func:`to_tensor` helper that coerces array-like inputs (lists, NumPy
arrays, existing tensors) into a :class:`torch.Tensor` on the chosen
device and dtype, and a :func:`maybe_compile` helper that wraps a callable
with :func:`torch.compile` when available.

The defaults can be overridden through three mechanisms, listed in
increasing precedence:

1. The hard-coded module-level constants :data:`DEFAULT_DEVICE` (CPU) and
   :data:`DEFAULT_DTYPE` (``float32``).
2. The environment variables ``LAKER_DEVICE`` and ``LAKER_DTYPE``. The
   latter accepts only the strings ``"float32"`` and ``"float64"``; any
   other value is silently ignored. Environment configuration is applied
   at most once per process (see :data:`_ENV_INIT_DONE`).
3. Direct calls to :func:`set_default_device` / :func:`set_default_dtype`.

Thread-safety: the module-level sentinel :data:`_ENV_INIT_DONE` is read
and written non-atomically, so concurrent calls to
:func:`_init_from_env` from multiple threads at startup could in theory
double-apply the environment overrides. This is benign (the overrides are
idempotent) and no user-visible state is corrupted, but callers that need
strict thread-safety at import time should set defaults explicitly via
the public setters.
"""

import logging
import os
import platform
from typing import Optional, Union

# Environment variable controlling torch.compile behaviour.
# Set LAKER_COMPILE_MODE to one of "default", "reduce-overhead", or
# "max-autotune" to enable compilation.  Unset or empty => no compile.
_LAKER_COMPILE_MODE = os.environ.get("LAKER_COMPILE_MODE", "")

import torch

logger = logging.getLogger(__name__)

# Package-wide defaults. These are mutable module-level globals; mutations
# are intentional and surfaced via the public setters below.
DEFAULT_DEVICE: torch.device = torch.device("cpu")
DEFAULT_DTYPE: torch.dtype = torch.float32

# Sentinel ensuring that environment-driven configuration runs at most once
# per process. Read/written non-atomically; see module docstring.
_ENV_INIT_DONE = False


def _init_from_env() -> None:
    """Apply ``LAKER_DEVICE`` / ``LAKER_DTYPE`` environment variables once.

    Reads two environment variables and routes them through the public
    setters (which also emit an informational log message):

    * ``LAKER_DEVICE``: any string accepted by :func:`torch.device`. If
      unset or empty, the existing default is preserved.
    * ``LAKER_DTYPE``: must be exactly ``"float32"`` or ``"float64"``.
      Any other value is silently ignored to avoid surprising type
      promotions.

    The function is idempotent within a process: a module-level
    :data:`_ENV_INIT_DONE` sentinel guarantees the environment is parsed
    at most once, even if many public API entry points trigger it.
    Subsequent calls become no-ops without re-reading the environment.

    Note:
        The non-atomic check-then-set of :data:`_ENV_INIT_DONE` is safe
        in practice because the body is idempotent; see the module-level
        thread-safety note.

    """
    global _ENV_INIT_DONE
    if _ENV_INIT_DONE:
        return
    _ENV_INIT_DONE = True
    env_device = os.environ.get("LAKER_DEVICE")
    if env_device:
        # Route through the public setter so the resulting log message
        # and validation logic remain consistent with manual calls.
        set_default_device(env_device)
    env_dtype = os.environ.get("LAKER_DTYPE")
    if env_dtype == "float32":
        # Same rationale as for ``LAKER_DEVICE``: route through the
        # public setter for consistency.
        set_default_dtype(torch.float32)
    elif env_dtype == "float64":
        set_default_dtype(torch.float64)

    # TF32 matmul acceleration (Ampere+ GPUs only, no-op on other hardware).
    if os.environ.get("LAKER_TF32", "1") != "0":
        torch.set_float32_matmul_precision("high")


def get_default_device() -> torch.device:
    """Return the current default compute device.

    Triggers a one-shot environment lookup on the first call of the
    process (see :func:`_init_from_env`); subsequent calls just return
    the cached module-level constant.

    Returns:
        The currently configured default :class:`torch.device`.

    """
    _init_from_env()
    return DEFAULT_DEVICE


def set_default_device(
    device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """Set and return the default compute device.

    If ``device`` is ``None``, the auto-selection order is
    CUDA → Apple MPS → CPU. Otherwise ``device`` is normalised via
    :class:`torch.device` so callers may pass either a string
    (``"cuda:0"``) or an existing :class:`torch.device`.

    Args:
        device: Desired device string or :class:`torch.device` instance.
            If ``None``, auto-select from the available backends.

    Returns:
        The resolved :class:`torch.device` actually stored as the
        package default. Useful for chaining: ``dev = set_default_device()``.

    Side effects:
        Mutates :data:`DEFAULT_DEVICE` and emits an
        ``INFO`` log entry via :mod:`logging`.

    """
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
    """Return the current default floating-point dtype.

    Returns:
        The currently configured default :class:`torch.dtype`.

    """
    _init_from_env()
    return DEFAULT_DTYPE


def set_default_dtype(dtype: torch.dtype) -> None:
    """Set the default floating-point dtype.

    The dtype is used by every LAKER component when an explicit dtype is
    not provided at construction. Because most kernel computations are
    numerically sensitive, switching to ``torch.float64`` is recommended
    for ill-conditioned problems.

    Args:
        dtype: A :class:`torch.dtype` instance. The function does not
            validate the value; passing ``torch.int64`` is allowed but
            almost certainly a bug.

    Side effects:
        Mutates :data:`DEFAULT_DTYPE` and emits an ``INFO`` log entry.

    """
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype
    logger.info("Default dtype set to %s", dtype)


def maybe_compile(func, mode: str = "reduce-overhead"):
    """Optionally compile a function with :func:`torch.compile`.

    Compilation only happens when ``LAKER_COMPILE_MODE`` is set in the
    environment — the *mode* parameter is a default that applies when the
    env var is present but empty-valued (edge case).  When the env var is
    unset entirely the original callable is returned unchanged, keeping
    compilation fully opt-in.

    When active, deprecation warnings from ``torch.compile`` are suppressed.

    Args:
        func: Callable to compile.
        mode: Fallback compilation mode when ``LAKER_COMPILE_MODE`` is
            set to an empty string.

    Returns:
        Compiled function (opt-in) or original callable.

    """
    # Compile is opt-in: skip when the env var is not present.
    if "LAKER_COMPILE_MODE" not in os.environ:
        return func
    if not hasattr(torch, "compile"):
        return func
    compile_mode = _LAKER_COMPILE_MODE or mode
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return torch.compile(func, mode=compile_mode)


def to_tensor(
    data,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert array-like data to a :class:`torch.Tensor` on the requested device/dtype.

    This is a thin convenience wrapper that:

    * Returns existing tensors unchanged (modulo device/dtype cast).
    * Wraps NumPy arrays via :func:`torch.as_tensor`, sharing memory.
    * Coerces Python lists/tuples to a tensor on the chosen backend.

    Args:
        data: A NumPy ``ndarray``, Python list/tuple, or existing
            :class:`torch.Tensor`.
        device: Target device. Defaults to :func:`get_default_device`.
        dtype: Target dtype. Defaults to :func:`get_default_dtype`.

    Returns:
        A :class:`torch.Tensor` on the requested device/dtype.

    """
    if device is None:
        device = get_default_device()
    if dtype is None:
        dtype = get_default_dtype()
    if isinstance(data, torch.Tensor):
        # ``.to`` is a no-op (cheap) when device and dtype already match,
        # so we always normalise here for predictable downstream behaviour.
        return data.to(device=device, dtype=dtype)
    # ``torch.as_tensor`` avoids a copy when ``data`` is a NumPy array
    # that already matches the requested dtype.
    return torch.as_tensor(data, device=device, dtype=dtype)
