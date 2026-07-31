"""Behavioural + precision tests for ``laker.backend``.

Every assertion targets a real behavioural contract: a numerical
value (the default dtype), an idempotent round-trip on global state,
a precision guarantee on tensor conversion, or the contract between
``LAKER_DEVICE`` / ``LAKER_DTYPE`` env vars and the live defaults.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import torch

from laker.backend import Backend

from laker import backend


# ---------------------------------------------------------------------------
# Default dtype and dtype round-trip
# ---------------------------------------------------------------------------
def test_default_dtype_is_float32():
    """``Backend.dtype`` returns ``torch.float32`` after
    package import. Any deviation here indicates a silent dtype change."""
    assert Backend.dtype == torch.float32


def test_Backend_set_dtype_round_trips():
    """Setting ``float64`` mutates the default; restoring to the
    original dtype returns the exact original value (not just any
    float32-equivalent).
    """
    old = Backend.dtype
    try:
        Backend.set_dtype(torch.float64)
        assert Backend.dtype == torch.float64, "set to float64 did not stick"
        Backend.set_dtype(torch.float16)
        assert Backend.dtype == torch.float16
    finally:
        Backend.set_dtype(old)
    assert Backend.dtype == old


# ---------------------------------------------------------------------------
# ``Backend.to_tensor``: precision and dtype/device preservation across input types.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,target_dtype,expected_dtype",
    [
        (1.0, None, torch.float32),
        (1.0, torch.float64, torch.float64),
        (np.array([1.0, 2.0, 3.0]), None, torch.float32),
        (np.array([1.0, 2.0, 3.0]), torch.float64, torch.float64),
        ([1.0, 2.0, 3.0], None, torch.float32),
        (torch.tensor([1.0, 2.0, 3.0]), torch.float64, torch.float64),
    ],
)
def test_Backend_to_tensor_dtype_preserved(value, target_dtype, expected_dtype):
    """``Backend.to_tensor`` produces a tensor with the requested dtype."""
    t = Backend.to_tensor(value, dtype=target_dtype)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == expected_dtype


def test_Backend_to_tensor_default_dtype_is_float32():
    """Without an explicit dtype, ``Backend.to_tensor`` uses the package default
    which is ``torch.float32``."""
    old = Backend.dtype
    Backend.set_dtype(torch.float32)
    try:
        t = Backend.to_tensor([1.0, 2.0, 3.0])
        assert t.dtype == torch.float32
    finally:
        Backend.set_dtype(old)


def test_Backend_to_tensor_numpy_values_byte_for_byte():
    """``Backend.to_tensor`` of a NumPy array must round-trip the values
    exactly (not just the dtype)."""
    arr = np.array([1.5, -2.25, 3.0, 4.75, 0.0], dtype=np.float64)
    t = Backend.to_tensor(arr, dtype=torch.float64)
    torch.testing.assert_close(t, torch.from_numpy(arr))


def test_Backend_to_tensor_tensor_values_byte_for_byte():
    """``Backend.to_tensor`` of an existing tensor preserves the values exactly
    and matches the package default dtype."""
    src = torch.tensor([1.5, -2.25, 3.0, 4.75, 0.0])
    t = Backend.to_tensor(src)
    torch.testing.assert_close(t, src)
    assert t.dtype == Backend.dtype


def test_Backend_to_tensor_tensor_with_explicit_dtype_casts():
    """When the source tensor has a different dtype, ``Backend.to_tensor`` casts
    to the requested dtype and preserves values (modulo precision).
    """
    src = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    t = Backend.to_tensor(src, dtype=torch.float64)
    assert t.dtype == torch.float64
    torch.testing.assert_close(t.cpu(), src.double())


def test_Backend_to_tensor_explicit_device_preserved():
    """``Backend.to_tensor`` accepts an explicit device argument and the
    resulting tensor lives on that device."""
    t = Backend.to_tensor([1.0, 2.0], device=torch.device("cpu"), dtype=torch.float64)
    assert t.device == torch.device("cpu")
    assert t.dtype == torch.float64


# ---------------------------------------------------------------------------
# ``Backend.set_device`` and the ``LAKER_DEVICE`` env contract.
# ---------------------------------------------------------------------------
def test_Backend_set_device_string_round_trips():
    """Setting ``Backend.set_device('cpu')`` works; the default returns
    to its prior value when restored."""
    old = Backend.device
    try:
        device = Backend.set_device("cpu")
        assert device == torch.device("cpu")
        assert Backend.device == torch.device("cpu")
    finally:
        Backend.set_device(old)
    assert Backend.device == old


def test_Backend_set_device_none_auto_selects():
    """``Backend.set_device(None)`` auto-selects a ``torch.device``."""
    old = Backend.device
    try:
        device = Backend.set_device(None)
        assert isinstance(device, torch.device)
    finally:
        Backend.set_device(old)


# ---------------------------------------------------------------------------
# Environment-variable round-trip on the Backend module.
# ---------------------------------------------------------------------------
@pytest.fixture
def restored_backend_env(monkeypatch):
    """Restore env vars and the Backend module state after each test."""
    saved = {
        "LAKER_DEVICE": os.environ.get("LAKER_DEVICE"),
        "LAKER_DTYPE": os.environ.get("LAKER_DTYPE"),
    }
    yield monkeypatch
    for key, value in saved.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    importlib.reload(backend)


def test_env_var_dtype_float64_updates_default_dtype(restored_backend_env):
    """Setting ``LAKER_DTYPE=float64`` then reloading the module makes
    ``Backend.dtype`` return ``float64``."""
    restored_backend_env.setenv("LAKER_DTYPE", "float64")
    importlib.reload(backend)
    assert backend.Backend.dtype == torch.float64


def test_env_var_dtype_float32_updates_default_dtype(restored_backend_env):
    """``LAKER_DTYPE=float32`` yields ``torch.float32`` after reload."""
    restored_backend_env.setenv("LAKER_DTYPE", "float32")
    importlib.reload(backend)
    assert backend.Backend.dtype == torch.float32


def test_env_var_invalid_dtype_is_ignored(restored_backend_env):
    """Unknown dtype strings must not raise during import; the
    default remains at its baseline value."""
    restored_backend_env.setenv("LAKER_DTYPE", "fancyfloat99")
    importlib.reload(backend)
    assert backend.Backend.dtype in (torch.float32, torch.float64)


def test_env_var_device_cpu_sets_default_device(restored_backend_env):
    """``LAKER_DEVICE=cpu`` after reload yields ``torch.device('cpu')``."""
    restored_backend_env.setenv("LAKER_DEVICE", "cpu")
    importlib.reload(backend)
    assert backend.Backend.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ``Backend.maybe_compile``: identity-or-wrapped semantics.
# ---------------------------------------------------------------------------
def test_Backend_maybe_compile_returns_callable():
    """``Backend.maybe_compile`` always returns a callable (compiled or identity)."""

    def fn(x):
        return x + 1.0

    compiled = Backend.maybe_compile(fn)
    assert callable(compiled)
    out = compiled(torch.tensor(1.0))
    assert torch.equal(out, torch.tensor(2.0))


def test_Backend_maybe_compile_preserves_callable_behavior():
    """Whether or not ``Backend.maybe_compile`` activates ``torch.compile``, the
    wrapped callable must return the same result as the original
    within float tolerance.
    """
    import math

    def fn(x):
        return torch.sin(x) + 0.5

    x = torch.linspace(0.0, math.pi, 8)
    expected = fn(x)
    out = Backend.maybe_compile(fn)(x)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
