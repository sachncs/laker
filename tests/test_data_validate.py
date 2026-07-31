"""Validation helper tests for ``laker.base.Base``.

Covers ``Base.validate_split_indices``, ``Base.validate_inputs``,
and ``Base.validate_target`` — the public validation helpers that
gate ``Laker.fit`` and other entry points.
"""

from __future__ import annotations

import pytest
import torch

from laker.base import Base


def test_validate_split_indices_zero_n():
    with pytest.raises(ValueError, match="at least 2"):
        Base.validate_split_indices(n=1, val_fraction=0.2)


def test_validate_split_indices_bad_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        Base.validate_split_indices(n=10, val_fraction=0.0)
    with pytest.raises(ValueError, match="val_fraction"):
        Base.validate_split_indices(n=10, val_fraction=1.0)


def test_validate_split_indices_basic():
    n_train, n_val = Base.validate_split_indices(n=100, val_fraction=0.2)
    assert n_train + n_val == 100
    assert n_val == 20


def test_validate_split_indices_edge_clamp():
    # Tiny n -> n_val must never exceed n-1.
    n_train, n_val = Base.validate_split_indices(n=2, val_fraction=0.5)
    assert n_train == 1
    assert n_val == 1


def test_validate_inputs_rejects_extra_dim():
    x = torch.randn(20, 3, 4)
    with pytest.raises(ValueError, match="2-D"):
        Base.validate_inputs(x)


def test_validate_inputs_rejects_non_finite():
    x = torch.randn(20, 3)
    x[5, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        Base.validate_inputs(x)


def test_validate_inputs_rejects_empty():
    x = torch.zeros(0, 3)
    with pytest.raises(ValueError, match="at least one row"):
        Base.validate_inputs(x)


def test_validate_target_accepts_n_shape():
    y = torch.randn(20)
    out = Base.validate_target(y)
    assert out.shape == (20,)


def test_validate_target_accepts_n1_shape():
    y = torch.randn(20, 1)
    out = Base.validate_target(y)
    assert out.shape == (20,)


def test_validate_target_rejects_scalar():
    y = torch.tensor(1.5)
    with pytest.raises(ValueError, match="scalar"):
        Base.validate_target(y)


def test_validate_target_rejects_multi_dim():
    y = torch.randn(20, 3, 2)
    with pytest.raises(ValueError, match="1-D"):
        Base.validate_target(y)
