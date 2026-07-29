"""Tests for streaming updates."""

import pytest
import torch

from laker.models import LAKERRegressor
from laker.streaming import StreamingUpdater


def test_partial_fit_before_fit():
    """partial_fit before fit should raise RuntimeError."""
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    updater = StreamingUpdater(model.core)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        updater.partial_fit(model, torch.randn(3, 2), torch.randn(3))


def test_partial_fit():
    """partial_fit should append data and update coefficients."""
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    x = torch.rand(20, 2)
    y = torch.randn(20)
    model.fit(x, y)
    n_orig = model.embeddings.shape[0]

    updater = StreamingUpdater(model.core)
    x_new = torch.rand(5, 2)
    y_new = torch.randn(5)
    updater.partial_fit(model, x_new, y_new)

    assert model.embeddings.shape[0] == n_orig + 5
    assert model.alpha.shape[0] == n_orig + 5


def test_partial_fit_invalid_shapes():
    """partial_fit should validate input shapes."""
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    model.fit(torch.rand(20, 2), torch.randn(20))
    updater = StreamingUpdater(model.core)
    with pytest.raises(ValueError, match="x_new must be 2-D"):
        updater.partial_fit(model, torch.randn(3), torch.randn(3))
    with pytest.raises(ValueError, match="y_new must be 1-D"):
        updater.partial_fit(model, torch.randn(3, 2), torch.randn(3, 1))


def test_partial_fit_rebuild_threshold():
    """partial_fit should raise when rebuild_threshold exceeded."""
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    model.fit(torch.rand(20, 2), torch.randn(20))
    updater = StreamingUpdater(model.core)
    with pytest.raises(RuntimeError, match="rebuild threshold exceeded"):
        updater.partial_fit(model, torch.randn(100, 2), torch.randn(100), rebuild_threshold=50)
