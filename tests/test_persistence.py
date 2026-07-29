"""Tests for model persistence module."""

import tempfile

import pytest
import torch

from laker.models import LAKERRegressor
from laker.persistence import ModelPersistence


def test_save_and_load_roundtrip():
    """ModelPersistence save/load should roundtrip model state."""
    model = LAKERRegressor(embedding_dim=4, num_probes=20, cccp_max_iter=10, verbose=False)
    model.fit(torch.rand(40, 2), torch.randn(40))

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        ModelPersistence.save(model, path)
        loaded = ModelPersistence.load(path)

        assert torch.allclose(loaded.alpha, model.alpha)
        assert torch.allclose(loaded.embeddings, model.embeddings)
        assert loaded.embedding_dim == model.embedding_dim
        assert loaded.lambda_reg == model.lambda_reg

        x_test = torch.rand(10, 2) * 100.0
        torch.testing.assert_close(loaded.predict(x_test), model.predict(x_test), atol=1e-5, rtol=1e-5)
    finally:
        import os
        os.unlink(path)


def test_load_nonexistent():
    """Loading a nonexistent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ModelPersistence.load("/nonexistent/path/model.pt")


def test_load_invalid_state():
    """Loading a file with invalid state should raise KeyError."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save({"invalid": "data"}, path)

    try:
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            ModelPersistence.load(path)
    finally:
        import os
        os.unlink(path)
