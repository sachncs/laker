"""Tests for residual corrector models."""

import torch

from laker.correctors import ResidualCorrector


def test_corrector_forward_shape():
    """ResidualCorrector forward should return correct shape."""
    m = ResidualCorrector(input_dim=2)
    x = torch.rand(10, 2)
    out = m(x)
    assert out.shape == (10, 1)


def test_corrector_forward_multi_output():
    """ResidualCorrector with multiple outputs should work."""
    m = ResidualCorrector(input_dim=2, output_dim=3)
    x = torch.rand(10, 2)
    out = m(x)
    assert out.shape == (10, 3)


def test_corrector_single_input():
    """ResidualCorrector should handle a single input."""
    m = ResidualCorrector(input_dim=2)
    x = torch.rand(2)
    out = m(x)
    assert out.shape == (1,)


def test_corrector_parameters():
    """ResidualCorrector should store constructor parameters."""
    m = ResidualCorrector(input_dim=3, output_dim=1, hidden_dim=64, dropout=0.2)
    assert m.input_dim == 3
    assert m.output_dim == 1
    assert m.hidden_dim == 64
