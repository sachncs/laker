"""Tests for :mod:`laker.plot`."""

import torch

from laker.plot import Plot


class TestImage:
    def test_image_shape(self):
        preds = torch.arange(16.0).reshape(16)
        img = Plot.image(preds, 4)
        assert img.shape == (4, 4)

    def test_image_values_preserved(self):
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        img = Plot.image(preds, 2)
        import numpy as np

        np.testing.assert_array_equal(img, np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestField:
    def test_field_requires_matplotlib(self):
        preds = torch.zeros(4)
        try:
            fig, ax = Plot.field(preds, 2)
        except ImportError as exc:
            assert "matplotlib" in str(exc).lower()


class TestConvergence:
    def test_convergence_requires_matplotlib(self):
        try:
            Plot.convergence([[0.1, 0.01, 0.001]], labels=["a"])
        except ImportError as exc:
            assert "matplotlib" in str(exc).lower()

    def test_convergence_accepts_multiple_series(self):
        try:
            Plot.convergence([[0.1, 0.01], [0.2, 0.02]], labels=["a", "b"])
        except ImportError:
            pass  # matplotlib may not be available
