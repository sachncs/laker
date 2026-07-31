"""Tests for :mod:`laker.corrector`."""

import torch

from laker.corrector import Corrector


class TestCorrector:
    def test_shape_1d(self):
        c = Corrector(input_dim=2, output_dim=1)
        x = torch.randn(10, 2)
        out = c(x)
        assert out.shape == (10, 1)

    def test_shape_2d_output(self):
        c = Corrector(input_dim=3, output_dim=2)
        x = torch.randn(7, 3)
        out = c(x)
        assert out.shape == (7, 2)

    def test_attrs(self):
        c = Corrector(input_dim=2, output_dim=1, hidden_dim=64, dropout=0.2)
        assert c.input_dim == 2
        assert c.output_dim == 1
        assert c.hidden_dim == 64
        assert isinstance(next(m for m in c.net.modules() if isinstance(m, torch.nn.Dropout)), torch.nn.Dropout)
        assert c.net[2].p == 0.2

    def test_defaults(self):
        c = Corrector(input_dim=2)
        assert c.hidden_dim == 32
        assert c.output_dim == 1

    def test_eval_deterministic(self):
        c = Corrector(input_dim=2)
        c.eval()
        x = torch.randn(5, 2)
        torch.testing.assert_close(c(x), c(x))

    def test_train_mode_changes_output(self):
        c = Corrector(input_dim=2, dropout=0.5)
        torch.manual_seed(0)
        c.train()
        x = torch.randn(5, 2)
        y1 = c(x)
        y2 = c(x)
        assert not torch.equal(y1, y2)

    def test_gradients_flow(self):
        c = Corrector(input_dim=2)
        x = torch.randn(3, 2, requires_grad=True)
        out = c(x).sum()
        out.backward()
        grads_present = sum(1 for p in c.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads_present > 0

    def test_finite(self):
        c = Corrector(input_dim=2)
        x = torch.randn(10, 2)
        assert torch.isfinite(c(x)).all()


class TestArchitecture:
    def test_two_linear_layers(self):
        c = Corrector(input_dim=2, hidden_dim=8)
        linears = [m for m in c.net.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 2
        assert linears[0].in_features == 2
        assert linears[0].out_features == 8
        assert linears[1].in_features == 8
        assert linears[1].out_features == 1