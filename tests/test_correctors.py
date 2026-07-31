"""Behavioural + precision tests for the residual corrector.

The corrector is a two-layer Tanh MLP that lives under the legacy
``laker.models`` API. It is consumed by ``LAKERRegressor`` via the
``fit_residual_corrector`` workflow. Each test asserts a real
contract: output shape, deterministic parameters, dropout behaviour
during eval, and finite outputs.
"""

from __future__ import annotations

import torch

from laker.correctors import ResidualCorrector


# ---------------------------------------------------------------------------
# Shape and parameter storage.
# ---------------------------------------------------------------------------
def test_forward_shape_batch():
    """A batched (n, d) input produces a (n, output_dim) output."""
    m = ResidualCorrector(input_dim=2, output_dim=1)
    x = torch.rand(10, 2)
    out = m(x)
    assert out.shape == (10, 1)


def test_forward_shape_single():
    """A single (d,) input is treated as one sample and produces a
    (output_dim,) output."""
    m = ResidualCorrector(input_dim=3, output_dim=2)
    x = torch.rand(3)
    out = m(x)
    assert out.shape == (2,)


def test_forward_shape_multi_output():
    """``output_dim > 1`` produces one column per output dimension."""
    m = ResidualCorrector(input_dim=2, output_dim=4)
    x = torch.rand(7, 2)
    out = m(x)
    assert out.shape == (7, 4)


# ---------------------------------------------------------------------------
# Constructor parameters round-trip via attributes.
# ---------------------------------------------------------------------------
def test_constructor_parameters_round_trip():
    """The constructor arguments are stored on self as attributes."""
    m = ResidualCorrector(input_dim=3, output_dim=2, hidden_dim=64, dropout=0.2)
    assert m.input_dim == 3
    assert m.output_dim == 2
    assert m.hidden_dim == 64
    # dropout is consumed inside nn.Dropout, not stored; verify
    # the network's dropout module carries the configured value.
    dropout_module = m.net[2]
    assert isinstance(dropout_module, torch.nn.Dropout)
    assert dropout_module.p == 0.2


def test_constructor_defaults():
    """Defaults: output_dim=1, hidden_dim=32, dropout=0.1."""
    m = ResidualCorrector(input_dim=2)
    assert m.input_dim == 2
    assert m.output_dim == 1
    assert m.hidden_dim == 32
    assert m.net[2].p == 0.1


# ---------------------------------------------------------------------------
# Determinism: same seed produces the same output.
# ---------------------------------------------------------------------------
def test_forward_is_deterministic_in_eval_mode():
    """In ``eval()`` mode (dropout disabled) the corrector is fully
    deterministic. Two forward passes with the same input agree
    exactly.
    """
    torch.manual_seed(0)
    m = ResidualCorrector(input_dim=3, hidden_dim=8)
    m.eval()
    x = torch.randn(4, 3)
    y1 = m(x)
    y2 = m(x)
    torch.testing.assert_close(y1, y2)


def test_forward_changes_with_seed_in_train_mode():
    """In ``train()`` mode dropout is active, so two independent
    forward passes with different random states produce different
    outputs (with overwhelming probability).
    """
    torch.manual_seed(0)
    m = ResidualCorrector(input_dim=4, hidden_dim=16, dropout=0.5)
    m.train()
    x = torch.randn(8, 4)
    y1 = m(x)
    torch.manual_seed(1)
    # Re-instantiate so the dropout RNG state restarts fresh.
    torch.manual_seed(0)
    m2 = ResidualCorrector(input_dim=4, hidden_dim=16, dropout=0.5)
    m2.train()
    y2 = m2(x)
    # dropout=0.5 with non-trivial input should produce distinct outputs.
    assert not torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# Numerical behaviour: outputs are finite, gradient flows through.
# ---------------------------------------------------------------------------
def test_forward_outputs_are_finite():
    """``forward`` returns finite values for typical inputs."""
    torch.manual_seed(0)
    m = ResidualCorrector(input_dim=2, hidden_dim=8, dropout=0.0)
    m.eval()
    x = torch.randn(32, 2) * 5.0  # moderately-scaled inputs
    out = m(x)
    assert torch.isfinite(out).all()


def test_backward_propagates_gradients():
    """A scalar loss derived from the output back-propagates through
    every parameter of the network.
    """
    torch.manual_seed(0)
    m = ResidualCorrector(input_dim=3, hidden_dim=8, dropout=0.0)
    m.train()
    x = torch.randn(4, 3)
    y = m(x)
    y.sum().backward()
    grads_present = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None)
    grads_nonzero = sum(
        1
        for p in m.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grads_present > 0
    assert grads_nonzero == grads_present


def test_eval_mode_disables_dropout():
    """In ``eval()`` the corrector output does not depend on random
    state: two forward passes on the same model instance give
    identical outputs even when dropout is configured.
    """
    torch.manual_seed(0)
    m = ResidualCorrector(input_dim=2, hidden_dim=8, dropout=0.9)
    m.eval()
    x = torch.randn(16, 2)
    y_first = m(x).clone()
    y_second = m(x).clone()
    torch.testing.assert_close(y_first, y_second)


# ---------------------------------------------------------------------------
# Architectural sanity: the network has exactly two Linear modules.
# ---------------------------------------------------------------------------
def test_network_has_two_linear_layers():
    """The architecture is documented as a two-layer MLP; we verify
    it actually contains exactly two ``nn.Linear`` modules."""
    m = ResidualCorrector(input_dim=2, hidden_dim=8, output_dim=1)
    linears = [layer for layer in m.net if isinstance(layer, torch.nn.Linear)]
    assert len(linears) == 2
    assert linears[0].in_features == 2
    assert linears[0].out_features == 8
    assert linears[1].in_features == 8
    assert linears[1].out_features == 1
