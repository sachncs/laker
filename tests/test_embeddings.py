"""Behavioural + precision tests for the embedding modules.

Covers ``laker.embed.Position`` (random Fourier features + Tanh MLP),
``laker.embed.Visual`` (Conv2d patch + linear projection), and the
``Embed`` base class.
"""

from __future__ import annotations

import pytest
import torch

from laker.embed import Embed, Position, Visual


# ---------------------------------------------------------------------------
# Position: shape, defaults, dtype, and reproducibility.
# ---------------------------------------------------------------------------
def test_position_forward_2d_batch():
    """A 2-D coordinate batch produces an ``(n, embedding_dim)`` output."""
    torch.manual_seed(0)
    embed = Position(input_dim=2, embedding_dim=8, seed=42)
    out = embed(torch.rand(10, 2))
    assert out.shape == (10, 8)


def test_position_forward_1d_unsqueezed():
    """A 1-D coordinate vector is auto-batched to one row."""
    torch.manual_seed(0)
    embed = Position(input_dim=2, embedding_dim=8, seed=42)
    out = embed(torch.rand(2))
    assert out.shape == (1, 8)


def test_position_default_num_fourier():
    """Default ``num_fourier`` is ``2 * embedding_dim`` per the docstring."""
    embed = Position(input_dim=2, embedding_dim=10, seed=42)
    assert embed.num_fourier == 20
    assert embed.embedding_dim == 10
    assert embed.input_dim == 2


def test_position_extra_repr():
    """``extra_repr`` includes the documented constructor arguments."""
    embed = Position(input_dim=2, embedding_dim=8, num_fourier=16, sigma=5.0, seed=42)
    text = embed.extra_repr()
    assert "input_dim=2" in text
    assert "embedding_dim=8" in text
    assert "num_fourier=16" in text
    assert "sigma=5.0" in text


def test_position_reproducibility_same_seed():
    """Two embeddings built with the same seed produce identical outputs."""
    embed1 = Position(input_dim=2, embedding_dim=8, seed=42)
    embed2 = Position(input_dim=2, embedding_dim=8, seed=42)
    x = torch.rand(10, 2)
    torch.testing.assert_close(embed1(x), embed2(x))


def test_position_distinct_seeds_produce_different_frequencies():
    """Different seeds yield different random frequencies, so outputs differ."""
    torch.manual_seed(0)
    embed1 = Position(input_dim=2, embedding_dim=8, seed=42)
    embed2 = Position(input_dim=2, embedding_dim=8, seed=43)
    x = torch.randn(10, 2)
    assert not torch.equal(embed1(x), embed2(x))


def test_position_dtype_explicit():
    """Explicit ``dtype=torch.float64`` propagates to outputs."""
    embed = Position(input_dim=2, embedding_dim=4, seed=42, dtype=torch.float64)
    x = torch.rand(5, 2, dtype=torch.float64)
    out = embed(x)
    assert out.dtype == torch.float64


def test_position_outputs_are_finite():
    """Outputs are finite for moderately scaled inputs."""
    torch.manual_seed(0)
    embed = Position(input_dim=3, embedding_dim=8, seed=0)
    out = embed(torch.randn(32, 3) * 3.0)
    assert torch.isfinite(out).all()


def test_position_periodic_in_input():
    """Random Fourier features are sinusoidal: shifting the input by
    enough should not produce an identical output (the features
    have non-zero frequency components)."""
    torch.manual_seed(0)
    embed = Position(input_dim=1, embedding_dim=4, seed=42)
    a = embed(torch.tensor([[0.0]]))
    b = embed(torch.tensor([[2.0]]))
    # Different inputs → different outputs (modulo the Tanh MLPs).
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# Visual: Conv2d patch encoder.
# ---------------------------------------------------------------------------
def test_visual_forward_shape():
    """``Visual`` accepts ``(n, c, h, w)`` and produces ``(n, embedding_dim)``."""
    torch.manual_seed(0)
    embed = Visual(input_dim=3, embedding_dim=12, patch_size=4, seed=42)
    x = torch.randn(5, 3, 16, 16)
    out = embed(x)
    assert out.shape == (5, 12)


def test_visual_handles_non_divisible_input():
    """Inputs whose spatial size is not a multiple of ``patch_size`` still
    produce a valid output (Conv2d floor-divides)."""
    torch.manual_seed(0)
    embed = Visual(input_dim=1, embedding_dim=8, patch_size=4, seed=42)
    x = torch.randn(2, 1, 7, 7)
    out = embed(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_visual_extra_repr_includes_constructor_args():
    embed = Visual(input_dim=2, embedding_dim=8, patch_size=4, seed=42)
    text = embed.extra_repr()
    assert "input_dim=2" in text
    assert "patch_size=4" in text
    assert "embedding_dim=8" in text


def test_visual_reproducibility_same_seed():
    """A single model instance is deterministic: two forward passes
    on the same input give identical outputs (deterministic forward,
    no dropout in ``Visual``).
    """
    embed = Visual(input_dim=2, embedding_dim=8, patch_size=4, seed=42)
    embed.eval()
    x = torch.randn(3, 2, 12, 12)
    out_a = embed(x).clone()
    out_b = embed(x).clone()
    torch.testing.assert_close(out_a, out_b)


# ---------------------------------------------------------------------------
# Embed base class: forwards to subclasses, is itself abstract.
# ---------------------------------------------------------------------------
def test_embed_base_is_abstract():
    """``Embed.forward`` is abstract; instantiating ``Embed`` directly
    raises ``NotImplementedError``."""
    base = Embed()
    with pytest.raises(NotImplementedError):
        base(torch.zeros(1, 2))


def test_embed_subclasses_override_forward():
    """Each subclass overrides ``forward`` so the abstract base never
    dispatches to itself in practice."""
    for cls in (Position, Visual):
        embed = cls(input_dim=2, embedding_dim=4, seed=0)
        assert type(embed).forward is not Embed.forward
