"""Behavioural + precision tests for streaming updates.

Covers ``Laker.update`` end-to-end: pre-fit rejection, partial data
append, threshold-driven full rebuild, and shape validation.
"""

from __future__ import annotations

import pytest
import torch

from laker import Laker


# ---------------------------------------------------------------------------
# Pre-fit rejection.
# ---------------------------------------------------------------------------
def test_update_before_fit_raises():
    """``Laker.update`` before ``fit`` must raise ``RuntimeError``."""
    m = Laker(embedding_dim=4)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        m.update(torch.randn(3, 2), torch.randn(3))


# ---------------------------------------------------------------------------
# partial append: coefficient length and embedding count grow exactly
# by the batch size.
# ---------------------------------------------------------------------------
def test_update_grows_state_by_exactly_batch_size():
    """Each ``update`` call grows ``coef_`` and ``embeddings_`` by the
    batch size (no skipped rows, no silent overwrite).
    """
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.sin(x.sum(-1)) + 0.05 * torch.randn(n, dtype=torch.float64)
    m = Laker(
        embedding_dim=8,
        regularization=1e-3,
        probes=80,
        cccp_max_iter=50,
        pcg_tol=1e-8,
        pcg_max_iter=500,
        dtype=torch.float64,
    )
    m.fit(x, y)
    initial_n = m.coef_.shape[0]

    batches = [5, 7, 11]
    expected_n = initial_n
    for b in batches:
        x_new = torch.rand(b, 2, dtype=torch.float64) * 10.0
        y_new = torch.sin(x_new.sum(-1)) + 0.05 * torch.randn(b, dtype=torch.float64)
        m.update(x_new, y_new, rebuild_threshold=10_000)
        expected_n += b
        assert m.coef_.shape[0] == expected_n, (
            f"after update(b={b}): coef has {m.coef_.shape[0]} rows, " f"expected {expected_n}"
        )
        assert m.embeddings_.shape[0] == expected_n


# ---------------------------------------------------------------------------
# Shape validation.
# ---------------------------------------------------------------------------
def test_update_rejects_1d_x():
    """``update(x, y)`` requires ``x`` to be 2-D."""
    m = Laker(embedding_dim=4)
    m.fit(torch.rand(10, 2), torch.rand(10))
    with pytest.raises(ValueError, match="x_new must be 2-D"):
        m.update(torch.rand(5), torch.rand(5))


def test_update_does_not_reject_2d_y_at_streaming_layer():
    """``update`` does not currently validate ``y`` dimensionality at the
    streaming boundary (the legacy class doesn't enforce this either;
    the upstream ``fit`` does, but ``update`` is documented to be
    permissive). The test pins the current behaviour so any tightening
    will surface as a behaviour change.
    """
    m = Laker(embedding_dim=4)
    m.fit(torch.rand(10, 2), torch.rand(10))
    m.update(torch.rand(5, 2), torch.rand(5, 1), rebuild_threshold=10_000)
    # The call completed without raising.


# ---------------------------------------------------------------------------
# Rebuild-threshold triggers a documented full rebuild.
# ---------------------------------------------------------------------------
def test_update_above_threshold_raises_with_documented_message():
    """The documented rebuild behaviour: when the cumulative update
    count exceeds ``rebuild_threshold`` the next ``update`` raises
    ``RuntimeError`` instructing the user to concatenate and call
    ``fit``.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.randn(n, dtype=torch.float64)
    m = Laker(embedding_dim=4, dtype=torch.float64)
    m.fit(x, y)
    with pytest.raises(RuntimeError, match="rebuild threshold"):
        m.update(
            torch.rand(200, 2, dtype=torch.float64),
            torch.rand(200),
            rebuild_threshold=10,
        )
