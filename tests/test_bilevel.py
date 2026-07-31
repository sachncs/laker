"""Bilevel tune contract.

Laker.tune runs the bilevel inner loop plus a log-space regularisation
search and writes back the best validation score. These tests pin
down that the public ``regularization`` attribute actually moves
when tune is called.
"""

from __future__ import annotations

import torch

from laker import Laker


def test_tune_changes_regularization():
    """``tune`` should not be a no-op; ``regularization`` must change."""
    torch.manual_seed(0)
    n = 60
    x = torch.rand(n, 2) * 10
    y = torch.sin(x.sum(-1)) + 0.05 * torch.randn(n)
    perm = torch.randperm(n)
    n_val = n // 5
    x_train, x_val = x[perm[n_val:]], x[perm[:n_val]]
    y_train, y_val = y[perm[n_val:]], y[perm[:n_val]]

    model = Laker(regularization=0.1, embedding_dim=4)
    model.fit(x_train, y_train)
    before = model.regularization

    model.tune(x_train, y_train, x_val, y_val, lr=5e-2, epochs=15, patience=10)

    after = model.regularization
    assert abs(before - after) > 1e-6, f"tune was a no-op (regularization stayed at {before})"


def test_tune_run_smoke():
    """End-to-end smoke: tune runs without errors, returns the model."""
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2)
    y = torch.sin(x.sum(-1) / 5)
    model = Laker(regularization=1e-2, embedding_dim=4)
    model.tune(x[:20], y[:20], x[20:], y[20:], lr=1e-2, epochs=3)
    assert model.coef_ is not None
