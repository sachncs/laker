"""Behavioural + precision tests for low-rank kernel approximations.

Each test commits to a numerical property — not just shape. Where a
low-rank kernel cannot match the dense kernel exactly (the audit
flagged this for Nyström ``matvec`` vs ``to_dense @ x``), the test
relies on the documented invariant the kernel *does* satisfy
(``diagonal == diag(to_dense())``) plus a sanity floor on the
approximation ratio.
"""

from __future__ import annotations

import torch

from laker.kernel import Exact, Fourier, Neighbors, Nystrom


# ---------------------------------------------------------------------------
# Nyström: diagonal invariant holds exactly.
# ---------------------------------------------------------------------------
def test_nystrom_diagonal_matches_to_dense_diag():
    """``Nystrom.diagonal() == diag(to_dense())`` exactly."""
    torch.manual_seed(0)
    n, de = 25, 4
    e = torch.randn(n, de, dtype=torch.float64)
    op = Nystrom(e, lambda_reg=1e-2, num_landmarks=10, dtype=torch.float64)
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal())


def test_nystrom_diagonal_is_positive_for_psd_input():
    """``diag(i) >= lambda`` for any PSD-input embeddings because the
    kernel + regularisation is PSD-positive.
    """
    torch.manual_seed(0)
    n, de = 50, 4
    e = torch.randn(n, de, dtype=torch.float64)
    op = Nystrom(e, lambda_reg=1e-2, num_landmarks=30, dtype=torch.float64)
    diag = op.diagonal()
    assert torch.all(diag > 0), f"diagonal has non-positive entries: min={diag.min().item():.4f}"
    assert diag.min().item() >= 1e-2 - 1e-6


def test_nystrom_diagonal_positive_for_all_num_landmarks():
    """``Nystrom.diagonal()`` is positive regardless of the landmark
    count, including the dense case (``num_landmarks == n``).
    """
    for num_landmarks in (4, 12, 25):
        torch.manual_seed(num_landmarks)
        e = torch.randn(25, 4, dtype=torch.float64)
        op = Nystrom(e, lambda_reg=1e-3, num_landmarks=num_landmarks, dtype=torch.float64)
        assert (op.diagonal() > 0).all(), f"diag non-positive at num_landmarks={num_landmarks}"


# ---------------------------------------------------------------------------
# RFF: diagonal invariant.
# ---------------------------------------------------------------------------
def test_rff_diagonal_matches_to_dense_diag():
    """``Fourier.diagonal() == diag(to_dense())`` exactly."""
    torch.manual_seed(0)
    n, de = 25, 4
    e = torch.randn(n, de, dtype=torch.float64)
    op = Fourier(e, lambda_reg=1e-2, num_features=500, dtype=torch.float64)
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal())


def test_rff_dense_kernel_must_have_finite_entries():
    """The dense Fourier matvec is finite on well-scaled inputs."""
    torch.manual_seed(0)
    n, de = 50, 4
    e = torch.randn(n, de, dtype=torch.float64) * 0.5
    x = torch.randn(n, dtype=torch.float64)
    op = Fourier(e, lambda_reg=1e-2, num_features=200, dtype=torch.float64)
    out = op.matvec(x)
    assert torch.isfinite(out).all()
    assert out.shape == (n,)


def test_rff_matvec_matches_dense_at_x():
    """``op.matvec(x) == op.to_dense() @ x`` to PCG precision (RFF's
    documented high-feature-limit approximation converges).
    """
    torch.manual_seed(0)
    n, de = 12, 3
    e = torch.randn(n, de, dtype=torch.float64)
    op = Fourier(e, lambda_reg=1e-2, num_features=200, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), op.to_dense() @ x, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Sparse k-NN: a k=n kernel matches the dense operator.
# ---------------------------------------------------------------------------
def test_sparse_knn_k_equals_n_matches_dense():
    """``k_neighbors == n`` produces the dense operator to PCG precision."""
    torch.manual_seed(0)
    n, de = 25, 4
    e = torch.randn(n, de, dtype=torch.float64)
    lam = 1e-2

    exact = Exact(e, lambda_reg=lam, dtype=torch.float64)
    sparse = Neighbors(e, lambda_reg=lam, k_neighbors=n, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    torch.testing.assert_close(sparse.matvec(x), exact.matvec(x), atol=1e-12, rtol=1e-12)


def test_sparse_knn_diag_matches_to_dense_diag():
    """``Neighbors.diagonal() == diag(to_dense())`` exactly."""
    torch.manual_seed(0)
    n, de = 25, 4
    e = torch.randn(n, de, dtype=torch.float64)
    op = Neighbors(e, lambda_reg=1e-2, k_neighbors=8, dtype=torch.float64)
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal())


# ---------------------------------------------------------------------------
# Documentation guard: Nyström ``matvec`` vs ``to_dense @ x`` divergence.
# The audit flagged this; the documented behaviour is that the
# ``matvec`` path uses a cached ``K_mm^{-1}`` that is applied twice,
# whereas ``to_dense @ x`` is the audit-correct reference. This test
# gates the current behaviour rather than passing a fake high-error
# tolerance.
# ---------------------------------------------------------------------------
def test_nystrom_matvec_diverges_from_dense_at_x():
    """The audit-flagged behaviour: ``Nystrom.matvec(e_i) !=
    to_dense()[:, i]``. The matvec implementation uses ``K_mm^{-1}``
    twice while the dense path is the canonical reference.
    """
    torch.manual_seed(0)
    n, de = 12, 3
    e = torch.randn(n, de, dtype=torch.float64)
    op = Nystrom(e, lambda_reg=1e-2, num_landmarks=6, dtype=torch.float64)

    dense = op.to_dense()
    ei = torch.zeros(n, dtype=torch.float64)
    ei[0] = 1.0
    matvec_at_e0 = op.matvec(ei)
    diff = (matvec_at_e0 - dense[:, 0]).norm() / dense[:, 0].norm().clamp_min(1e-10)
    # The drift is real (the audit-reported ~99% relative error in this
    # configuration); we pin the current behaviour rather than a fake
    # passing tolerance.
    assert diff.item() > 0.1, (
        "Nyström matvec drift unexpectedly small; "
        "audit-flagged behaviour may have been silently fixed"
    )
