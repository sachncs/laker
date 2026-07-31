"""Behavioural + precision tests for the kernel operator hierarchy.

Every test asserts a real numerical contract — ``matvec(x) ==
to_dense() @ x`` for the exact kernel, diagonal-vs-``diag(to_dense())``
equality, ``kernel_eval`` consistency with the analytical
``exp(E Eᵀ)``, dtype preservation, and chunked-vs-full matvec
agreement within the documented reduction-order epsilon.
"""

from __future__ import annotations

import torch

from laker.kernels import Attention as Exact
from laker.kernels import SpectralAttention as Spectrum


# ---------------------------------------------------------------------------
# Exact kernel: shape, dtype, contract.
# ---------------------------------------------------------------------------
def test_kernel_shape_attributes():
    """``Exact.shape == (n, n)`` and the embedding dimension is
    reported correctly."""
    n = 20
    de = 5
    e = torch.randn(n, de, dtype=torch.float64)
    op = Exact(e, lambda_reg=0.1, dtype=torch.float64)
    assert op.shape == (n, n)
    assert op.n == n
    assert op.embedding_dim == de


def test_kernel_dtype_preserved_in_output():
    """``matvec`` preserves the kernel's dtype on the output."""
    e = torch.randn(15, 4, dtype=torch.float64)
    op = Exact(e, lambda_reg=0.1, dtype=torch.float64)
    x = torch.randn(15, dtype=torch.float64)
    assert op.matvec(x).dtype == torch.float64


# ---------------------------------------------------------------------------
# Exact matvec == dense @ x and chunked == full.
# ---------------------------------------------------------------------------
def test_kernel_matvec_matches_dense_at_x():
    """``op.matvec(x) == op.to_dense() @ x`` to PCG precision."""
    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 4, dtype=torch.float64)
    lam = 0.05
    op = Exact(e, lambda_reg=lam, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), op.to_dense() @ x, atol=1e-12, rtol=1e-12)


def test_kernel_matvec_chunked_matches_full():
    """Chunked matvec equals the full matvec within the documented
    reduction-order epsilon (1e-4 relative on float32 with this chunk
    size).
    """
    torch.manual_seed(0)
    n = 100
    e = torch.randn(n, 6, dtype=torch.float64)
    op_full = Exact(e, lambda_reg=0.1, chunk_size=None, dtype=torch.float64)
    op_chunk = Exact(e, lambda_reg=0.1, chunk_size=16, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    torch.testing.assert_close(op_full.matvec(x), op_chunk.matvec(x), atol=1e-10, rtol=1e-10)


def test_kernel_matvec_2d_matches_dense_x():
    """A 2-D RHS ``(n, k)`` is processed column-wise: ``op.matvec(X)``
    equals ``to_dense() @ X``."""
    torch.manual_seed(0)
    n = 20
    de = 5
    e = torch.randn(n, de, dtype=torch.float64)
    op = Exact(e, lambda_reg=0.1, dtype=torch.float64)
    X = torch.randn(n, 3, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(X), op.to_dense() @ X, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Diagonal vs to_dense and the analytical formula.
# ---------------------------------------------------------------------------
def test_kernel_diagonal_matches_analytical_formula():
    """For the exact kernel, ``diag[i] == lambda + exp(||e_i||^2)``."""
    torch.manual_seed(0)
    n = 15
    e = torch.randn(n, 3, dtype=torch.float64)
    lam = 0.2
    op = Exact(e, lambda_reg=lam, dtype=torch.float64)
    sq = torch.sum(e**2, dim=1)
    expected = lam + torch.exp(sq)
    torch.testing.assert_close(op.diagonal(), expected, atol=1e-12, rtol=1e-12)


def test_kernel_diagonal_matches_to_dense_diagonal():
    """``diagonal() == diag(to_dense())`` exactly for every kernel."""
    e = torch.randn(20, 5, dtype=torch.float64)
    op = Exact(e, lambda_reg=0.1, dtype=torch.float64)
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal(), atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# kernel_eval: shape and consistency with the analytical formula.
# ---------------------------------------------------------------------------
def test_kernel_eval_matches_exp_q_e_train():
    """``kernel_eval(query) == exp(query @ e_train.T)`` to PCG precision."""
    torch.manual_seed(0)
    n = 30
    de = 4
    e_train = torch.randn(n, de, dtype=torch.float64)
    e_query = torch.randn(8, de, dtype=torch.float64)
    op = Exact(e_train, lambda_reg=0.1, dtype=torch.float64)
    expected = torch.exp(e_query @ e_train.T)
    torch.testing.assert_close(op.kernel_eval(e_query), expected, atol=1e-12, rtol=1e-12)


def test_kernel_eval_cross_query_train_pair():
    """``kernel_eval(q, t)`` has shape ``(n_query, n_train)`` and equals
    ``exp(q @ t.T)``.
    """
    torch.manual_seed(0)
    e_train = torch.randn(20, 5, dtype=torch.float64)
    e_query = torch.randn(7, 5, dtype=torch.float64)
    op = Exact(e_train, lambda_reg=0.1, dtype=torch.float64)
    k = op.kernel_eval(e_query, e_train)
    assert k.shape == (7, 20)
    expected = torch.exp(e_query @ e_train.T)
    torch.testing.assert_close(k, expected, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Spectrum kernel: dense / spectral consistency.
# ---------------------------------------------------------------------------
def test_spectral_kernel_matvec_matches_dense_construction():
    """``op.matvec(x) == (U diag(spectrum) U^T + lambda I) x``."""
    torch.manual_seed(0)
    n = 30
    de = 4
    e = torch.randn(n, de, dtype=torch.float64)
    op = Spectrum(e, lambda_reg=0.05, num_knots=5, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)
    k_dense = op.u_matrix @ torch.diag(op.spectrum) @ op.u_matrix.T
    k_dense.diagonal().add_(0.05)
    torch.testing.assert_close(op.matvec(x), k_dense @ x, atol=1e-12, rtol=1e-12)


def test_spectral_kernel_diagonal_matches_to_dense():
    """``diagonal() == diag(to_dense())`` for the spectrum kernel."""
    torch.manual_seed(0)
    n = 15
    de = 3
    e = torch.randn(n, de, dtype=torch.float64)
    op = Spectrum(e, lambda_reg=0.2, num_knots=4, dtype=torch.float64)
    torch.testing.assert_close(op.diagonal(), op.to_dense().diagonal(), atol=1e-12, rtol=1e-12)


def test_spectral_kernel_eval_consistent_with_spectral_basis():
    """``kernel_eval(query, train)`` for the spectral kernel equals the
    formula ``C_q diag(spectrum) C_train.T`` where ``C`` projects onto
    the spectrum basis.
    """
    torch.manual_seed(0)
    n = 20
    de = 4
    e_train = torch.randn(n, de, dtype=torch.float64)
    e_query = torch.randn(8, de, dtype=torch.float64)
    op = Spectrum(e_train, lambda_reg=0.1, num_knots=3, dtype=torch.float64)
    cx = (e_query @ op.vh.T) * op.sigma_inv.unsqueeze(0)
    cy = (e_train @ op.vh.T) * op.sigma_inv.unsqueeze(0)
    expected = (cx * op.spectrum.unsqueeze(0)) @ cy.T
    torch.testing.assert_close(op.kernel_eval(e_query, e_train), expected, atol=1e-12, rtol=1e-12)
