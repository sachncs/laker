"""Tests for laker.utils module."""

import torch

from laker.utils import adaptive_shrinkage_rho, eigh_stable, trace_normalize


def test_trace_normalize_identity():
    """trace_normalize(identity) should return identity."""
    n = 10
    ident = torch.eye(n)
    result = trace_normalize(ident)
    torch.testing.assert_close(result, ident)


def test_trace_normalize_constant():
    """trace_normalize(scalar * I) should return identity."""
    result = trace_normalize(torch.eye(5) * 3.0)
    torch.testing.assert_close(result, torch.eye(5))


def test_adaptive_shrinkage_rho():
    """adaptive_shrinkage_rho should return a positive rho."""
    rho = adaptive_shrinkage_rho(num_probes=10, problem_size=100, gamma=0.1)
    assert rho > 0
    assert rho < 1.0


def test_adaptive_shrinkage_rho_few_probes():
    """adaptive_shrinkage_rho with few probes should return higher rho."""
    rho_many = adaptive_shrinkage_rho(num_probes=50, problem_size=100, gamma=0.01)
    rho_few = adaptive_shrinkage_rho(num_probes=5, problem_size=100, gamma=0.01)
    assert rho_few > rho_many


def test_eigh_stable_clamps():
    """eigh_stable should clamp eigenvalues to eps."""
    a = torch.diag(torch.tensor([1e-12, 1.0, 2.0]))
    vals, _ = eigh_stable(a, eps=1e-8)
    assert vals[0].item() > 0
    assert vals[0].item() >= 1e-9


def test_eigh_stable_preserves_eigenvectors():
    """eigh_stable should return valid eigenvectors."""
    a = torch.randn(5, 5)
    a = a @ a.T + torch.eye(5) * 0.1
    vals, vecs = eigh_stable(a)
    recon = vecs @ torch.diag(vals) @ vecs.T
    torch.testing.assert_close(recon, a, atol=1e-5, rtol=1e-5)
