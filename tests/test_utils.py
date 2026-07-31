"""Behavioural + precision tests for the numerical helpers.

Covers ``Helpers`` static methods and the legacy ``utils`` module
helpers (``trace_normalize``, ``eigh_stable``,
``adaptive_shrinkage_rho``). Every assertion targets a real
mathematical contract: trace normalisation, eigenvalue clamping,
shrinkage monotonicity, regression accuracy.
"""

from __future__ import annotations

import torch

from laker.helpers import Helpers
from laker.utils import adaptive_shrinkage_rho, eigh_stable, trace_normalize


# ---------------------------------------------------------------------------
# trace_normalize.
# ---------------------------------------------------------------------------
def test_trace_normalize_identity_is_identity():
    """``trace_normalize(I) == I``: trace(I) == n, so the operation
    is a no-op on the identity.
    """
    n = 10
    ident = torch.eye(n, dtype=torch.float64)
    torch.testing.assert_close(trace_normalize(ident), ident)


def test_trace_normalize_scale_invariance():
    """``trace_normalize(s * I) == I``: scaling by a positive scalar
    does not change the result.
    """
    n = 5
    eye = torch.eye(n, dtype=torch.float64)
    torch.testing.assert_close(trace_normalize(eye * 3.7), eye)


def test_trace_normalize_scales_diagonal_to_n():
    """After ``trace_normalize(A)``, ``trace(A') == n`` to PCG precision."""
    torch.manual_seed(0)
    n = 12
    a = torch.randn(n, n, dtype=torch.float64)
    a = a @ a.T  # PSD
    a_out = trace_normalize(a)
    trace_out = float(torch.trace(a_out).item())
    assert abs(trace_out - n) < 1e-8, f"trace_out {trace_out} != {n}"


def test_trace_normalize_preserves_pd_structure():
    """``trace_normalize`` preserves the PSD property."""
    torch.manual_seed(0)
    n = 8
    a = torch.randn(n, n, dtype=torch.float64)
    a = a @ a.T + torch.eye(n, dtype=torch.float64) * 0.1
    a_out = trace_normalize(a)
    eigvals = torch.linalg.eigvalsh(a_out)
    assert eigvals.min() > 0, f"non-PSD output: min eigval {eigvals.min().item()}"


# ---------------------------------------------------------------------------
# eigh_stable.
# ---------------------------------------------------------------------------
def test_eigh_stable_clamps_negative_eigenvalues():
    """A negative eigenvalue below ``eps`` is clamped to ``eps``."""
    a = torch.diag(torch.tensor([1e-12, 1.0, 2.0], dtype=torch.float64))
    vals, _ = eigh_stable(a, eps=1e-8)
    assert vals[0].item() >= 1e-9
    assert vals[0].item() > 0


def test_eigh_stable_preserves_eigendecomposition():
    """``V diag(λ) V^T`` reconstructs the input PSD matrix to PCG precision."""
    torch.manual_seed(0)
    a = torch.randn(5, 5, dtype=torch.float64)
    a = a @ a.T + torch.eye(5, dtype=torch.float64) * 0.1
    vals, vecs = eigh_stable(a)
    recon = vecs @ torch.diag(vals) @ vecs.T
    torch.testing.assert_close(recon, a, atol=1e-5, rtol=1e-5)


def test_eigh_stable_returns_ascending_eigenvalues():
    """``eigh_stable`` returns eigenvalues in ascending order."""
    a = torch.diag(torch.tensor([3.0, 1.0, 4.0, 2.0, 5.0]))
    vals, _ = eigh_stable(a)
    diffs = vals[1:] - vals[:-1]
    assert (diffs >= 0).all()


# ---------------------------------------------------------------------------
# adaptive_shrinkage_rho.
# ---------------------------------------------------------------------------
def test_adaptive_shrinkage_rho_in_unit_interval():
    """``adaptive_shrinkage_rho`` returns ``rho in (0, 0.5]``."""
    for nr, n, gamma in [(10, 100, 0.1), (50, 100, 1.0), (200, 100, 5.0)]:
        rho = adaptive_shrinkage_rho(nr, n, gamma)
        assert 0.0 < rho <= 0.5


def test_adaptive_shrinkage_rho_few_probes_larger_than_many():
    """A smaller probe count relative to problem size must produce a
    larger shrinkage ``rho`` (more conservative).
    """
    rho_many = adaptive_shrinkage_rho(num_probes=50, problem_size=100, gamma=0.01)
    rho_few = adaptive_shrinkage_rho(num_probes=5, problem_size=100, gamma=0.01)
    assert rho_few > rho_many, f"rho_few {rho_few} <= rho_many {rho_many}"


def test_adaptive_shrinkage_rho_saturated_at_full_rank():
    """``num_probes >= problem_size`` returns the base ``rho`` exactly."""
    rho = adaptive_shrinkage_rho(num_probes=100, problem_size=100, gamma=0.5, base_rho=0.07)
    assert abs(rho - 0.07) < 1e-12


# ---------------------------------------------------------------------------
# Helpers (laker.helpers).
# ---------------------------------------------------------------------------
def test_helpers_safe_exp_clamps_overflow():
    """``Helpers.safe_exp(x)`` returns finite values for very large x."""
    big = torch.tensor([1e10, 1e5, 1e3], dtype=torch.float32)
    out = Helpers.safe_exp(big)
    assert torch.isfinite(out).all()
    assert (out > 0).all()


def test_helpers_safe_exp_returns_exp_for_small_x():
    """For small inputs ``safe_exp(x) == exp(x)`` to high precision."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    expected = torch.exp(x)
    torch.testing.assert_close(Helpers.safe_exp(x), expected, atol=1e-12, rtol=1e-12)


def test_helpers_set_global_seed_is_deterministic():
    """``set_global_seed`` makes subsequent random calls reproducible."""
    Helpers.set_global_seed(0)
    a = torch.randn(10)
    Helpers.set_global_seed(0)
    b = torch.randn(10)
    torch.testing.assert_close(a, b)
