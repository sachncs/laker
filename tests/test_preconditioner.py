"""Behavioural + precision tests for the CCCP preconditioner.

The tests pin down: build-time tensor shapes, the documented
isotropic-coef sign, the property that ``op.matvec`` preconditioned
by the inverse square root shrinks the operator's largest eigenvalue
relative to the unpreconditioned system, and the behaviour of the
two probe strategies (Gaussian vs power-iteration).
"""

from __future__ import annotations

import torch

from laker.kernels import Attention as Exact
from laker.preconditioner import CCCPPreconditioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_op(n: int = 30, de: int = 4, lam: float = 1e-2, dtype=torch.float64):
    """Build an exact-kernel operator with random embeddings."""
    torch.manual_seed(0)
    e = torch.randn(n, de, dtype=dtype)
    return Exact(e, lambda_reg=lam, dtype=dtype), e


def _build_prec(
    n: int,
    op,
    num_probes: int = 50,
    max_iter: int = 50,
    probe_strategy: str = "gaussian",
    power_iter_steps: int = 3,
):
    prec = CCCPPreconditioner(
        num_probes=num_probes,
        gamma=1e-1,
        epsilon=1e-8,
        base_rho=0.05,
        max_iter=max_iter,
        tol=1e-5,
        verbose=False,
        device=op.device,
        dtype=op.dtype,
        probe_strategy=probe_strategy,
        power_iter_steps=power_iter_steps,
    )
    prec.build(op.matvec, n, seed=42)
    return prec


def _power_iter(matvec, n, steps=40):
    """Estimate the largest eigenvalue of a linear operator on R^n."""
    v = torch.randn(n, dtype=torch.float64)
    v = v / v.norm()
    for _ in range(steps):
        v = matvec(v)
        v = v / v.norm()
    return torch.dot(v, matvec(v)).item()


# ---------------------------------------------------------------------------
# Build-time shape and isotropic-coef sign.
# ---------------------------------------------------------------------------
def test_preconditioner_build_stores_correct_q_basis_shape():
    """``q_basis`` has shape ``(n, min(num_probes, n))`` after build.
    The CCCP code clamps the effective probe count to ``n``.
    """
    op, _ = _make_op(n=30, de=4)
    prec = _build_prec(30, op, num_probes=50)
    assert prec.q_basis is not None
    assert prec.q_basis.shape == (30, 30)


def test_preconditioner_isotropic_coef_positive():
    """``isotropic_coef`` is strictly positive: it represents the
    diagonal of the preconditioner, which must be > 0 for stability.
    """
    op, _ = _make_op(n=30)
    prec = _build_prec(30, op)
    assert prec.isotropic_coef > 0.0


def test_preconditioner_apply_preserves_shape_and_dtype():
    """``apply(v)`` returns a tensor of the same shape and dtype as v."""
    op, _ = _make_op(n=30)
    prec = _build_prec(30, op)
    v = torch.randn(30, dtype=torch.float64)
    pv = prec.apply(v)
    assert pv.shape == v.shape
    assert pv.dtype == v.dtype


# ---------------------------------------------------------------------------
# Preconditioner effect on the largest eigenvalue.
# ---------------------------------------------------------------------------
def test_preconditioned_system_has_smaller_largest_eigenvalue():
    """Preconditioned ``op.matvec`` has a smaller largest eigenvalue
    than the unpreconditioned operator. This is the property that
    reduces PCG iteration count.
    """
    op, _ = _make_op(n=30)
    prec = _build_prec(30, op, num_probes=80)

    lam_max_orig = _power_iter(op.matvec, n=30)
    lam_max_pre = _power_iter(lambda v: prec.apply(op.matvec(v)), n=30)
    assert lam_max_pre < lam_max_orig, (
        f"preconditioner should shrink λ_max: " f"orig={lam_max_orig:.3f}, pre={lam_max_pre:.3f}"
    )


# ---------------------------------------------------------------------------
# Probe strategies: both reduce the largest eigenvalue; power_iter
# is comparable to Gaussian.
# ---------------------------------------------------------------------------
def test_both_probe_strategies_reduce_largest_eigenvalue():
    """Both Gaussian and power-iteration probe strategies shrink
    ``λ_max``; the preconditioner is non-degenerate in either mode.
    """
    op, _ = _make_op(n=30)
    pre_gauss = _build_prec(30, op, num_probes=50, probe_strategy="gaussian")
    pre_power = _build_prec(30, op, num_probes=50, probe_strategy="power_iter")

    lam_max_orig = _power_iter(op.matvec, n=30)
    lam_max_gauss = _power_iter(lambda v: pre_gauss.apply(op.matvec(v)), n=30)
    lam_max_power = _power_iter(lambda v: pre_power.apply(op.matvec(v)), n=30)

    assert lam_max_gauss < lam_max_orig
    assert lam_max_power < lam_max_orig


def test_power_iter_strategy_matches_gaussian_within_factor():
    """The mixed power-iteration strategy is comparable to the pure
    Gaussian strategy: ``λ_max(power_iter) < λ_max(gaussian) * 2``.
    """
    op, _ = _make_op(n=30)
    pre_gauss = _build_prec(30, op, num_probes=50, probe_strategy="gaussian")
    pre_power = _build_prec(30, op, num_probes=50, probe_strategy="power_iter")
    lam_max_gauss = _power_iter(lambda v: pre_gauss.apply(op.matvec(v)), n=30)
    lam_max_power = _power_iter(lambda v: pre_power.apply(op.matvec(v)), n=30)
    assert lam_max_power < lam_max_gauss * 2.0, (
        f"power_iter λ_max {lam_max_power:.3f} should be within 2x of "
        f"gaussian {lam_max_gauss:.3f}"
    )


# ---------------------------------------------------------------------------
# Math: `q_basis` has orthonormal columns.
# ---------------------------------------------------------------------------
def test_q_basis_columns_are_unit_norm():
    """``q_basis`` is the orthonormal basis from the QR factorisation;
    every column has unit L2 norm.
    """
    op, _ = _make_op(n=30, de=4)
    prec = _build_prec(30, op, num_probes=15)
    norms = prec.q_basis.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones(15, dtype=torch.float64), atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Construction argument round-trip via attribute access.
# ---------------------------------------------------------------------------
def test_constructor_arguments_round_trip():
    """Constructor arguments are stored as attributes and the
    default values match the documented contract.
    """
    pre = CCCPPreconditioner(
        num_probes=33,
        gamma=0.123,
        epsilon=1e-9,
        base_rho=0.07,
        max_iter=99,
        tol=1e-7,
    )
    assert pre.num_probes == 33
    assert abs(pre.gamma - 0.123) < 1e-12
    assert abs(pre.epsilon - 1e-9) < 1e-15
    assert abs(pre.base_rho - 0.07) < 1e-12
    assert pre.max_iter == 99
    assert abs(pre.tol - 1e-7) < 1e-15
    assert pre.probe_strategy == "gaussian"
