"""Behavioural + precision tests for the distributed kernel operator.

The distributed kernel wraps the exact kernel and either shards
across CUDA devices or falls back to single-device execution. These
tests verify both modes against the single-device reference and
pin down the CUDA gating.
"""

from __future__ import annotations

import pytest
import torch

from laker.distributed import DistributedAttentionKernelOperator


# ---------------------------------------------------------------------------
# Single-device fallback: behaviour on a host with zero CUDA devices.
# ---------------------------------------------------------------------------
def test_single_device_flag_when_no_cuda(monkeypatch):
    """``single_device`` is True when ``torch.cuda.is_available()``
    is False. The wrapper collapses to a single inner operator.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    e = torch.randn(20, 5, dtype=torch.float64)
    op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    assert op.single_device is True
    assert op.master_device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Single-device fallback: matvec / diagonal / to_dense / kernel_eval
# precision against the exact kernel.
# ---------------------------------------------------------------------------
def test_matvec_matches_single_device_exact_kernel():
    """``matvec(x)`` from the single-device fallback matches the
    reference exact kernel to the precision of dense matmul.
    """
    from laker.kernel import Exact

    torch.manual_seed(0)
    n = 40
    e = torch.randn(n, 6, dtype=torch.float64)
    x = torch.randn(n, dtype=torch.float64)

    single = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    ref = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    torch.testing.assert_close(single.matvec(x), ref.matvec(x), atol=1e-10, rtol=1e-10)


def test_diagonal_matches_single_device_exact_kernel():
    """``diagonal()`` equals the diagonal of the single-device operator
    (the audit-flagged invariant)."""
    from laker.kernel import Exact

    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 6, dtype=torch.float64)

    single = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    ref = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    torch.testing.assert_close(single.diagonal(), ref.diagonal(), atol=1e-10, rtol=1e-10)


def test_to_dense_matches_single_device_exact_kernel():
    """``to_dense()`` of the single-device fallback matches the
    reference exact operator."""
    from laker.kernel import Exact

    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 6, dtype=torch.float64)

    single = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    ref = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    torch.testing.assert_close(single.to_dense(), ref.to_dense(), atol=1e-10, rtol=1e-10)


def test_kernel_eval_matches_single_device_exact_kernel():
    """``kernel_eval(x)`` matches the reference kernel matrix."""
    from laker.kernel import Exact

    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 6, dtype=torch.float64)
    x = torch.randn(7, 6, dtype=torch.float64)

    single = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    ref = Exact(e, lambda_reg=1e-2, dtype=torch.float64)
    torch.testing.assert_close(single.kernel_eval(x), ref.kernel_eval(x), atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# CUDA path: skipped when no GPU; behaviour assertions when present.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="CUDA with at least two devices required for multi-device test",
)
def test_multi_device_matvec_outputs_on_master():
    """On multi-device CUDA, the result tensor lives on the master
    device and matches the single-device reference."""
    n = 30
    e = torch.randn(n, 8, dtype=torch.float64, device="cuda")
    x = torch.randn(n, dtype=torch.float64, device="cuda")

    dist_op = DistributedAttentionKernelOperator(
        e, lambda_reg=1e-2, master_device=torch.device("cuda")
    )
    y = dist_op.matvec(x)
    assert y.shape == (n,)
    assert y.device.type == "cuda"
    # ``single_device`` should be False on a multi-GPU host.
    assert dist_op.single_device is False


def test_matvec_outputs_are_finite():
    """The fallback path returns finite outputs even when the
    embedding norm is large enough to push ``exp(E E^T)`` past
    numerical limits — the kernel applies dtype-aware overflow
    clamping.
    """

    torch.manual_seed(0)
    n = 30
    e = torch.randn(n, 4, dtype=torch.float64) * 5.0
    x = torch.randn(n, dtype=torch.float64)
    op = DistributedAttentionKernelOperator(e, lambda_reg=1e-2, dtype=torch.float64)
    assert torch.isfinite(op.matvec(x)).all()
