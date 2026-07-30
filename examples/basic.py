"""Reproduce the worked example from Section IV-E of the LAKER paper.

Three hand-crafted embeddings and observations yield a 3x3 kernel
system. The example verifies that :class:`laker.Laker` recovers the
exact solve ``alpha = (K + lambda I)^{-1} y`` (Eq. 53) and that the
kernel matrix produced by the ``AttentionKernelOperator`` matches the
reference exponential dot-product kernel.

Run:
    python -m examples.basic
"""
from __future__ import annotations

import torch

from laker import Laker
from laker.kernels import AttentionKernelOperator


class PaperExample:
    """Self-contained reproduction of the n=3 worked example."""

    EMBEDDINGS = torch.tensor(
        [[0.241, 0.444], [-0.336, 0.112], [-0.220, 0.353]],
        dtype=torch.float64,
    )
    OBSERVATIONS = torch.tensor(
        [-66.14, -65.77, -77.30], dtype=torch.float64
    )
    QUERY = torch.tensor([[0.051, 0.452]], dtype=torch.float64)
    REGULARIZATION = 0.1

    @classmethod
    def _exact_alpha(cls) -> torch.Tensor:
        gram = torch.exp(cls.EMBEDDINGS @ cls.EMBEDDINGS.T)
        system = gram + cls.REGULARIZATION * torch.eye(3, dtype=torch.float64)
        return torch.linalg.solve(system, cls.OBSERVATIONS)

    @classmethod
    def run(cls) -> None:
        """Verify alpha recovery and the kernel matrix against Eq. (53)."""
        alpha_exact = cls._exact_alpha()

        class FixedEmbedding(torch.nn.Module):
            def forward(self, x):
                return cls.EMBEDDINGS

        model = Laker(
            embedding_dim=2,
            regularization=cls.REGULARIZATION,
            gamma=0.0,
            probes=3,
            cccp_max_iter=10,
            pcg_tol=1e-12,
            pcg_max_iter=10,
            encoder=FixedEmbedding(),
            dtype=torch.float64,
        )
        model.fit(torch.zeros(3, 2, dtype=torch.float64), cls.OBSERVATIONS)

        rel_err = torch.norm(model.coef_ - alpha_exact) / torch.norm(alpha_exact)
        print(f"Laker alpha:  {model.coef_.tolist()}")
        print(f"Exact alpha:   {alpha_exact.tolist()}")
        print(f"Relative err:  {rel_err.item():.3e}")
        assert rel_err < 1e-6, f"alpha mismatch: rel_err={rel_err.item():.3e}"

        # Verify the kernel matrix used by the operator matches
        # Eq. (53): k(x, x') = exp(x @ x'.T).
        op = AttentionKernelOperator(cls.EMBEDDINGS, lambda_reg=cls.REGULARIZATION)
        kernel_train = op.kernel_eval(cls.EMBEDDINGS, cls.EMBEDDINGS)
        kernel_query_train = op.kernel_eval(cls.QUERY, cls.EMBEDDINGS)

        expected_kernel = torch.exp(cls.EMBEDDINGS @ cls.EMBEDDINGS.T)
        err = (
            torch.linalg.norm(kernel_train - expected_kernel)
            / torch.linalg.norm(expected_kernel)
        ).item()
        assert err < 1e-12, f"kernel mismatch: err={err}"
        print(f"Kernel matrix relative err: {err:.3e}")

        prediction_exact = (
            torch.exp(cls.QUERY @ cls.EMBEDDINGS.T) @ alpha_exact
        ).item()
        prediction_via_op = (kernel_query_train @ model.coef_).item()
        pred_err = abs(prediction_via_op - prediction_exact)
        print(f"Exact prediction:  {prediction_exact:.2f} dBm")
        print(f"Kernel prediction: {prediction_via_op:.2f} dBm")
        print(f"Prediction err:    {pred_err:.3e}")
        assert pred_err < 1e-6


if __name__ == "__main__":
    PaperExample.run()
