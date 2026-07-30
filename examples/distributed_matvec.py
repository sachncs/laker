"""Distributed multi-device matvec sanity check.

Compares the matvec and diagonal of the multi-device distributed kernel
against the single-device exact kernel.

Run:
    python -m examples.distributed_matvec --devices 0,1
"""
from __future__ import annotations

import argparse

import torch

from laker.distributed_kernels import DistributedAttentionKernelOperator
from laker.kernels import AttentionKernelOperator


class Distributed:
    """Compare single-device and multi-device kernel operators."""

    @staticmethod
    def run(devices: str = "0", n: int = 200, regularization: float = 1e-2) -> None:
        """Smoke-test the two operators on identical inputs."""
        if not torch.cuda.is_available():
            print("CUDA not available; skipping distributed smoke test.")
            return

        device_ids = [int(d) for d in devices.split(",") if d]
        torch.manual_seed(0)
        embeddings = torch.randn(n, 8)
        target = torch.randn(n)

        single = AttentionKernelOperator(embeddings, regularization=regularization)
        distributed = DistributedAttentionKernelOperator(
            embeddings,
            regularization=regularization,
            master_device="cuda",
            devices=device_ids,
        )

        for name in ("matvec", "diagonal"):
            v = torch.randn(n)
            a_single = (
                single.matvec(v)
                if name == "matvec"
                else single.diagonal()
            )
            a_dist = (
                distributed.matvec(v)
                if name == "matvec"
                else distributed.diagonal()
            )
            err = (a_single - a_dist).norm().item() / a_single.norm().item()
            print(f"{name}: relative error = {err:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated CUDA device indices. Skip if no CUDA.",
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--regularization", type=float, default=1e-2)
    args = parser.parse_args()
    Distributed.run(
        devices=args.devices, n=args.n, regularization=args.regularization
    )
