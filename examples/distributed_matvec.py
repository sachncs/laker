"""Distributed multi-device matvec sanity check.

Run:
    python -m examples.distributed_matvec --devices 0,1
"""
from __future__ import annotations

import argparse

import torch

from laker.kernel import Distribute, Exact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated CUDA device indices (skips on non-CUDA hosts).",
    )
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping distributed smoke test.")
        return

    device_ids = [int(d) for d in args.devices.split(",") if d]
    n = args.n
    torch.manual_seed(0)
    embeddings = torch.randn(n, 8)
    targets = torch.randn(n)

    base = Exact(embeddings, regularization=1e-2)
    distributed = Distribute(embeddings, regularization=1e-2, devices=device_ids)

    v = torch.randn(n)
    a_single = base.matvec(v)
    a_dist = distributed.matvec(v)
    err = (a_single - a_dist).norm().item() / a_single.norm().item()
    print(f"Relative error vs single-device: {err:.2e}")

    d_single = base.diagonal()
    d_dist = distributed.diagonal()
    err = (d_single - d_dist).norm().item() / d_single.norm().item()
    print(f"Diagonal relative error: {err:.2e}")


if __name__ == "__main__":
    main()
