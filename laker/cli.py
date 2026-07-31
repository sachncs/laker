"""Command-line interface for LAKER.

Public class :class:`CLI` with single-word static methods.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger("laker")


class CLI:
    """Single class exposing the LAKER CLI."""

    @staticmethod
    def logging(verbose: bool) -> None:
        """Configure root logger level for the CLI invocation."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @staticmethod
    def load(path: str) -> torch.Tensor:
        """Load a tensor from ``.pt``/``.pth``/``.npy`` files."""
        if path.endswith(".npy"):
            return torch.from_numpy(np.load(path))
        if path.endswith(".pt") or path.endswith(".pth"):
            return torch.load(path, weights_only=True)
        raise ValueError(f"Unsupported file extension for {path}. Expected .pt, .pth, or .npy.")

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Build the argument parser with single-word flags."""
        from laker import __version__

        p = argparse.ArgumentParser(
            description="LAKER: Learning-based Attention Kernel Regression",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
        p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
        sub = p.add_subparsers(dest="command")

        fit_p = sub.add_parser("fit", help="Fit a LAKER model to data")
        fit_p.add_argument("--locations", required=True, help="Locations .pt/.npy")
        fit_p.add_argument("--measurements", required=True, help="Measurements .pt/.npy")
        fit_p.add_argument("--output", required=True, help="Where to save the model")
        fit_p.add_argument(
            "--regularization",
            "--lam",
            dest="lam",
            type=float,
            default=1e-2,
            help="Regularisation lambda",
        )
        fit_p.add_argument("--gamma", type=float, default=1e-1, help="CCCP gamma")
        fit_p.add_argument("--embed-dim", type=int, default=10, help="Embedding dimension")
        fit_p.add_argument(
            "--probes",
            "--num",
            dest="num",
            type=int,
            default=None,
            help="Number of random probes",
        )
        fit_p.add_argument("--device", default="cpu", help="torch device")
        fit_p.add_argument(
            "--dtype",
            default="float32",
            choices=["float16", "bfloat16", "float32", "float64"],
        )
        fit_p.add_argument(
            "--kernel",
            default="exact",
            choices=["exact", "nystrom", "fourier", "neighbors", "grid", "spectrum", "hybrid"],
        )

        pred_p = sub.add_parser("predict", help="Predict using a fitted model")
        pred_p.add_argument("--model", required=True, help="Fitted model .pt path")
        pred_p.add_argument("--locations", required=True, help="Query locations")
        pred_p.add_argument("--output", required=True, help="Where to save predictions")
        return p

    @staticmethod
    def fit(args) -> None:
        """Dispatch the ``fit`` subcommand."""
        from laker import Laker

        logger.info("Loading data...")
        x = CLI.load(args.locations)
        y = CLI.load(args.measurements)
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[args.dtype]
        model = Laker(
            lam=args.lam,
            gamma=args.gamma,
            embed_dim=args.embed_dim,
            num=args.num,
            device=args.device,
            dtype=dtype,
            kernel=args.kernel,
        )
        model.fit(x, y)
        model.save(args.output)
        logger.info("Model saved to %s", args.output)

    @staticmethod
    def predict(args) -> None:
        """Dispatch the ``predict`` subcommand."""
        from laker import Laker

        logger.info("Loading model...")
        model = Laker.load(args.model)
        x = CLI.load(args.locations)
        preds = model.predict(x)
        torch.save(preds, args.output)
        logger.info("Predictions saved to %s", args.output)

    @classmethod
    def run(cls, argv: Optional[Sequence[str]] = None) -> int:
        """Run the CLI and exit with the appropriate status."""
        p = cls.parser()
        args = p.parse_args(argv)
        cls.logging(args.verbose)
        if args.command == "fit":
            cls.fit(args)
            sys.exit(0)
        if args.command == "predict":
            cls.predict(args)
            sys.exit(0)
        p.print_help()
        sys.exit(1)


__all__ = ["CLI"]