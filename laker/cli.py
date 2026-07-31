"""Command-line interface (CLI) class.

The ``laker`` console script entry point declared in
``pyproject.toml`` invokes :meth:`CLI.run` directly. New code should
depend on :class:`CLI` directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

import numpy
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
            return torch.from_numpy(numpy.load(path))
        if path.endswith(".pt") or path.endswith(".pth"):
            return torch.load(path, weights_only=True)
        raise ValueError(f"Unsupported file extension for {path}. Expected .pt, .pth, or .npy.")

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        from laker import __version__

        arg_parser = argparse.ArgumentParser(
            description="LAKER: Learning-based Attention Kernel Regression",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        arg_parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
        arg_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
        subparsers = arg_parser.add_subparsers(dest="command")

        fit_parser = subparsers.add_parser("fit", help="Fit a LAKER model to data")
        fit_parser.add_argument(
            "--locations",
            required=True,
            help="Path to locations .pt or .npy file",
        )
        fit_parser.add_argument(
            "--measurements",
            required=True,
            help="Path to measurements .pt or .npy file",
        )
        fit_parser.add_argument("--output", required=True, help="Path to save fitted model")
        fit_parser.add_argument(
            "--regularization",
            "--lambda-reg",
            dest="regularization",
            type=float,
            default=1e-2,
            help="Regularisation lambda",
        )
        fit_parser.add_argument(
            "--gamma", type=float, default=1e-1, help="CCCP regularisation gamma"
        )
        fit_parser.add_argument("--embedding-dim", type=int, default=10, help="Embedding dimension")
        fit_parser.add_argument(
            "--probes",
            "--num-probes",
            dest="probes",
            type=int,
            default=None,
            help="Number of random probes",
        )
        fit_parser.add_argument("--device", default="cpu", help="torch device")
        fit_parser.add_argument(
            "--dtype",
            default="float32",
            choices=["float16", "bfloat16", "float32", "float64"],
        )
        fit_parser.add_argument(
            "--kernel",
            default="exact",
            choices=["exact", "nystrom", "fourier", "neighbors", "grid", "spectrum", "hybrid"],
        )

        pred_parser = subparsers.add_parser("predict", help="Predict using a fitted model")
        pred_parser.add_argument("--model", required=True, help="Path to fitted model .pt file")
        pred_parser.add_argument("--locations", required=True, help="Path to query locations")
        pred_parser.add_argument("--output", required=True, help="Path to save predictions")
        return arg_parser

    @staticmethod
    def fit(args) -> None:
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
            regularization=args.regularization,
            gamma=args.gamma,
            embedding_dim=args.embedding_dim,
            probes=args.probes,
            device=args.device,
            dtype=dtype,
            kernel=args.kernel,
        )
        model.fit(x, y)
        model.save(args.output)
        logger.info("Model saved to %s", args.output)

    @staticmethod
    def predict(args) -> None:
        from laker import Laker

        logger.info("Loading model...")
        model = Laker.load(args.model)
        x = CLI.load(args.locations)
        predictions = model.predict(x)
        torch.save(predictions, args.output)
        logger.info("Predictions saved to %s", args.output)

    @classmethod
    def run(cls, argv: Optional[Sequence[str]] = None) -> int:
        """Run the CLI and ``sys.exit`` with the appropriate status.

        Successful fit / predict exits ``0``; a no-subcommand
        invocation prints help and exits ``1``. The ``return`` value
        is documented for tests that patch ``sys.exit``.
        """
        arg_parser = cls.parser()
        args = arg_parser.parse_args(argv)
        cls.logging(args.verbose)
        if args.command == "fit":
            cls.fit(args)
            sys.exit(0)
            return 0  # unreachable; satisfies static type checkers
        if args.command == "predict":
            cls.predict(args)
            sys.exit(0)
            return 0  # unreachable; satisfies static type checkers
        arg_parser.print_help()
        sys.exit(1)
        return 1  # unreachable; satisfies static type checkers


__all__ = ["CLI"]
