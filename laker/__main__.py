"""Command-line interface for LAKER.

The :mod:`laker` package can be invoked from a shell via the console script
entry-point declared in ``pyproject.toml``. New code should depend on
:class:`laker.cli.CLI`; the legacy free functions here remain as
backward-compatible shims so existing tests and downstream code keep
working.
"""

import argparse
import logging
import sys

import numpy
import torch

logger = logging.getLogger("laker")


def setup_logging(verbose: bool) -> None:
    """Configure root logger level for the CLI invocation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_tensor(path: str) -> torch.Tensor:
    """Load a tensor from a ``.pt``/``.pth`` or ``.npy`` file."""
    if path.endswith(".npy"):
        return torch.from_numpy(numpy.load(path))
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, weights_only=True)
    raise ValueError(f"Unsupported file extension for {path}. Expected .pt, .pth, or .npy.")


def main() -> int:
    """Run the LAKER command-line interface."""
    from laker import __version__

    parser = argparse.ArgumentParser(
        description="LAKER: Learning-based Attention Kernel Regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command")

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
    fit_parser.add_argument("--gamma", type=float, default=1e-1, help="CCCP regularisation gamma")
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
        default="float64",
        choices=["float32", "float64"],
    )
    fit_parser.add_argument(
        "--kernel",
        default="exact",
        choices=["exact", "nystrom", "fourier", "neighbors", "grid", "spectrum", "hybrid"],
        help="Kernel approximation (default: exact).",
    )

    pred_parser = subparsers.add_parser("predict", help="Predict using a fitted model")
    pred_parser.add_argument("--model", required=True, help="Path to fitted model .pt file")
    pred_parser.add_argument("--locations", required=True, help="Path to query locations")
    pred_parser.add_argument("--output", required=True, help="Path to save predictions")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "fit":
        cmd_fit(args)
        return 0
    if args.command == "predict":
        cmd_predict(args)
        return 0
    parser.print_help()
    sys.exit(1)
    return 1  # unreachable, kept for static type checkers


def cmd_fit(args) -> None:
    """Handle the ``fit`` subcommand."""
    from laker.models import LAKERRegressor

    logger.info("Loading data...")
    x = load_tensor(args.locations)
    y = load_tensor(args.measurements)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    # Accept both legacy ``lambda_reg`` and current ``regularization``.
    regularization = getattr(args, "regularization", getattr(args, "lambda_reg", 1e-2))
    num_probes = getattr(args, "probes", getattr(args, "num_probes", None))
    model = LAKERRegressor(
        embedding_dim=args.embedding_dim,
        lambda_reg=regularization,
        gamma=args.gamma,
        num_probes=num_probes,
        device=args.device,
        dtype=dtype,
        verbose=True,
    )
    if hasattr(args, "kernel") and args.kernel and args.kernel != "exact":
        _KERNEL_MAP = {
            "nystrom": "nystrom",
            "fourier": "rff",
            "neighbors": "knn",
            "grid": "ski",
            "spectrum": "spectral",
            "hybrid": "twoscale",
        }
        model.kernel_approx = _KERNEL_MAP.get(args.kernel, args.kernel)
    model.fit(x, y)
    model.save(args.output)
    logger.info("Model saved to %s", args.output)


def cmd_predict(args) -> None:
    """Handle the ``predict`` subcommand."""
    from laker.models import LAKERRegressor

    logger.info("Loading model...")
    model = LAKERRegressor.load(args.model)
    x = load_tensor(args.locations)
    predictions = model.predict(x)
    torch.save(predictions, args.output)
    logger.info("Predictions saved to %s", args.output)


if __name__ == "__main__":
    sys.exit(main())
