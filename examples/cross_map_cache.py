"""Build the cross-map train-mean cache once, for reuse across runs.

Usage::

    python -m examples.cross_map_cache --workers 8

Writes ``<data-dir>/cross_map_mean.npy`` and prints a one-line summary.
The file is reused by ``examples.scalable --cross-map`` on subsequent runs.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from examples.scalable import Scalable
from examples.scalable_data import ScalableData


class _NullEvents:
    def write(self, *_args, **_kwargs) -> None:
        pass

    def flush(self) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ucf50k")
    parser.add_argument("--unpack-dir", default="data/ucf50k/unpacked")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ScalableData.prepare(
        data_dir=args.data_dir,
        unpacked_dir=args.unpack_dir,
        download=False,
        extract=False,
        verify=False,
        seed=0,
    )
    entries = ScalableData.index(args.unpack_dir)
    cache = os.path.join(args.data_dir, "cross_map_mean.npy")
    if os.path.exists(cache):
        print(f"cache already exists at {cache}; delete to rebuild.")
        return 0
    mean = Scalable.cross_map_mean(
        entries, args.workers, args.data_dir, _NullEvents()
    )
    covered = int((~np.isnan(mean)).sum())
    print(
        f"wrote {cache}  shape={mean.shape}  covered_pixels={covered}  "
        f"mean_of_means={float(np.nanmean(mean)):.4f} dBm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
