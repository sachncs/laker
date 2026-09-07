"""Score the cached cross-map mean against every map in the corpus.

Usage::

    python -m examples.cross_map_score --workers 8

Writes ``<data-dir>/cross_map_score.csv`` and prints a per-split summary.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

from examples.scalable_data import ScalableData


def _score_one(entry: dict, mean_map: np.ndarray) -> Optional[dict]:
    scene = ScalableData.clean(entry)
    valid_idx = np.flatnonzero(scene.valid)
    pred = np.asarray(mean_map).ravel()[valid_idx]
    truth = scene.radio.ravel()[valid_idx]
    good = np.isfinite(pred)
    if not good.any():
        return None
    err = pred[good] - truth[good]
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "scene": f"{scene.split}/{scene.name}",
        "split": scene.split,
        "rmse": rmse,
        "coverage": float(valid_idx.size / scene.radio.size),
        "evaluated_pixels": int(good.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ucf50k")
    parser.add_argument("--unpack-dir", default="data/ucf50k/unpacked")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits", default="test,val,train")
    parser.add_argument(
        "--out",
        default=None,
        help="output CSV path (default: <data-dir>/cross_map_score.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cache = os.path.join(args.data_dir, "cross_map_mean.npy")
    if not os.path.exists(cache):
        print(
            f"missing {cache}; build with `python -m examples.cross_map_cache` first.",
            file=sys.stderr,
        )
        return 1
    mean_map = np.load(cache)
    assert mean_map.shape == (256, 256), f"bad cache shape: {mean_map.shape}"

    ScalableData.prepare(
        data_dir=args.data_dir,
        unpacked_dir=args.unpack_dir,
        download=False,
        extract=False,
        verify=False,
        seed=0,
    )
    entries = ScalableData.index(args.unpack_dir)
    want = set(args.splits.split(","))
    pool = [entry for entry in entries if entry["split"] in want]
    print(f"scoring {len(pool)} maps with {args.workers} workers...")

    out_csv = args.out or os.path.join(args.data_dir, "cross_map_score.csv")
    rows: list[dict] = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_score_one, entry, mean_map): entry for entry in pool}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result is not None:
                    rows.append(result)
                if done % 5000 == 0:
                    print(f"  {done}/{len(pool)}")
    else:
        for entry in pool:
            result = _score_one(entry, mean_map)
            if result is not None:
                rows.append(result)

    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["scene", "split", "rmse", "coverage", "evaluated_pixels"]
        )
        writer.writeheader()
        writer.writerows(rows)
    rmses = np.array([r["rmse"] for r in rows])
    print(
        f"\ncross-map (per-pixel train mean) RMSE on {len(rows)} maps: "
        f"{rmses.mean():.2f} +/- {rmses.std():.2f} dB  "
        f"(median {float(np.median(rmses)):.2f}, range {rmses.min():.2f} .. {rmses.max():.2f})"
    )
    for split in sorted({r["split"] for r in rows}):
        s = np.array([r["rmse"] for r in rows if r["split"] == split])
        print(
            f"  {split:<6}: n={len(s)} mean={s.mean():.2f} +/- {s.std():.2f}  "
            f"median={float(np.median(s)):.2f}"
        )
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
