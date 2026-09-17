#!/usr/bin/env python3
"""
check_correctness.py -- numerically compare a parallel/GPU-offloaded
binary's output against a sequential golden reference, with tolerance
(needed because parallel reductions reorder floating-point sums).

Assumes each binary prints its output array as whitespace/newline-separated
numbers to stdout. If your benchmarks dump a binary blob instead, swap the
`load()` function for a struct/np.fromfile reader.

Usage:
  ./bin/gemm__seq            2048 > golden.out
  ./bin/gemm__gpu_profitable 2048 > candidate.out
  python3 check_correctness.py golden.out candidate.out --rtol 1e-5 --atol 1e-8
"""
import argparse
import sys

import numpy as np


def load(path):
    with open(path) as f:
        return np.array([float(x) for x in f.read().split()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    g, c = load(args.golden), load(args.candidate)
    if g.shape != c.shape:
        print(f"FAIL: shape mismatch golden={g.shape} candidate={c.shape}")
        sys.exit(1)

    close_mask = np.isclose(g, c, rtol=args.rtol, atol=args.atol)
    ok = bool(close_mask.all())
    diff = np.abs(g - c)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / (np.abs(g) + args.atol)).max()) if diff.size else 0.0
    n_bad = int((~close_mask).sum())

    print(f"elements={g.size}  max_abs_diff={max_abs:.3e}  "
          f"max_rel_diff={max_rel:.3e}  mismatches={n_bad}/{g.size}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
