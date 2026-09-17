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
import math
import sys


def load(path):
    with open(path) as f:
        return [float(x) for x in f.read().split()]


def isclose(a, b, rtol, atol):
    return abs(a - b) <= (atol + rtol * abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    g, c = load(args.golden), load(args.candidate)
    if len(g) != len(c):
        print(f"FAIL: shape mismatch golden={len(g)} candidate={len(c)}")
        sys.exit(1)

    if not g:
        print("elements=0  max_abs_diff=0.000e+00  max_rel_diff=0.000e+00  mismatches=0/0")
        print("PASS")
        sys.exit(0)

    max_abs = 0.0
    max_rel = 0.0
    n_bad = 0
    for gv, cv in zip(g, c):
        adiff = abs(gv - cv)
        max_abs = max(max_abs, adiff)
        rdiff = adiff / (abs(gv) + args.atol)
        max_rel = max(max_rel, rdiff)
        if not isclose(gv, cv, args.rtol, args.atol):
            n_bad += 1

    ok = n_bad == 0
    print(f"elements={len(g)}  max_abs_diff={max_abs:.3e}  "
          f"max_rel_diff={max_rel:.3e}  mismatches={n_bad}/{len(g)}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
