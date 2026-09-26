#!/usr/bin/env python3
"""
validate_results.py -- keyed numerical comparison of two program outputs.

Why this exists
---------------
`tests/run_tests.sh` used to compare only process exit codes. Exit status is
truncated to 8 bits by POSIX, so a checksum of 500500, 500756 or 501012 all
present as exit code 20. That check therefore passes for a wide band of wrong
answers, and it cannot report *how* wrong a result is.

The kernels in `tests/validate/` print `<key> <value>` lines at full precision.
This script matches them by key and compares numerically, so:

  * a missing or renamed key is a hard failure (not silently tolerated),
  * integer-valued outputs are compared exactly, since reassociating an
    integer sum is still exact,
  * floating-point outputs are compared with an explicit tolerance, because a
    correct parallel reduction reorders additions and the last bits will move.

Usage:
  ./prog                      > golden.out
  ./prog_parallel             > candidate.out
  validate_results.py golden.out candidate.out [--rtol R] [--atol A]
                                  [--exact-keys] [--backend openacc]
                                  [--kernel v_gemm.c]
"""
import argparse
import sys

FLOATY = (int, float)


def load(path):
    """Parse `<key> <value>` lines into an ordered dict of strings."""
    fields = {}
    with open(path) as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.partition(' ')
            if not sep:
                raise SystemExit(f"{path}:{lineno}: expected '<key> <value>', got {line!r}")
            if key in fields:
                raise SystemExit(f"{path}:{lineno}: duplicate key {key!r}")
            fields[key] = value.strip()
    return fields


def classify(value):
    """Return ('int', int) / ('float', float) / ('str', str) for a token."""
    try:
        return 'int', int(value)
    except ValueError:
        pass
    try:
        return 'float', float(value)
    except ValueError:
        return 'str', value


def is_exact_reference(key):
    """
    Keys whose value must match bit-for-bit.

    Integer reductions and permutations/transposes are exactly reproducible
    under reordering, so any deviation is a genuine defect rather than benign
    floating-point drift. Float sums are the opposite: they are expected to
    move in the low bits.
    """
    for marker in ('int_', 'hist_', 'transpose_', 'sub_reduce'):
        if key.startswith(marker):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('golden')
    ap.add_argument('candidate')
    ap.add_argument('--rtol', type=float, default=1e-8,
                    help='relative tolerance for inexact float keys')
    ap.add_argument('--atol', type=float, default=1e-12,
                    help='absolute tolerance floor, also guards /0')
    ap.add_argument('--exact-keys', action='store_true',
                    help='require every integer/permute key to match exactly')
    ap.add_argument('--kernel', default='?')
    ap.add_argument('--backend', default='?')
    args = ap.parse_args()

    golden = load(args.golden)
    cand = load(args.candidate)

    failures = []
    checked = 0
    max_rel = 0.0
    max_abs = 0.0

    # Key set must match exactly. A dropped key is the failure mode that
    # flattening all numbers into one list would hide.
    only_golden = sorted(set(golden) - set(cand))
    only_cand = sorted(set(cand) - set(golden))
    for key in only_golden:
        failures.append(f"missing key {key!r} (serial printed it, transformed did not)")
    for key in only_cand:
        failures.append(f"unexpected key {key!r} (transformed printed it, serial did not)")

    for key in sorted(set(golden) & set(cand)):
        gkind, gval = classify(golden[key])
        ckind, cval = classify(cand[key])
        if gkind != ckind:
            failures.append(f"{key}: type changed {gkind} -> {ckind} "
                            f"({golden[key]!r} vs {cand[key]!r})")
            continue
        if gkind == 'str':
            if gval != cval:
                failures.append(f"{key}: text differs {gval!r} vs {cval!r}")
            checked += 1
            continue

        checked += 1

        # NaN/Inf must be handled explicitly, otherwise every comparison
        # against them is False and the reason is invisible in the report.
        if isinstance(gval, float) and isinstance(cval, float):
            if gval != gval or cval != cval:  # NaN
                if (gval != gval) != (cval != cval):
                    failures.append(f"{key}: NaN mismatch {gval} vs {cval}")
                continue
            if gval in (float('inf'), float('-inf')) or cval in (float('inf'), float('-inf')):
                if gval != cval:
                    failures.append(f"{key}: inf mismatch {gval} vs {cval}")
                continue

        adiff = abs(gval - cval)
        denom = abs(gval) if gval != 0 else 1.0
        rdiff = adiff / denom
        max_abs = max(max_abs, adiff)
        max_rel = max(max_rel, rdiff)

        if gkind == 'int' or is_exact_reference(key) or args.exact_keys:
            if gval != cval:
                failures.append(f"{key}: EXACT mismatch {gval} vs {cval} (delta {gval - cval})")
        else:
            if adiff > (args.atol + args.rtol * abs(gval)):
                failures.append(f"{key}: {gval!r} vs {cval!r} "
                                f"abs={adiff:.3e} rel={rdiff:.3e} "
                                f"exceeds rtol={args.rtol:g}")

    label = f"{args.kernel}/{args.backend}"
    if failures:
        print(f"FAIL {label}  keys={checked} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        for line in failures:
            print(f"     - {line}")
        return 1

    print(f"PASS {label}  keys={checked} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
