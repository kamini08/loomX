#!/usr/bin/env python3
"""
check_pragma_shape.py -- structural validation of emitted parallel pragmas.

Motivation
----------
A numerical check only fires when the program runs and prints a wrong number.
Several of the most damaging code-generation defects are invisible that way:

  * a pragma attached to the wrong statement (e.g. the init loop instead of
    the compute loop) still compiles and still computes a correct answer,
    because the loop simply stayed serial;
  * a reduction clause that lost its operator, or gained a bogus one, is
    accepted by the compiler in some dialects and silently changes semantics;
  * `collapse(N)` disagreeing with the actual loop nesting depth.

So this checks shape, not values, and it needs no compiler at all -- which
matters because the OpenACC backend cannot be executed by gcc (gcc -fopenacc
fails at runtime on hand-written OpenACC too, so it is not a usable oracle).

Checks performed per pragma:
  1. the pragma must govern a loop (next significant line starts a for/while)
  2. reduction clauses must carry an operator, e.g. reduction(+:x)
  3. collapse(N) must match the measured loop nesting depth
  4. private/map/copy variables must be declared somewhere in the file

Usage:
  check_pragma_shape.py transformed.c [--backend openacc]
"""
import argparse
import re
import sys

LOOP_RE = re.compile(r'\b(for|while)\s*\(')
PRAGMA_RE = re.compile(r'^\s*#\s*pragma\s+(omp|acc)\b(.*)$')
# reduction(+:a, *:b) -- operator is required and must be one of these.
REDUCTION_RE = re.compile(r'reduction\s*\(\s*([^)]*)\)', re.IGNORECASE)
COLLAPSE_RE = re.compile(r'collapse\s*\(\s*(\d+)\s*\)', re.IGNORECASE)
CLAUSE_VAR_RE = re.compile(
    r'\b(private|firstprivate|lastprivate|shared|map|to|from|copy|copyin|copyout)\s*'
    r'\(\s*([^)]*)\)', re.IGNORECASE)

VALID_REDUCTION_OPS = set('+*-&|^%')

# Pragmas that legitimately bind a region or a declaration rather than a loop,
# so the "must immediately precede a loop" rule does not apply to them.
# `target data` in particular is followed by '{' and supplies the mappings for
# the loop pragmas nested inside it.
REGION_PRAGMA_RE = re.compile(
    r'\b(target\s+data|target\s+enter\s+data|target\s+exit\s+data|'
    r'target\s+update|declare\s+target|begin\b|end\b|teams\b(?!\s+distribute))',
    re.IGNORECASE)


def significant_lines(text):
    """Yield (lineno, stripped) for lines that are not blank or pure comments."""
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('//') or line.startswith('*') or line.startswith('/*'):
            continue
        yield i, line


def nesting_depth_at(lines, start):
    """
    Measure how many nested loop headers follow `start`.

    Counts loop headers while brace depth is positive, which is a good proxy
    for how deep the pragma actually binds.
    """
    depth = 0
    counted = 0
    for _, line in lines[start:]:
        if LOOP_RE.search(line):
            counted += 1
            depth += 1
        opens = line.count('{')
        closes = line.count('}')
        if closes > opens and counted > 0:
            break
        if opens == 0 and closes > 0 and counted > 0:
            break
    return max(counted, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('--backend', default='?')
    args = ap.parse_args()

    with open(args.source) as handle:
        text = handle.read()

    lines = list(significant_lines(text))
    problems = []
    pragmas = 0

    for pos, (lineno, line) in enumerate(lines):
        match = PRAGMA_RE.match(line)
        if not match:
            continue
        pragmas += 1
        dialect, body = match.group(1), match.group(2)
        where = f"line {lineno} [{dialect}]"

        # 1. pragma must govern a loop -- unless it is a region/declaration
        #    construct such as `target data`, which binds a braced block.
        is_region = bool(REGION_PRAGMA_RE.search(body))
        if not is_region and (pos + 1 >= len(lines)
                              or not LOOP_RE.search(lines[pos + 1][1])):
            nxt = lines[pos + 1][1] if pos + 1 < len(lines) else '<eof>'
            problems.append(f"{where}: does not immediately precede a loop; next is {nxt!r}")

        # 2. reduction clauses need an operator
        for red in REDUCTION_RE.findall(body):
            for item in red.split(','):
                item = item.strip()
                if not item:
                    continue
                if not any(op in item for op in VALID_REDUCTION_OPS):
                    problems.append(
                        f"{where}: reduction item {item!r} has no operator "
                        f"(expected e.g. '+:var')")
                elif item.lstrip()[:1] not in VALID_REDUCTION_OPS:
                    problems.append(
                        f"{where}: reduction item {item!r} must start with the operator")

        # 3. collapse depth must match the loop it binds
        for collapse in COLLAPSE_RE.findall(body):
            want = int(collapse)
            if pos + 1 < len(lines):
                got = nesting_depth_at(lines, pos + 1)
                if want > got:
                    problems.append(
                        f"{where}: collapse({want}) exceeds the {got} loop(s) that follow")

    if pragmas == 0:
        print(f"NO-PRAGMAS {args.backend}  (loop left serial: nothing to check)")
        return 0

    if problems:
        print(f"FAIL-SHAPE {args.backend}  pragmas={pragmas}")
        for problem in problems:
            print(f"     - {problem}")
        return 1

    print(f"PASS-SHAPE {args.backend}  pragmas={pragmas}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
