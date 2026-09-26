#!/usr/bin/env python3
"""Compiler-independent regression test for pointer-argument offload shape.

Background
----------
An offloaded loop that walks a `double *` parameter (say `a`) used to render
its target data clause as a bare `map(tofrom:a)`.  For a *pointer* that name
denotes the host address value, not the array it points at, so the directive
transferred eight bytes of stack and the device then dereferenced whatever it
found there.  Every generated binary died with:

    CUDA error: an illegal memory access was encountered

...while the emitted code still looked reasonable, so a shape check that only
asked "is there a map clause" would have passed it.

The fix only ever emits a section that provably spans the whole allocation
(`a[0:n]` for a canonical `for (i = 0; i < n; i++)` walk), and otherwise
suppresses the target clause, leaving the loop on the CPU.  Both outcomes are
correct, so this test asserts the *invariant* rather than one exact spelling:

  * every mapped argument in a target clause is either a plain identifier or a
    well-formed `[lo:len]` section, and
  * a plain-identifier mapping never applies to an argument that the loop
    actually dereferences through a subscript.

What this deliberately does not do
----------------------------------
It does not compile, link, or run anything, so it works with no CUDA, no
OpenCL runtime, and no GPU.  It cannot prove the generated code is numerically
correct -- `run_validation.sh` does that on real hardware.  What it catches is
the class of bug where the shape is wrong but the numbers still come out
plausible.

Usage:  check_pointer_map_shape.py <translated.c> [--backend omp|openacc]
Exit 0 when the shape is acceptable, 1 with a diagnosis when it is not.
"""

import argparse
import re
import sys

# `#pragma omp target ...` / `#pragma acc target ...`, one line only.
TARGET_RE = re.compile(
    r'^\s*#\s*pragma\s+(omp\s+target|acc\s+target)\b(?P<rest>.*)$'
)

# map(...) / map_to(...) / map_from(...) argument lists.  Both the `map(...)`
# form and the combined `map(tofrom:...)` form are accepted.
MAP_RE = re.compile(
    r'\bmap(?:_to|_from)?\s*\(\s*(?P<clause>[^)]*)\)', re.IGNORECASE
)

# `name[0:n]`, `a[lo:len]`, allowing an optional space after the colon.
SECTION_RE = re.compile(r'^(?P<var>[A-Za-z_][A-Za-z0-9_]*)\[\s*[^:]+:[^\]]+\]$')

# A bare name, no brackets.  `a[*]` is deliberately not matched: it is a
# section shape this generator is not allowed to emit.
BARE_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# `a[...]` in ordinary code -- evidence the argument is dereferenced.
SUBSCRIPT_RE = re.compile(r'\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*\[')


def target_clauses(rest):
    """Yield the data clauses found on a target pragma line."""
    for match in MAP_RE.finditer(rest):
        yield match.group('clause')


def clause_items(clause):
    """Split `to: a[0:n], b[0:m]` into individual items.

    Bracketed sections can legally contain commas (`a[f(1,2):n]`), so the split
    tracks bracket depth rather than calling str.split(',').  Separators are
    commas and whitespace runs, since a clause may be written `tofrom: a` or
    `tofrom:a` or with several space-separated entries.
    """
    items = []
    current = []
    depth = 0
    for char in clause:
        if char == '[':
            depth += 1
        elif char == ']':
            depth = max(0, depth - 1)
        if char == ',' and depth == 0:
            items.append(''.join(current))
            current = []
        else:
            current.append(char)
    items.append(''.join(current))
    return [item.strip() for item in items if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--backend', default='?')
    args = parser.parse_args()

    with open(args.source) as handle:
        text = handle.read()

    lines = text.splitlines()

    problems = []
    checked = 0
    guarded = 0

    for lineno, line in enumerate(lines, 1):
        target = TARGET_RE.match(line)
        if not target:
            continue
        rest = target.group('rest')
        clauses = list(target_clauses(rest))
        if not clauses:
            # No explicit mapping: the enclosing region or firstprivate
            # reference-sharing rules decide.  Not this test's business.
            continue

        # Names the loop body dereferences, e.g. `a[i * 4 + 2]`.
        dereferenced = {
            m.group('var')
            for m in SUBSCRIPT_RE.finditer('\n'.join(lines[lineno:lineno + 40]))
        }

        for clause in clauses:
            for item in clause_items(clause):
                # Drop a leading direction marker such as `to:` / `from:`.
                if ':' in item and not item.startswith('['):
                    marker, _, remainder = item.partition(':')
                    if marker.strip().lower() in (
                        'to', 'from', 'tofrom', 'alloc', 'alloc_shape',
                    ):
                        item = remainder.strip()
                if not item:
                    continue

                checked += 1
                if BARE_RE.match(item):
                    if item in dereferenced:
                        problems.append(
                            f"{args.source}:{lineno}: `{item}` is mapped as a bare "
                            f"name but the loop body dereferences it as "
                            f"`{item}[...]`. A bare map transfers the pointer "
                            f"value, not the array it points at."
                        )
                    continue
                if SECTION_RE.match(item):
                    continue
                # `a[*]` and anything else unrecognised: refuse it rather than
                # guess what extent it meant.
                problems.append(
                    f"{args.source}:{lineno}: mapped argument `{item}` is neither "
                    f"a plain identifier nor a `[lo:len]` section"
                )

    # A suppressed offload is a legitimate outcome; report it so the run shows
    # the guard firing rather than silently producing no target region.
    if checked == 0 and re.search(r'^\s*#\s*pragma\s+(omp|acc)\s+target',
                                  text, re.MULTILINE):
        guarded += 1

    if problems:
        for problem in problems:
            print(problem)
        return 1

    if guarded:
        print(f"pointer map shape ok ({args.backend}): no mapped argument to "
              f"check; offload was suppressed for this loop")
    return 0


if __name__ == '__main__':
    sys.exit(main())
