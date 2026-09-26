#!/usr/bin/env bash
# Run loomX tests and compare original vs transformed exit codes.
#
# This is the smoke test: it proves every tests/*.c still transforms and
# builds. It is NOT a correctness test. Exit status is truncated to 8 bits by
# POSIX, so a checksum of 500500, 500756 and 501012 all present as exit code 20
# -- a wide band of wrong answers passes this check, and it cannot say how
# wrong. Numerical validation lives in run_validation.sh, which runs below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the translator inside the tree. The old default,
# "${SCRIPT_DIR}/../../build/loomX", pointed at <repo>/build/loomX, which is
# not a directory this project builds into, so the script only ever worked via
# the /tmp fallback below.
resolve_loomx() {
    if [[ -n "${LOOMX:-}" ]]; then
        printf '%s' "$LOOMX"
        return
    fi
    local candidate
    for candidate in \
        "$SCRIPT_DIR/../build/loomX" \
        "$SCRIPT_DIR/../../build/loomX" \
        "/tmp/opencode/loomX-cmake-build/loomX"; do
        if [[ -x "$candidate" && ! -d "$candidate" ]]; then
            printf '%s' "$candidate"
            return
        fi
    done
    printf '%s' "$SCRIPT_DIR/../build/loomX"
}

LOOMX_BIN="$(resolve_loomx)"
TEST_DIR="${SCRIPT_DIR}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASS=0
FAIL=0
FAILED_TESTS=""

if [[ ! -x "$LOOMX_BIN" ]]; then
    echo "ERROR: loomX binary not found at $LOOMX_BIN" >&2
    echo "       build it, or set LOOMX=/path/to/loomX" >&2
    exit 1
fi

echo "loomX: $LOOMX_BIN"
echo

for src in "$TEST_DIR"/*.c; do
    name="$(basename "$src" .c)"
    # Skip previously transformed outputs and support headers.
    if [[ "$name" == rose_* ]] || [[ "$name" == bench_minimal ]]; then
        continue
    fi
    echo "=== $name ==="

    # Run loomX.
    if ! "$LOOMX_BIN" "$src" -rose:skipfinalCompileStep -o "$WORK_DIR/rose_${name}.c" >"$WORK_DIR/${name}.loomx.log" 2>&1; then
        echo "  loomX failed; see $WORK_DIR/${name}.loomx.log"
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS $name(loomX)"
        continue
    fi

    # Compile original.
    if ! gcc -O2 -fopenmp -I"$TEST_DIR" "$src" -o "$WORK_DIR/${name}.orig" -lm >"$WORK_DIR/${name}.orig.build.log" 2>&1; then
        echo "  original build failed; see $WORK_DIR/${name}.orig.build.log"
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS $name(orig-build)"
        continue
    fi

    # Compile transformed.
    if ! gcc -O2 -fopenmp -I"$TEST_DIR" "$WORK_DIR/rose_${name}.c" -o "$WORK_DIR/${name}.rose" -lm >"$WORK_DIR/${name}.rose.build.log" 2>&1; then
        echo "  transformed build failed; see $WORK_DIR/${name}.rose.build.log"
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS $name(rose-build)"
        continue
    fi

    # Run and capture exit codes.
    orig_rc=0
    "$WORK_DIR/${name}.orig" >"$WORK_DIR/${name}.orig.out" 2>&1 || orig_rc=$?
    rose_rc=0
    "$WORK_DIR/${name}.rose" >"$WORK_DIR/${name}.rose.out" 2>&1 || rose_rc=$?

    if [[ "$orig_rc" == "$rose_rc" ]]; then
        echo "  PASS (rc=$orig_rc)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: original rc=$orig_rc, transformed rc=$rose_rc"
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS $name(rc:$orig_rc->$rose_rc)"
    fi
done

echo ""
echo "===================================="
echo "SMOKE (exit-code) PASS: $PASS  FAIL: $FAIL"
if [[ -n "$FAILED_TESTS" ]]; then
    echo "Failed:$FAILED_TESTS"
    SMOKE_FAILED=1
else
    SMOKE_FAILED=0
fi

# ---------------------------------------------------------------------------
# Codegen-shape regression tests. These need no GPU, no CUDA and no OpenCL
# runtime, and they catch a wrong map-clause shape that would still compile and
# could still produce plausible numbers. Kept separate from both the smoke and
# the numerical checks because it answers a third question: not "does it build"
# and not "is the answer right", but "is the offload directive attached to the
# right thing with the right extent".
# ---------------------------------------------------------------------------
echo ""
echo "===================================="
echo "CODEGEN SHAPE (no GPU required)"
echo "===================================="
SHAPE_FAILED=0
if ! "$SCRIPT_DIR/run_shape_tests.sh"; then
    SHAPE_FAILED=1
fi

# ---------------------------------------------------------------------------
# Numerical validation. Runs regardless of the smoke result, because it is a
# different question: not "does it still build", but "is the answer right".
# ---------------------------------------------------------------------------
echo ""
echo "===================================="
echo "NUMERICAL VALIDATION"
echo "===================================="
VALIDATION_FAILED=0
if ! "$SCRIPT_DIR/run_validation.sh"; then
    VALIDATION_FAILED=1
fi

if [[ "$SMOKE_FAILED" -ne 0 || "$SHAPE_FAILED" -ne 0 || \
      "$VALIDATION_FAILED" -ne 0 ]]; then
    exit 1
fi
exit 0
