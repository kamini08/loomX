#!/usr/bin/env bash
# Run loomX tests and compare original vs transformed exit codes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOOMX_BIN="${LOOMX:-${SCRIPT_DIR}/../../build/loomX}"
TEST_DIR="${SCRIPT_DIR}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASS=0
FAIL=0
FAILED_TESTS=""

if [[ ! -x "$LOOMX_BIN" ]]; then
    # Fallback to the documented out-of-tree build location.
    LOOMX_BIN="/tmp/opencode/loomX-cmake-build/loomX"
fi

if [[ ! -x "$LOOMX_BIN" ]]; then
    echo "ERROR: loomX binary not found at $LOOMX_BIN" >&2
    exit 1
fi

for src in "$TEST_DIR"/*.c; do
    name="$(basename "$src" .c)"
    echo "=== $name ==="

    # Run loomX.
    if ! "$LOOMX_BIN" "$src" -rose:skipfinalCompileStep -o "$WORK_DIR/rose_${name}.c" >"$WORK_DIR/${name}.loomx.log" 2>&1; then
        echo "  loomX failed; see $WORK_DIR/${name}.loomx.log"
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS $name(loomX)"
        continue
    fi

    # Compile original.
    if ! gcc -O2 -fopenmp "$src" -o "$WORK_DIR/${name}.orig" -lm >"$WORK_DIR/${name}.orig.build.log" 2>&1; then
        echo "  original build failed; see $WORK_DIR/${name}.orig.build.log"
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS $name(orig-build)"
        continue
    fi

    # Compile transformed.
    if ! gcc -O2 -fopenmp "$WORK_DIR/rose_${name}.c" -o "$WORK_DIR/${name}.rose" -lm >"$WORK_DIR/${name}.rose.build.log" 2>&1; then
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
echo "PASS: $PASS  FAIL: $FAIL"
if [[ -n "$FAILED_TESTS" ]]; then
    echo "Failed:$FAILED_TESTS"
    exit 1
fi
exit 0
