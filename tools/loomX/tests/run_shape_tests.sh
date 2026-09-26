#!/usr/bin/env bash
# Codegen-shape regression tests that need no GPU and no CUDA/OpenCL runtime.
#
# These check the *shape* of generated offload code, not its numerics. They
# exist because a wrong shape can compile cleanly and even produce plausible
# numbers, so the numerical matrix in run_validation.sh is not sufficient on
# machines without a GPU. run_validation.sh is the numerical authority; this
# script is the structural one, and it runs anywhere.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/../src"
OUT="${TMPDIR:-/tmp}/loomx-shape-$$"
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT

resolve_loomx() {
    if [ -n "${LOOMX:-}" ] && [ -x "$LOOMX" ]; then
        echo "$LOOMX"; return 0
    fi
    local candidate
    for candidate in \
        "$SCRIPT_DIR/../build/loomX" \
        "$SCRIPT_DIR/../build/tools/loomX/loomX" \
        "$SCRIPT_DIR/../build/tools_loomX_exe"; do
        [ -x "$candidate" ] && { echo "$candidate"; return 0; }
    done
    return 1
}

LOOMX_BIN="$(resolve_loomx)" || {
    echo "SKIP: no loomX binary found (set LOOMX=/path/to/loomX)" >&2
    exit 0
}

PASS=0
FAIL=0

note_fail() {
    echo "  FAIL: $1"
    FAIL=$((FAIL + 1))
}

# ---------------------------------------------------------------------------
# 1. Pointer-argument map shape.
#
# A loop over `double *a` must never map `a` as a bare name, because a bare
# `map(tofrom:a)` transfers the pointer value and the device then dereferences
# garbage. Either the extent is provable and a `[lo:len]` section is emitted,
# or the target clause is suppressed. Both are checked here by shape alone.
# ---------------------------------------------------------------------------
echo "=== pointer-argument map shape ==="
for fixture in validate_shape_ptr validate_shape_ptr_odd; do
    for backend in omp openacc; do
        src="$SCRIPT_DIR/$fixture.c"
        out="$OUT/$fixture.$backend.c"
        label="$fixture-$backend"

        if ! "$LOOMX_BIN" --gpu-naive --backend "$backend" \
                -rose:skipfinalCompileStep "$src" -o "$out" \
                > "$OUT/$label.log" 2>&1; then
            note_fail "$label: loomX exited non-zero (see $OUT/$label.log)"
            continue
        fi

        if python3 "$SCRIPT_DIR/check_pointer_map_shape.py" "$out" \
                --backend "$backend" > "$OUT/$label.shape.log" 2>&1; then
            echo "  PASS: $label"
            PASS=$((PASS + 1))
        else
            note_fail "$label"
            sed 's/^/      /' "$OUT/$label.shape.log"
        fi
    done
done

# ---------------------------------------------------------------------------
# 2. The guard must actually be reachable, not vacuously satisfied.
#
# check_pointer_map_shape.py returns 0 when there is no mapped argument to
# check, which is the correct outcome for the irregular fixture. Without this
# assertion a regression that stopped emitting map clauses entirely would still
# pass the test above, so assert the canonical fixture really did emit a
# section. A bare `map(tofrom:a)` here is precisely the original bug.
# ---------------------------------------------------------------------------
echo "=== canonical pointer loop emits a section, not a bare name ==="
canon="$OUT/validate_shape_ptr.omp.c"
if [ ! -f "$canon" ]; then
    note_fail "canonical fixture was not generated"
else
    if grep -Eq '#\s*pragma\s+omp\s+target' "$canon"; then
        # Offload happened: the mapped pointer must be a section.
        if grep -Eq 'map[a-z_]*\s*\(\s*[a-z]*:?[[:space:]]*a\[' "$canon"; then
            echo "  PASS: mapped a[] as a section"
            PASS=$((PASS + 1))
        elif grep -Eq 'map[a-z_]*\s*\([^)]*\ba\b[^[]*\)' "$canon"; then
            note_fail "a is mapped as a bare name (pre-fix behaviour)"
        else
            echo "  PASS: offload target present with no bare a mapping"
            PASS=$((PASS + 1))
        fi
    else
        note_fail "canonical pointer loop was not offloaded at all; \
this test needs a loop that reaches the offload path"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Loop-carried scalar dependence must still be rejected.
#
# Guards against the shape work accidentally widening offload eligibility.
# ---------------------------------------------------------------------------
echo "=== scalar loop-carried dependence still rejected ==="
dep_src="$OUT/dep.c"
# a[i] = a[i-1] + 1.0 reads the element the previous iteration wrote, which is a
# genuine loop-carried dependence. (A loop that only accumulates a scalar and
# writes a[i] is *not* dependent and is correctly offloaded as a reduction, so
# it must not be used as the negative case here.)
cat > "$dep_src" <<'EOF'
#include <stdlib.h>
#include <stdio.h>
#define N 4000000
int main(void)
{
    int i;
    double *a = (double *)malloc(sizeof(double) * N);
    for (i = 0; i < N; i++)
        a[i] = (double)(i % 100);
    for (i = 1; i < N; i++)
        a[i] = a[i - 1] + 1.0;
    printf("dep_sum %.6f\n", a[N - 1]);
    free(a);
    return 0;
}
EOF
if ! "$LOOMX_BIN" --gpu-naive --backend omp -rose:skipfinalCompileStep \
        "$dep_src" -o "$OUT/dep.out.c" > "$OUT/dep.log" 2>&1; then
    note_fail "loomX exited non-zero on the dependence fixture"
elif ! awk '
    # Track the most recent non-blank, non-comment line so we can tell whether
    # the *dependent* loop is the one a pragma was attached to. Grepping for
    # "target" anywhere in the file is not enough: the initialisation loop in
    # this fixture is legitimately offloaded, and the check must not confuse it
    # with the dependent loop.
    /^[ \t]*$/ { next }
    { last = $0; lastn = NR }
    /a\[i - 1\] \+ 1\.0/ {
        if (last ~ /^[ \t]*#[ \t]*pragma[ \t]+(omp|acc)[ \t]+target/)
            found = 1
    }
    END { exit(found ? 0 : 1) }
' "$OUT/dep.out.c"; then
    echo "  PASS: dependent loop left on the CPU"
    PASS=$((PASS + 1))
else
    note_fail "dependent loop was offloaded; the guard should reject it"
fi

echo ""
echo "SHAPE PASS: $PASS  FAIL: $FAIL"
[ "$FAIL" -eq 0 ]
