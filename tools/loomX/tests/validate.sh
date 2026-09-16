#!/bin/bash
# Validation script for loomX
# Compiles sequential and GPU versions, compares outputs

GPU_PAR="/home/kamini/projects/hackathon-rose/loomX/loomX"
NVC="/home/kamini/projects/hackathon-rose/nvhpc_2025_253_Linux_x86_64_cuda_12.8/install_components/Linux_x86_64/25.3/compilers/bin/nvc"
TESTDIR="/home/kamini/projects/hackathon-rose/loomX/tests"
export LD_LIBRARY_PATH=/home/kamini/projects/hackathon-rose/rose-install/lib:/home/kamini/projects/hackathon-rose/boost-install/lib:/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH

run_test() {
    local name=$1
    local src="${TESTDIR}/${name}.c"
    
    echo "=== Testing ${name} ==="
    
    # Run loomX
    cd "$TESTDIR"
    "$GPU_PAR" -rose:skipfinalCompileStep "${name}.c" > /dev/null 2>&1
    
    # Compile sequential
    "$NVC" -o "${name}_seq" "${name}.c" 2>&1 | grep -v "warning:\|Remark:" || true
    
    # Compile GPU
    "$NVC" -mp=gpu -gpu=cc120 -o "${name}_gpu" "rose_${name}.c" 2>&1 | grep -v "warning:\|Remark:" || true
    
    # Run and compare exit codes (do not let non-zero exit codes abort the script)
    local seq_code gpu_code
    "./${name}_seq"; seq_code=$?
    "./${name}_gpu"; gpu_code=$?
    
    if [ "$seq_code" -eq "$gpu_code" ]; then
        echo "PASS: Exit codes match ($seq_code)"
    else
        echo "FAIL: Sequential=$seq_code, GPU=$gpu_code"
    fi
    echo ""
}

# Run tests
run_test "benchmark2"
run_test "benchmark3"
run_test "interprocedural_demo"
run_test "divergence_test"
run_test "side_effect_test"
run_test "polybench_gemm"

# Interprocedural-analysis feature tests
run_test "transitive_global"
run_test "pointer_modref"
run_test "recursive"
run_test "pure_stdlib"
run_test "disjoint_pointer"
run_test "reduction_local"
run_test "reduction_sub"

# Baseline-mode check: intraprocedural tool must reject disjoint_pointer.
echo "=== Checking --intraprocedural-baseline rejects disjoint_pointer ==="
cd "$TESTDIR"
if "$GPU_PAR" --intraprocedural-baseline -rose:skipfinalCompileStep "disjoint_pointer.c" 2>&1 | grep -q "Function call not safe for parallelization"; then
    echo "PASS: Baseline mode correctly rejects disjoint pointer write"
else
    echo "FAIL: Baseline mode did not reject disjoint pointer write"
fi
echo ""

echo "=== Checking loop-local temporaries are not reductions ==="
if [ -f "rose_reduction_local.c" ] && ! grep -q "reduction(" "rose_reduction_local.c"; then
    echo "PASS: No reduction clause emitted for loop-local temporary"
else
    echo "FAIL: Unexpected reduction clause on loop-local temporary"
fi
echo ""

echo "=== Validation complete ==="
