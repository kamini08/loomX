# loomX benchmarking harness

Five scripts, one workflow: fetch suites -> compile 4 configs per benchmark
-> lock GPU clocks -> time + profile each config N times -> check numerical
correctness -> aggregate into speedup numbers.

The four configs per benchmark (this is what makes the ablation experiments
possible):

| config            | what it is                                         |
|-------------------|-----------------------------------------------------|
| `seq`             | untouched sequential source, `-O3`, no pragmas       |
| `cpu_omp`         | loomX's CPU-threaded fallback path                   |
| `gpu_naive`       | loomX with the profitability gate forced off (offload everything proven safe) -- ablation baseline |
| `gpu_profitable`  | loomX with the profitability gate on (the actual system) |

## 0. One-time environment checks

```bash
# Confirm compilers and GPU compute capability
clang --version          # must support -fopenmp-targets=nvptx64-nvidia-cuda for GPU configs
gcc --version            # used for seq / cpu_omp
nvidia-smi --query-gpu=compute_cap,name --format=csv
# sm_XX in commands below = compute_cap with the dot removed, e.g. 8.0 -> sm_80
```

**Important:** GPU OpenMP offload requires a clang built with `openmp` in
`LLVM_ENABLE_PROJECTS` and the matching `libomptarget-nvptx.bc` bitcode library.
If `clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda` fails with
`no library 'libomptarget-nvptx.bc' found`, the GPU configs will be skipped
automatically and the harness reports CPU-only numbers.  See
`build_gpu_clang.md` for the LLVM build configuration that produces a working
offload compiler.

## 1. Fetch benchmark suites

```bash
chmod +x setup_benchmarks.sh lock_gpu_clocks.sh
./setup_benchmarks.sh ./bench-suites
```

This pulls PolyBench/C 4.2.1, Rodinia, and DataRaceBench.  The custom
`interproc-microbench/` corpus (already included next to this README) stresses
real function calls in hot loops, including cases a naive inliner cannot solve
(recursive calls and function pointers) plus negative controls that are
genuinely unsafe.

## 2. Run the full driver

```bash
./run_benchmarks.sh interproc-microbench
```

This single command generates the four source variants, compiles them, checks
correctness, runs the timing harness, and prints the aggregate speedup table.

Environment variables to override defaults:

```bash
LOOMX=/path/to/loomX \
COMPILER_CPU=gcc COMPILER_GPU=clang \
GPU_ARCH=sm_80 \
RUNS=10 \
BENCH_ARGS="200000" \
LOCK_CLOCKS=yes \
./run_benchmarks.sh interproc-microbench
```

## 3. Manual variant generation (optional)

```bash
mkdir -p out bin
SRC=interproc-microbench/leaf_call.c

../loomX --cpu-only             "$SRC" -o out/leaf_call__cpu_omp.c
../loomX --gpu-naive            "$SRC" -o out/leaf_call__gpu_naive.c
../loomX --gpu-profitable       "$SRC" -o out/leaf_call__gpu_profitable.c
cp "$SRC" out/leaf_call__seq.c
```

## 4. Compile all four configs

```bash
BENCH=leaf_call

# Sequential
gcc -O3 out/${BENCH}__seq.c -lm -o bin/${BENCH}__seq

# CPU OpenMP
gcc -O3 -fopenmp out/${BENCH}__cpu_omp.c -lm -o bin/${BENCH}__cpu_omp

# GPU configs: match -march to YOUR compute_cap; requires offload-capable clang
clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  -Xopenmp-target -march=sm_80 \
  out/${BENCH}__gpu_naive.c -lm -o bin/${BENCH}__gpu_naive

clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  -Xopenmp-target -march=sm_80 \
  out/${BENCH}__gpu_profitable.c -lm -o bin/${BENCH}__gpu_profitable
```

## 5. Lock GPU clocks, then time + profile each config

```bash
./lock_gpu_clocks.sh lock

for cfg in seq cpu_omp gpu_naive gpu_profitable; do
  extra_flag=""
  [[ "$cfg" == gpu_* ]] && extra_flag="--nsys"
  python3 bench_harness.py \
    --binary bin/${BENCH}__${cfg} --args "200000" --runs 10 \
    --label "${BENCH}__${cfg}" --out results.csv $extra_flag
done

./lock_gpu_clocks.sh reset
```

## 6. Correctness checks

Numerical comparison against the sequential golden output:

```bash
bin/${BENCH}__seq      200000 > golden.out
bin/${BENCH}__cpu_omp  200000 > candidate.out
python3 check_correctness.py golden.out candidate.out --rtol 1e-5 --atol 1e-8
```

DataRaceBench parallel-safety ground truth (yes/no labels):

```bash
python3 check_dataracebench.py \
  --loomx ../loomX \
  --suite bench-suites/dataracebench/micro-benchmarks \
  --mode cpu-only \
  --output dataracebench_results.csv
```

This reports false positives (loomX accepted a `-yes` race) and false
negatives (loomX rejected a `-no` race-free case).

## 7. Aggregate into the numbers for your slides

```bash
python3 aggregate_results.py results.csv --baseline seq
```

Prints a per-benchmark speedup table for `cpu_omp` / `gpu_naive` /
`gpu_profitable` vs. `seq`, plus the geometric mean across the whole
benchmark set for each config. The `gpu_naive` vs `gpu_profitable` geomean
comparison is the headline "profitability gate beats offload-everything"
number.

## What to actually report

- Full end-to-end wall clock (from `median_s`), not kernel-only time.
- Geometric mean across benchmarks, not best-case or arithmetic mean.
- Median of >=10 runs with clocks locked; report the stdev column too.
- Exact hardware/software: GPU model + compute_cap, host CPU, CUDA/ROCm
  version, compiler + version used for offload codegen, problem sizes.
- If GPU configs could not be compiled, say so explicitly and report
  CPU-only speedups plus source-level GPU-offload decisions.
