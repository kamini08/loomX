# AGENTS.md — loomX

loomX is a ROSE-based source-to-source tool in `tools/loomX/`. The rest of the repository is the ROSE compiler infrastructure; limit editing to `tools/loomX/` unless you are changing ROSE itself.

## Build

Standalone (preferred for development):

```bash
cd tools/loomX
mkdir build && cd build
cmake .. -DROSE_INSTALL_PREFIX=/path/to/rose/install
make -j$(nproc)
```

In-tree inside ROSE:

```bash
cd /path/to/rose-build
make tools_loomX_exe -j$(nproc)
```

The binary is named `loomX`. At runtime it needs `librose.so` and Clang/LLVM libraries on `LD_LIBRARY_PATH` / rpath.

## CLI modes

```bash
./loomX --cpu-only       input.c -o out.c   # CPU OpenMP for all safe loops
./loomX --gpu-naive      input.c -o out.c   # offload everything safe (ablation)
./loomX --gpu-profitable input.c -o out.c   # default: profitability gate chooses CPU vs GPU
./loomX --no-scalar-dep-check input.c -o out.c  # disable scalar loop-carried-dependence guard
./loomX --cpu-only --analyze-only input.c    # print per-loop verdicts, no output file
./loomX -v input.c                           # verbose analysis output
```

Tunable thresholds: `--min-gpu-speedup`, `--min-total-flop`, `--min-nested-flop`, `--compute-bound-threshold`.

Always pass `-rose:skipfinalCompileStep` if you do not want ROSE to invoke the backend compiler.

## Quick unit-test verification

```bash
cd tools/loomX
./tests/run_tests.sh
```

This runs `loomX` on every `tests/*.c`, compiles original and transformed with `gcc -O2 -fopenmp -lm`, and compares exit codes.

## Evaluation harness

The benchmark driver lives in `tools/loomX/benchmark/`.

Fetch external suites once:

```bash
cd tools/loomX/benchmark
./setup_benchmarks.sh ./bench-suites
```

Run the hand-written interprocedural corpus:

```bash
./run_benchmarks.sh interproc-microbench
```

Run a small PolyBench subset:

```bash
./run_benchmarks.sh polybench
```

The driver emits four configs per benchmark: `seq`, `cpu_omp`, `gpu_naive`, `gpu_profitable`. It checks correctness vs. `seq`, times each config, and aggregates geomean speedups.

## Important gotchas

- **ROSE library path**: `run_benchmarks.sh` hardcodes `LD_LIBRARY_PATH` to a specific install path. If that path is wrong in your environment, override it or set `LOOMX` to a binary with the right rpath.
- **GPU offload compiler**: GPU configs require a clang built with `openmp` and `libomptarget-nvptarget` support. If `clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda` fails with "no library 'libomptarget-nvptx.bc' found", GPU configs are skipped gracefully. See `benchmark/build_gpu_clang.md`.
- **GPU architecture**: `run_benchmarks.sh` auto-detects from `nvidia-smi`. Override with `GPU_ARCH=sm_80` if the detected arch is unsupported by your CUDA/clang combination (e.g. sm_120 on CUDA 12.4).
- **Clock locking**: Set `LOCK_CLOCKS=yes` when benchmarking for reproducible timing. Requires root for `nvidia-smi -lgc`.
- **DataRaceBench evaluation**: `check_dataracebench.py` judges acceptance by checking whether the generated source contains `#pragma omp`. Use `--strict` to strip input pragmas, enable the scalar-dependence guard, and use `--analyze-only` to match verdicts against the original hot loop. The default lenient mode leaves pragmas in place. It reports false positives (accepted a `-yes` race) and false negatives (rejected a `-no` safe case).
- **Reporting convention**: report full end-to-end wall clock (median of ≥10 runs), geometric mean across benchmarks, and failure/regression counts from `failures.csv`. Do not report kernel-only time or arithmetic mean.
- **Interprocedural micro-benchmarks** are included in `benchmark/interproc-microbench/`; PolyBench/Rodinia/DataRaceBench are fetched by `setup_benchmarks.sh`.

## Architecture pointers

- `src/main.cpp` — entry point, CLI parsing, drives IPA → loop discovery → dependence check → summary → codegen.
- `src/InterproceduralAnalysis.{h,cpp}` — whole-program side-effect summaries and call-graph safety.
- `src/GpuProfitability.{h,cpp}` — cost model and CPU/GPU/sequential target decision.
- `src/ComputeIntensityEstimator.{h,cpp}` — FLOP/memory-op counting, including a function-body estimator used by IPA.
- `src/OpenMPCodeGen.cpp` — pragma insertion and target-data-region post-processing.
- `src/LoopDependenceAnalysis.{h,cpp}` — loop-carried dependence check before parallelization.
