# loomX

loomX is an automatic GPGPU parallelizer for sequential C loops. It is built
as a tool inside the ROSE compiler infrastructure.

See the top-level `README.md` in the standalone repo for full documentation.

## Loop-analysis components

The profitability and safety pipeline is split into dedicated analyzers in
`src/`:

* `LoopCanonicalChecker` — decides whether a `for` loop is in a canonical
  OpenMP-parallelizable form and extracts bounds, stride, and iterator.
* `IterationCountEstimator` — statically estimates trip counts from constant
  or symbolic bounds.
* `DivergenceAnalyzer` — classifies data-dependent branches, early exits,
  nested loops, and function-call divergence.
* `ComputeIntensityEstimator` — counts FLOPs, integer ops, and memory accesses
  to label loops as memory-bound, balanced, or compute-bound.  Exposes a
  function-body work estimator for interprocedural analysis.
* `ReductionDetector` — finds reduction variables and their operators
  (`+`, `-`, `*`, `min`, `max`, `&`, `|`, `^`).
* `InterproceduralAnalysis` — builds per-function side-effect summaries and a
  call graph, then decides whether function calls are safe inside parallel
  loops.  Supports loop-disjoint pointer writes such as `arr[idx]`,
  `arr[idx + c]`, `arr[idx - c]`, and `*(p + idx)`.
* `LoopDependenceAnalysis` — checks for loop-carried dependences before
  parallelization.

`GpuProfitability` orchestrates these analyzers to choose between sequential,
CPU OpenMP, and GPU offload targets.  The cost model now includes PCIe
latency, data-transfer bandwidth, and callee work estimates so that small or
memory-bound loops stay on the CPU while large compute kernels are offloaded.

## Building inside ROSE (CMake)

When ROSE is configured with CMake, loomX is built as part of the `tools`
directory:

```bash
cd /path/to/rose-build
make tools_loomX_exe -j$(nproc)
```

Or build the full ROSE tree:

```bash
cd /path/to/rose-build
make -j$(nproc)
```

## Standalone build

If ROSE is installed but no `RoseConfig.cmake` is present, point CMake at the
installation prefix:

```bash
cd tools/loomX
mkdir build && cd build
cmake .. -DROSE_INSTALL_PREFIX=/path/to/rose/install
make -j$(nproc)
```

## Usage

```bash
./loomX -v -rose:skipfinalCompileStep input.c
```

The transformed source is written to `rose_input.c`.

Tunable profitability thresholds:

* `--min-gpu-speedup <f>` — minimum CPU/GPU speedup to justify offload
  (default 1.2).
* `--min-total-flop <n>` — minimum total FLOPs (body FLOPs × iterations) for
  GPU offload (default 10,000,000).
* `--min-nested-flop <n>` — minimum body FLOPs for the nested-loop GPU
  heuristic (default 1,000,000).
* `--compute-bound-threshold <f>` — FLOPs per memory-op threshold used to label
  a loop compute-bound (default 16.0).

Translation modes for ablation experiments:

* `--cpu-only` — force CPU OpenMP for all safe loops.
* `--gpu-naive` — offload every safe loop to the GPU.
* `--gpu-profitable` — use the profitability model (default).

Other flags:

* `-v` / `--verbose` — print analysis summaries.
* `--intraprocedural-baseline` — reject all pointer-parameter writes, mimicking
  a purely intraprocedural safety analysis.

## Quick validation

A test harness runs `loomX` on every `tests/*.c` file, compiles both the
original and transformed sources with `gcc -O2 -fopenmp`, and compares exit
codes:

```bash
./tools/loomX/tests/run_tests.sh
```

For manual validation of a single file:

```bash
./loomX -rose:skipfinalCompileStep tests/benchmark1.c
gcc -O2 -fopenmp -o bench_seq tests/benchmark1.c
gcc -O2 -fopenmp -o bench_gpu rose_benchmark1.c
./bench_seq; echo "seq exit: $?"
./bench_gpu; echo "gpu exit: $?"
```
