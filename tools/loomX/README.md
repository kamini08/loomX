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
  to label loops as memory-bound, balanced, or compute-bound.
* `ReductionDetector` — finds reduction variables and their operators
  (`+`, `-`, `*`, `min`, `max`, `&`, `|`, `^`).

`GpuProfitability` orchestrates these analyzers to choose between sequential,
CPU OpenMP, and GPU offload targets.

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

## Quick validation

After building, compile the original and transformed sources with an OpenMP
compiler and compare their outputs:

```bash
./loomX -rose:skipfinalCompileStep tests/benchmark1.c
gcc -O2 -fopenmp -o bench_seq tests/benchmark1.c
gcc -O2 -fopenmp -o bench_gpu rose_benchmark1.c
./bench_seq; echo "seq exit: $?"
./bench_gpu; echo "gpu exit: $?"
```
