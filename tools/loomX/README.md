# loomX

loomX is an automatic GPGPU parallelizer for sequential C loops. It is built
as a tool inside the ROSE compiler infrastructure.

See the top-level `README.md` in the standalone repo for full documentation.

## Building inside ROSE

When ROSE is configured and built, loomX is built as part of the `tools`
directory:

```bash
cd /path/to/rose-build
make loomX -j$(nproc)
```

Or build the full ROSE tree:

```bash
cd /path/to/rose-build
make -j$(nproc)
```

## Usage

```bash
./tools/loomX -v -rose:skipfinalCompileStep input.c
```

The transformed source is written to `rose_input.c`.
