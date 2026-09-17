#!/usr/bin/env bash
# Lock GPU clocks to their max for reproducible benchmark timing (NVIDIA only).
# GPU clock boosting/throttling is a major source of run-to-run variance --
# lock before benchmarking, reset when done so you're not left in a
# high-power state.
#
# Usage: ./lock_gpu_clocks.sh lock | reset
set -euo pipefail
ACTION="${1:-lock}"

case "$ACTION" in
  lock)
    sudo nvidia-smi -pm 1
    echo "Supported clocks:"
    nvidia-smi --query-supported-clocks=graphics,memory --format=csv
    MAXGC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
    MAXMC=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits)
    sudo nvidia-smi -lgc "${MAXGC},${MAXGC}"
    sudo nvidia-smi -lmc "${MAXMC},${MAXMC}"
    echo "Locked SM clock=${MAXGC}MHz, memory clock=${MAXMC}MHz"
    ;;
  reset)
    sudo nvidia-smi -rgc
    sudo nvidia-smi -rmc
    echo "Clocks reset to auto-boost"
    ;;
  *)
    echo "usage: $0 [lock|reset]"; exit 1
    ;;
esac
