#pragma once
#include "rose.h"

// Target for parallelized loop
enum class ParallelTarget { SEQUENTIAL, CPU_OPENMP, GPU_OFFLOAD };

// GPU profitability heuristic
class GpuProfitability {
public:
    // Analyze a loop and decide where it should run
    ParallelTarget classifyLoop(SgForStatement* loop);

private:
    // Evaluate a constant integer expression (-1 if not constant)
    long evaluateExpression(SgExpression* expr);

    // Estimate iteration count (conservative lower bound)
    long estimateIterationCount(SgForStatement* loop);

    // Check if memory access pattern is regular (affine array subscripts)
    bool hasRegularAccessPattern(SgForStatement* loop);

    // Check for divergent control flow within loop body
    bool hasDivergentControlFlow(SgForStatement* loop);

    // Check if loop contains only simple arithmetic (FLOP-friendly)
    bool isComputeIntensive(SgForStatement* loop);
};
