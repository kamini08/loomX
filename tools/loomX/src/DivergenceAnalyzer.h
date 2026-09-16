#pragma once
#include "LoopAnalysisTypes.h"

namespace loomX {

// Detects control-flow divergence inside a loop that can hurt GPU
// performance or prevent parallelization.  Divergence is classified
// by source (data-dependent branch, early exit, inner loop, etc.).
class DivergenceAnalyzer {
public:
    DivergenceAnalyzer() = default;

    // Analyze the loop body for divergent control flow.
    DivergenceResult analyze(SgForStatement* loop);

private:
    bool conditionDependsOnData(SgExpression* condition, SgInitializedName* loopVar);
    bool hasEarlyExit(SgForStatement* loop);
    bool hasInnerLoop(SgForStatement* loop);
    bool containsFunctionCallInBody(SgForStatement* loop);
};

} // namespace loomX
