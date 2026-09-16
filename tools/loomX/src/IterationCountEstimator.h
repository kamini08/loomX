#pragma once
#include "LoopAnalysisTypes.h"
#include "LoopCanonicalChecker.h"

namespace loomX {

// Estimates the number of iterations of a canonical for-loop.
// Returns a symbolic/static estimate; -1 means "unknown".
class IterationCountEstimator {
public:
    IterationCountEstimator(LoopCanonicalChecker& canonicalChecker);

    // Estimate iteration count.  If exactLiteral is true, only return a value
    // when the bound is a compile-time constant; otherwise fall back to a
    // heuristic default for symbolic bounds.
    long estimate(SgForStatement* loop, bool exactLiteral = false);

    // Try to evaluate an expression to a signed integer constant.
    static long evaluateExpression(SgExpression* expr);

private:
    LoopCanonicalChecker& canonicalChecker_;

    long applyStride(long iterations, SgExpression* stride);
    long defaultForSymbolicBound();
};

} // namespace loomX
