#pragma once
#include "LoopAnalysisTypes.h"

namespace loomX {

// Checks whether a C for-loop is in a canonical form suitable for
// OpenMP parallelization.  The checker records the loop index variable,
// bounds, stride, and a human-readable reason if the loop is rejected.
class LoopCanonicalChecker {
public:
    LoopCanonicalChecker() = default;

    // Analyze a single for-loop.  The result is cached per loop.
    const CanonicalResult& analyze(SgForStatement* loop);

    // Convenience predicates.
    bool isCanonical(SgForStatement* loop);

    // Reset cached results.
    void clear();

private:
    std::map<SgForStatement*, CanonicalResult> cache_;

    bool hasSingleIterator(SgForStatement* loop, SgInitializedName*& iter);
    bool incrementIsSimple(SgExpression* increment, SgInitializedName* iter, SgExpression*& stride);
    bool boundsUseSimpleExpressions(SgForStatement* loop);
    bool iteratorModifiedOnlyByIncrement(SgForStatement* loop, SgInitializedName* iter);
    std::string reasonToString(NonCanonicalReason reason);
};

} // namespace loomX
