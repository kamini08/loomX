#pragma once
#include "rose.h"
#include "LoopAnalysisTypes.h"
#include "LoopSummary.h"
#include "LoopCanonicalChecker.h"
#include "IterationCountEstimator.h"
#include "DivergenceAnalyzer.h"
#include "ComputeIntensityEstimator.h"

// GPU profitability heuristic.  Delegates to the dedicated analyzers for
// canonical-form checking, iteration-count estimation, divergence analysis,
// and compute-intensity estimation.
class GpuProfitability {
public:
    GpuProfitability();

    // Analyze a loop and decide where it should run (legacy convenience).
    loomX::ParallelTarget classifyLoop(SgForStatement* loop);

    // Build a complete LoopSummary for a loop, including the target decision.
    loomX::LoopSummary summarize(SgForStatement* loop);

    // Access the underlying analyzers for detailed diagnostics.
    loomX::LoopCanonicalChecker& getCanonicalChecker() { return canonicalChecker_; }
    loomX::IterationCountEstimator& getIterationEstimator() { return iterationEstimator_; }
    loomX::DivergenceAnalyzer& getDivergenceAnalyzer() { return divergenceAnalyzer_; }
    loomX::ComputeIntensityEstimator& getIntensityEstimator() { return intensityEstimator_; }

private:
    loomX::LoopCanonicalChecker canonicalChecker_;
    loomX::IterationCountEstimator iterationEstimator_;
    loomX::DivergenceAnalyzer divergenceAnalyzer_;
    loomX::ComputeIntensityEstimator intensityEstimator_;

    loomX::ParallelTarget decideTarget(const loomX::LoopSummary& summary);

    // Legacy helpers retained for backward compatibility.
    long estimateIterationCount(SgForStatement* loop);
    bool hasRegularAccessPattern(SgForStatement* loop);
    bool hasDivergentControlFlow(SgForStatement* loop);
    bool isComputeIntensive(SgForStatement* loop);
};
