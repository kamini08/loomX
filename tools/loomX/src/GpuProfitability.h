#pragma once
#include "rose.h"
#include "LoopAnalysisTypes.h"
#include "LoopSummary.h"
#include "LoopCanonicalChecker.h"
#include "IterationCountEstimator.h"
#include "DivergenceAnalyzer.h"
#include "ComputeIntensityEstimator.h"

// GPU profitability model. Combines analytical cost estimates with
// thresholds from a tunable ProfitabilityConfig to decide whether a loop
// should run sequentially, on the CPU with OpenMP, or offloaded to the GPU.
class GpuProfitability {
public:
    GpuProfitability();
    explicit GpuProfitability(const loomX::ProfitabilityConfig& config);

    // Analyze a loop and decide where it should run (legacy convenience).
    loomX::ParallelTarget classifyLoop(SgForStatement* loop);

    // Build a complete LoopSummary for a loop, including the target decision.
    loomX::LoopSummary summarize(SgForStatement* loop);

    // Re-run the target decision on a summary whose analysis fields have been
    // modified (e.g. after adding interprocedural callee work estimates).
    void reevaluateTarget(loomX::LoopSummary& summary);

    // Access the underlying analyzers for detailed diagnostics.
    loomX::LoopCanonicalChecker& getCanonicalChecker() { return canonicalChecker_; }
    loomX::IterationCountEstimator& getIterationEstimator() { return iterationEstimator_; }
    loomX::DivergenceAnalyzer& getDivergenceAnalyzer() { return divergenceAnalyzer_; }
    loomX::ComputeIntensityEstimator& getIntensityEstimator() { return intensityEstimator_; }

    // Access/modify the profitability configuration.
    const loomX::ProfitabilityConfig& getConfig() const { return config_; }
    void setConfig(const loomX::ProfitabilityConfig& config) { config_ = config; }

private:
    loomX::LoopCanonicalChecker canonicalChecker_;
    loomX::IterationCountEstimator iterationEstimator_;
    loomX::DivergenceAnalyzer divergenceAnalyzer_;
    loomX::ComputeIntensityEstimator intensityEstimator_;
    loomX::ProfitabilityConfig config_;

    loomX::ParallelTarget decideTarget(const loomX::LoopSummary& summary);

    // Cost-model estimates (abstract time units).
    double estimateCpuTime(const loomX::LoopSummary& summary) const;
    double estimateGpuTime(const loomX::LoopSummary& summary) const;
    double estimateDataMovementBytes(const loomX::LoopSummary& summary) const;

    // Gap-1 helpers: loop-purpose detection and total-work filtering.
    long long totalFlops(const loomX::LoopSummary& summary) const;
    bool isInitializationLoop(const loomX::LoopSummary& summary) const;
    bool isReductionOnlyLoop(const loomX::LoopSummary& summary) const;

    // Legacy helpers retained for backward compatibility.
    long estimateIterationCount(SgForStatement* loop);
    bool hasRegularAccessPattern(SgForStatement* loop);
    bool hasDivergentControlFlow(SgForStatement* loop);
    bool isComputeIntensive(SgForStatement* loop);

    // True if the loop body contains any nested for-loop.
    bool hasNestedLoops(SgForStatement* loop);
};
