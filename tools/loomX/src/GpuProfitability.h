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

    // Phase-couple initialization loops with a later GPU loop: if an init loop
    // writes arrays that a GPU_OFFLOAD loop later in program order consumes,
    // force the init loop onto the GPU too so the data is produced on the
    // device instead of ping-ponging across PCIe.  Loops are left untouched
    // unless every non-GPU loop between them avoids the written arrays (a CPU
    // reader in between would observe stale host data).  [in,out] summaries
    // must be in program order.
    void phaseCoupleInitLoops(std::vector<loomX::LoopSummary>& summaries);

    // Compute the perfect-nest depth for a "collapse(N)" clause on a
    // GPU-offloaded loop.  Returns >= 2 when the loop heads a perfect nest of
    // inner loops that are all canonical AND carry no loop-carried dependence
    // (flattening such a 2D/3D fan improves GPU occupancy without introducing
    // cross-iteration races on shared accumulators, so e.g. the dot-product
    // k-loop of gemm is correctly NOT flattened).  Returns 1 when nothing can
    // be collapsed.
    int collapseDepthFor(SgForStatement* loop);

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
