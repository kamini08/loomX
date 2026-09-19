#pragma once
#include "rose.h"
#include "LoopAnalysisTypes.h"
#include "PragmaAnalysis.h"
#include <set>
#include <vector>

namespace loomX {

// Unified summary of everything known about a loop, plus the final
// parallelization decision from the GPU cost model.
struct LoopSummary {
    SgForStatement* loop = nullptr;

    // ---------- Analyzer results ----------
    CanonicalResult canonical;
    long iterationCount = -1;
    bool regularAccess = false;
    DivergenceResult divergence;
    ComputeIntensityResult intensity;
    std::vector<ReductionInfo> reductions;
    PragmaAnalysisResult pragmas;

    // Interprocedural safety for any function calls inside the loop.
    bool hasFunctionCalls = false;
    bool allFunctionCallsSafe = true;

    // ---------- Decision ----------
    ParallelTarget target = ParallelTarget::SEQUENTIAL;

    // ---------- Codegen inputs ----------
    // Variables that must be private to each thread.
    std::set<SgInitializedName*> privateVars;

    // Map clauses for GPU target regions, grouped by direction.
    std::set<std::pair<SgInitializedName*, std::string>> mapClauses;

    // Perfect-nest depth for "collapse(N)" on the GPU pragma.  1 means the loop
    // is collapsed by itself (no collapse clause).  Computed for GPU-offloaded
    // loops by GpuProfitability::collapseDepthFor.
    int collapseDepth = 1;

    // Helper: collect just the reduction variables.
    std::set<SgInitializedName*> getReductionVariables() const {
        std::set<SgInitializedName*> vars;
        for (const ReductionInfo& r : reductions) {
            if (r.variable) vars.insert(r.variable);
        }
        return vars;
    }
};

} // namespace loomX
