#pragma once
#include "CodeGen.h"
#include "rose.h"
#include "LoopSummary.h"
#include <set>
#include <string>
#include <vector>

// OpenMP pragma generation driven by a LoopSummary.
class OpenMPCodeGen : public CodeGen {
public:
    // Insert all OpenMP directives required by the summary decision.
    // For GPU_OFFLOAD this includes:
    //   - #pragma omp declare target before any callee definitions
    //   - #pragma omp target teams distribute parallel for before the loop
    // For CPU_OPENMP this includes:
    //   - #pragma omp parallel for before the loop
    // For SEQUENTIAL this does nothing.
    void generatePragmas(const loomX::LoopSummary& summary) override;

    // Insert only the CPU `#pragma omp parallel for` (plus private/reduction
    // clauses).  Used by the CUDA/OpenCL backends as a fallback for loops that
    // cannot be expressed as device kernels.
    void generateCPUPragma(const loomX::LoopSummary& summary);

    // Text-based post-processing of the unparsed source: hoist consecutive
    // GPU-offload loops into a single #pragma omp target data region and
    // prepend #include <omp.h> if it is missing.
    void postProcessSource(std::string& source) override;

    // Helper used by postProcessSource.
    static void hoistTargetDataRegions(std::string& source);

private:
    void insertCPUPragma(const loomX::LoopSummary& summary);
    void insertGPUPragma(const loomX::LoopSummary& summary);
    void insertDeclareTargetPragmas(const loomX::LoopSummary& summary);

    std::string buildMapClause(
        const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);
    std::string buildVarList(const std::set<SgInitializedName*>& vars);
    std::string buildReductionClause(
        const std::vector<loomX::ReductionInfo>& reductionDetails);

    // Target-data hoisting helpers (text-based). collectMappedVars returns a
    // map from variable name to its OpenMP map direction (to/from/tofrom) as
    // extracted from the pragma text.
    static std::map<std::string, std::string> collectMappedVars(
        const std::string& regionText);
    static std::string stripTargetAndMap(const std::string& pragmaText);
    static std::string buildTargetDataMapClause(
        const std::map<std::string, std::string>& mappedVars);
};
