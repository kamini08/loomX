#pragma once
#include "rose.h"
#include "LoopSummary.h"
#include <set>
#include <string>
#include <vector>

// OpenMP pragma generation driven by a LoopSummary.
class OpenMPCodeGen {
public:
    // Insert all OpenMP directives required by the summary decision.
    // For GPU_OFFLOAD this includes:
    //   - #pragma omp declare target before any callee definitions
    //   - #pragma omp target teams distribute parallel for before the loop
    // For CPU_OPENMP this includes:
    //   - #pragma omp parallel for before the loop
    // For SEQUENTIAL this does nothing.
    void generatePragmas(const loomX::LoopSummary& summary);

private:
    void insertCPUPragma(const loomX::LoopSummary& summary);
    void insertGPUPragma(const loomX::LoopSummary& summary);
    void insertDeclareTargetPragmas(const loomX::LoopSummary& summary);

    std::string buildMapClause(
        const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);
    std::string buildVarList(const std::set<SgInitializedName*>& vars);
    std::string buildReductionClause(
        const std::vector<loomX::ReductionInfo>& reductionDetails);
};
