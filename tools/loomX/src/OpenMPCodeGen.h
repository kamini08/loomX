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

    // Text-based post-processing: rewrite the unparsed source so that
    // consecutive GPU-offload loops are wrapped in a single
    // #pragma omp target data region. This avoids fragile AST rewrites for
    // the structured block and is applied after project->unparse().
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
