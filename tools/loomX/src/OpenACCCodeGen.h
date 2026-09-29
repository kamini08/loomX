#pragma once
#include "CodeGen.h"
#include "rose.h"
#include "LoopSummary.h"
#include <set>
#include <string>
#include <vector>

// OpenACC pragma generation driven by a LoopSummary.
class OpenACCCodeGen : public CodeGen {
public:
    // Insert all OpenACC directives required by the summary decision.
    // For GPU_OFFLOAD this includes:
    //   - #pragma acc routine seq before any callee definitions
    //   - #pragma acc parallel loop [collapse(N)] before the loop
    // For CPU_OPENMP this includes:
    //   - #pragma acc parallel loop before the loop (multicore CPU)
    // For SEQUENTIAL this does nothing.
    void generatePragmas(const loomX::LoopSummary& summary) override;

    // Prepend #include <openacc.h> if it is missing.
    void postProcessSource(std::string& source) override;

private:
    void insertCPUPragma(const loomX::LoopSummary& summary);
    void insertGPUPragma(const loomX::LoopSummary& summary);
    void insertRoutineSeqPragmas(const loomX::LoopSummary& summary);

    std::string buildCopyClause(
        const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);
    std::string buildVarList(const std::set<SgInitializedName*>& vars);
    std::string buildReductionClause(
        const std::vector<loomX::ReductionInfo>& reductionDetails);
};
