#pragma once
#include "rose.h"
#include "GpuProfitability.h"
#include "LoopAnalysisTypes.h"
#include <set>
#include <string>
#include <vector>

// OpenMP pragma generation
class OpenMPCodeGen {
public:
    // Insert OpenMP directives for a loop based on target classification.
    // reductionDetails carries the detected operator for each variable.
    void generatePragmas(SgForStatement* loop,
                         ParallelTarget target,
                         const std::set<SgInitializedName*>& privateVars,
                         const std::set<SgInitializedName*>& reductionVars,
                         const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses,
                         const std::vector<loomX::ReductionInfo>& reductionDetails = {});

private:
    // Insert CPU OpenMP parallel for pragma
    void insertCPUPragma(SgForStatement* loop,
                         const std::set<SgInitializedName*>& privateVars,
                         const std::vector<loomX::ReductionInfo>& reductionDetails);

    // Insert GPU OpenMP target teams distribute parallel for pragma
    void insertGPUPragma(SgForStatement* loop,
                         const std::set<SgInitializedName*>& privateVars,
                         const std::vector<loomX::ReductionInfo>& reductionDetails,
                         const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);

    // Build map clause string from variable list
    std::string buildMapClause(const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);

    // Build variable list string (private, reduction, etc.)
    std::string buildVarList(const std::set<SgInitializedName*>& vars);
    std::string buildReductionClause(const std::vector<loomX::ReductionInfo>& reductionDetails);
};
