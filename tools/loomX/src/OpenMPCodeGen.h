#pragma once
#include "rose.h"
#include "GpuProfitability.h"
#include <set>
#include <string>

// OpenMP pragma generation
class OpenMPCodeGen {
public:
    // Insert OpenMP directives for a loop based on target classification
    void generatePragmas(SgForStatement* loop,
                         ParallelTarget target,
                         const std::set<SgInitializedName*>& privateVars,
                         const std::set<SgInitializedName*>& reductionVars,
                         const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);

private:
    // Insert CPU OpenMP parallel for pragma
    void insertCPUPragma(SgForStatement* loop,
                         const std::set<SgInitializedName*>& privateVars,
                         const std::set<SgInitializedName*>& reductionVars);

    // Insert GPU OpenMP target teams distribute parallel for pragma
    void insertGPUPragma(SgForStatement* loop,
                           const std::set<SgInitializedName*>& privateVars,
                           const std::set<SgInitializedName*>& reductionVars,
                           const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);

    // Build map clause string from variable list
    std::string buildMapClause(const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses);

    // Build variable list string (private, reduction, etc.)
    std::string buildVarList(const std::set<SgInitializedName*>& vars);
    std::string buildReductionClause(const std::set<SgInitializedName*>& reductionVars);
};
