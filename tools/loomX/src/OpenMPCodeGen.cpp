#include "OpenMPCodeGen.h"
#include <sstream>

void OpenMPCodeGen::generatePragmas(SgForStatement* loop,
                                      ParallelTarget target,
                                      const std::set<SgInitializedName*>& privateVars,
                                      const std::set<SgInitializedName*>& reductionVars,
                                      const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    switch (target) {
        case ParallelTarget::CPU_OPENMP:
            insertCPUPragma(loop, privateVars, reductionVars);
            break;
        case ParallelTarget::GPU_OFFLOAD:
            insertGPUPragma(loop, privateVars, reductionVars, mapClauses);
            break;
        case ParallelTarget::SEQUENTIAL:
            // No pragma inserted
            break;
    }
}

void OpenMPCodeGen::insertCPUPragma(SgForStatement* loop,
                                     const std::set<SgInitializedName*>& privateVars,
                                     const std::set<SgInitializedName*>& reductionVars) {
    std::ostringstream pragmaText;
    pragmaText << "omp parallel for";

    if (!privateVars.empty()) {
        pragmaText << " private(" << buildVarList(privateVars) << ")";
    }

    if (!reductionVars.empty()) {
        pragmaText << " " << buildReductionClause(reductionVars);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenMPCodeGen::insertGPUPragma(SgForStatement* loop,
                                      const std::set<SgInitializedName*>& privateVars,
                                      const std::set<SgInitializedName*>& reductionVars,
                                      const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    std::ostringstream pragmaText;
    pragmaText << "omp target teams distribute parallel for";

    if (!privateVars.empty()) {
        pragmaText << " private(" << buildVarList(privateVars) << ")";
    }

    if (!reductionVars.empty()) {
        pragmaText << " " << buildReductionClause(reductionVars);
    }

    if (!mapClauses.empty()) {
        pragmaText << " " << buildMapClause(mapClauses);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

std::string OpenMPCodeGen::buildMapClause(
    const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    // Group variables by direction
    std::map<std::string, std::vector<std::string>> directionGroups;
    for (const auto& [var, direction] : mapClauses) {
        directionGroups[direction].push_back(var->get_name().getString());
    }

    std::ostringstream oss;
    bool firstClause = true;
    for (const auto& [direction, vars] : directionGroups) {
        if (!firstClause) oss << " ";
        firstClause = false;
        oss << "map(" << direction << ":";
        bool firstVar = true;
        for (const std::string& varName : vars) {
            if (!firstVar) oss << ", ";
            firstVar = false;
            oss << varName;
        }
        oss << ")";
    }
    return oss.str();
}

std::string OpenMPCodeGen::buildVarList(const std::set<SgInitializedName*>& vars) {
    std::ostringstream oss;
    bool first = true;
    for (SgInitializedName* var : vars) {
        if (!first) oss << ", ";
        first = false;
        oss << var->get_name().getString();
    }
    return oss.str();
}

std::string OpenMPCodeGen::buildReductionClause(
    const std::set<SgInitializedName*>& reductionVars) {
    // For simplicity, assume all reductions are +
    // Real implementation would detect the operator
    std::ostringstream oss;
    oss << "reduction(+:" << buildVarList(reductionVars) << ")";
    return oss.str();
}
