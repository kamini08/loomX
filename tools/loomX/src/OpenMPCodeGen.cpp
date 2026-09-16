#include "OpenMPCodeGen.h"
#include <sstream>

void OpenMPCodeGen::generatePragmas(SgForStatement* loop,
                                    ParallelTarget target,
                                    const std::set<SgInitializedName*>& privateVars,
                                    const std::set<SgInitializedName*>& reductionVars,
                                    const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses,
                                    const std::vector<loomX::ReductionInfo>& reductionDetails) {
    switch (target) {
        case ParallelTarget::CPU_OPENMP:
            insertCPUPragma(loop, privateVars, reductionDetails);
            break;
        case ParallelTarget::GPU_OFFLOAD:
            insertGPUPragma(loop, privateVars, reductionDetails, mapClauses);
            break;
        case ParallelTarget::SEQUENTIAL:
            // No pragma inserted
            break;
    }
}

void OpenMPCodeGen::insertCPUPragma(SgForStatement* loop,
                                    const std::set<SgInitializedName*>& privateVars,
                                    const std::vector<loomX::ReductionInfo>& reductionDetails) {
    std::ostringstream pragmaText;
    pragmaText << "omp parallel for";

    if (!privateVars.empty()) {
        pragmaText << " private(" << buildVarList(privateVars) << ")";
    }

    if (!reductionDetails.empty()) {
        pragmaText << " " << buildReductionClause(reductionDetails);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenMPCodeGen::insertGPUPragma(SgForStatement* loop,
                                    const std::set<SgInitializedName*>& privateVars,
                                    const std::vector<loomX::ReductionInfo>& reductionDetails,
                                    const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    std::ostringstream pragmaText;
    pragmaText << "omp target teams distribute parallel for";

    if (!privateVars.empty()) {
        pragmaText << " private(" << buildVarList(privateVars) << ")";
    }

    if (!reductionDetails.empty()) {
        pragmaText << " " << buildReductionClause(reductionDetails);
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
    const std::vector<loomX::ReductionInfo>& reductionDetails) {
    // Group variables by operator string.
    std::map<std::string, std::vector<std::string>> groups;
    for (const loomX::ReductionInfo& info : reductionDetails) {
        if (!info.variable) continue;
        std::string op = info.opString.empty() ? "+" : info.opString;
        groups[op].push_back(info.variable->get_name().getString());
    }

    std::ostringstream oss;
    bool firstClause = true;
    for (const auto& [op, vars] : groups) {
        if (!firstClause) oss << " ";
        firstClause = false;
        oss << "reduction(" << op << ":";
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
