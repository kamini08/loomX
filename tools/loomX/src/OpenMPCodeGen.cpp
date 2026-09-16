#include "OpenMPCodeGen.h"
#include "LoopAnalysisUtil.h"
#include <sstream>

using namespace loomX;

void OpenMPCodeGen::generatePragmas(const loomX::LoopSummary& summary) {
    switch (summary.target) {
        case ParallelTarget::CPU_OPENMP:
            insertCPUPragma(summary);
            break;
        case ParallelTarget::GPU_OFFLOAD:
            insertDeclareTargetPragmas(summary);
            insertGPUPragma(summary);
            break;
        case ParallelTarget::SEQUENTIAL:
            // No pragma inserted.
            break;
    }
}

void OpenMPCodeGen::insertCPUPragma(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    std::ostringstream pragmaText;
    pragmaText << "omp parallel for";

    if (!summary.privateVars.empty()) {
        pragmaText << " private(" << buildVarList(summary.privateVars) << ")";
    }

    if (!summary.reductions.empty()) {
        pragmaText << " " << buildReductionClause(summary.reductions);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenMPCodeGen::insertGPUPragma(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    std::ostringstream pragmaText;
    pragmaText << "omp target teams distribute parallel for";

    if (!summary.privateVars.empty()) {
        pragmaText << " private(" << buildVarList(summary.privateVars) << ")";
    }

    if (!summary.reductions.empty()) {
        pragmaText << " " << buildReductionClause(summary.reductions);
    }

    if (!summary.mapClauses.empty()) {
        pragmaText << " " << buildMapClause(summary.mapClauses);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenMPCodeGen::insertDeclareTargetPragmas(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    // Find every function called inside the loop and mark its definition
    // with #pragma omp declare target so it can be used on the GPU.
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(loop, V_SgFunctionCallExp);

    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;

        SgFunctionDeclaration* callee = call->getAssociatedFunctionDeclaration();
        if (!callee) continue;

        // The declaration associated with a call may be non-defining.
        // Look up the defining declaration to find the function body.
        SgDeclarationStatement* definingDecl = callee->get_definingDeclaration();
        SgFunctionDeclaration* definingFunc = isSgFunctionDeclaration(definingDecl);
        if (!definingFunc) continue;

        SgFunctionDefinition* def = definingFunc->get_definition();
        if (!def) continue;

        SgFunctionDeclaration* decl = def->get_declaration();
        if (!decl) continue;

        // Avoid inserting duplicate declare-target pragmas.
        bool alreadyDeclared = false;
        SgStatement* prev = SageInterface::getPreviousStatement(decl);
        if (prev) {
            SgPragmaDeclaration* prevPragma = isSgPragmaDeclaration(prev);
            if (prevPragma) {
                std::string text = prevPragma->unparseToString();
                if (text.find("omp declare target") != std::string::npos) {
                    alreadyDeclared = true;
                }
            }
        }

        if (alreadyDeclared) continue;

        SgPragmaDeclaration* declareTarget =
            SageBuilder::buildPragmaDeclaration("omp declare target",
                                                decl->get_scope());
        SageInterface::insertStatementBefore(decl, declareTarget);

        SgPragmaDeclaration* endDeclareTarget =
            SageBuilder::buildPragmaDeclaration("omp end declare target",
                                                decl->get_scope());
        SageInterface::insertStatementAfter(decl, endDeclareTarget);
    }
}

std::string OpenMPCodeGen::buildMapClause(
    const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    // Group variables by direction.
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
