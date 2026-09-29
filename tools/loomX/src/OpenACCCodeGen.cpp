#include "OpenACCCodeGen.h"
#include "LoopAnalysisUtil.h"
#include <map>
#include <set>
#include <sstream>

using namespace loomX;

void OpenACCCodeGen::generatePragmas(const loomX::LoopSummary& summary) {
    switch (summary.target) {
        case ParallelTarget::CPU_OPENMP:
            insertCPUPragma(summary);
            break;
        case ParallelTarget::GPU_OFFLOAD:
            insertRoutineSeqPragmas(summary);
            insertGPUPragma(summary);
            break;
        case ParallelTarget::SEQUENTIAL:
            // No pragma inserted.
            break;
    }
}

void OpenACCCodeGen::insertCPUPragma(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    std::ostringstream pragmaText;
    pragmaText << "acc parallel loop";

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

void OpenACCCodeGen::insertGPUPragma(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    std::ostringstream pragmaText;
    pragmaText << "acc parallel loop";

    // Flatten a perfectly nested 2D/3D fan for better GPU occupancy.  The
    // collapse depth is precomputed by GpuProfitability::collapseDepthFor and
    // only ever >= 2 when every collapsed inner loop is canonical and carries
    // no loop-carried dependence.
    if (summary.collapseDepth > 1) {
        pragmaText << " collapse(" << summary.collapseDepth << ")";
    }

    if (!summary.privateVars.empty()) {
        pragmaText << " private(" << buildVarList(summary.privateVars) << ")";
    }

    if (!summary.reductions.empty()) {
        pragmaText << " " << buildReductionClause(summary.reductions);
    }

    if (!summary.mapClauses.empty()) {
        pragmaText << " " << buildCopyClause(summary.mapClauses);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenACCCodeGen::insertRoutineSeqPragmas(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    // Find every function called inside the loop and mark its definition
    // with #pragma acc routine seq so it can be used on the GPU.
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(loop, V_SgFunctionCallExp);

    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;

        SgFunctionDeclaration* callee = call->getAssociatedFunctionDeclaration();
        if (!callee) continue;

        SgDeclarationStatement* definingDecl = callee->get_definingDeclaration();
        SgFunctionDeclaration* definingFunc = isSgFunctionDeclaration(definingDecl);
        if (!definingFunc) continue;

        SgFunctionDefinition* def = definingFunc->get_definition();
        if (!def) continue;

        SgFunctionDeclaration* decl = def->get_declaration();
        if (!decl) continue;

        // Avoid inserting duplicate routine seq pragmas.
        bool alreadyDeclared = false;
        SgStatement* prev = SageInterface::getPreviousStatement(decl);
        if (prev) {
            SgPragmaDeclaration* prevPragma = isSgPragmaDeclaration(prev);
            if (prevPragma) {
                std::string text = prevPragma->unparseToString();
                if (text.find("acc routine") != std::string::npos) {
                    alreadyDeclared = true;
                }
            }
        }

        if (alreadyDeclared) continue;

        SgPragmaDeclaration* routineSeq =
            SageBuilder::buildPragmaDeclaration("acc routine seq",
                                                decl->get_scope());
        SageInterface::insertStatementBefore(decl, routineSeq);
    }
}

// Build a mapped variable reference for an OpenACC copy clause.  The rules are
// the same as for OpenMP: parameter arrays need per-dimension array sections,
// file-scope / static arrays use the bare name.
static std::string buildMappedVarName(SgInitializedName* var) {
    if (!var) return "";

    std::string name = var->get_name().getString();
    SgType* type = var->get_type();
    if (!type) return name;

    type = type->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
    SgArrayType* arrType = isSgArrayType(type);
    if (!arrType) return name;

    bool isParameter = isSgFunctionParameterList(var->get_parent()) != nullptr;
    if (!isParameter) return name;

    std::string section;
    while (arrType) {
        SgExpression* index = arrType->get_index();
        if (!index) return name;
        std::string dim = index->unparseToString();
        if (dim.empty()) return name;
        section += "[0:" + dim + "]";
        SgType* baseType = arrType->get_base_type();
        if (!baseType) break;
        baseType = baseType->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
        arrType = isSgArrayType(baseType);
    }

    if (section.empty()) return name;
    return name + section;
}

std::string OpenACCCodeGen::buildCopyClause(
    const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    // Deduplicate by variable name, keeping the most conservative direction
    // (copy > copyout > copyin) in case the same name was collected from
    // multiple references.
    std::map<std::string, std::string> directionByName;
    auto directionRank = [](const std::string& d) {
        if (d == "tofrom" || d == "copy") return 2;
        if (d == "from" || d == "copyout") return 1;
        return 0;
    };
    auto toOpenACCDirection = [](const std::string& d) {
        if (d == "tofrom" || d == "copy") return "copy";
        if (d == "from" || d == "copyout") return "copyout";
        return "copyin";  // "to" or "copyin"
    };
    for (const auto& [var, direction] : mapClauses) {
        std::string name = buildMappedVarName(var);
        std::string accDir = toOpenACCDirection(direction);
        auto it = directionByName.find(name);
        if (it == directionByName.end() || directionRank(accDir) > directionRank(it->second)) {
            directionByName[name] = accDir;
        }
    }

    // Group variables by direction.
    std::map<std::string, std::vector<std::string>> directionGroups;
    for (const auto& [name, direction] : directionByName) {
        directionGroups[direction].push_back(name);
    }

    std::ostringstream oss;
    bool firstClause = true;
    for (const auto& [direction, vars] : directionGroups) {
        if (!firstClause) oss << " ";
        firstClause = false;
        oss << direction << "(";
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

std::string OpenACCCodeGen::buildVarList(const std::set<SgInitializedName*>& vars) {
    std::ostringstream oss;
    bool first = true;
    for (SgInitializedName* var : vars) {
        if (!first) oss << ", ";
        first = false;
        oss << var->get_name().getString();
    }
    return oss.str();
}

std::string OpenACCCodeGen::buildReductionClause(
    const std::vector<loomX::ReductionInfo>& reductionDetails) {
    // Group variables by operator string.
    std::map<std::string, std::vector<std::string>> groups;
    for (const loomX::ReductionInfo& info : reductionDetails) {
        if (!info.variable) continue;
        std::string op = info.opString.empty() ? "+" : info.opString;
        std::string varName = info.variable->get_name().getString();
        if (info.isArrayElement && info.arrayIndex) {
            varName += "[" + info.arrayIndex->unparseToString() + "]";
        }
        groups[op].push_back(varName);
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

void OpenACCCodeGen::postProcessSource(std::string& source) {
    if (source.find("#include <openacc.h>") == std::string::npos &&
        source.find("#include \"openacc.h\"") == std::string::npos) {
        source = std::string("#include <openacc.h>\n\n") + source;
    }
}
