#include "OpenACCCodeGen.h"
#include "ParallelForPlan.h"
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

    ParallelForPlan plan(summary);

    if (!summary.privateVars.empty()) {
        pragmaText << " private(" << plan.privateList() << ")";
    }

    std::string reductionText = plan.reductionClause();
    if (!reductionText.empty()) {
        pragmaText << " " << reductionText;
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenACCCodeGen::insertGPUPragma(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    ParallelForPlan plan(summary);

    // Same guard as the OpenMP backend: a bare pointer name in a copy clause
    // moves the pointer, not the allocation, so the device faults. Refuse to
    // offload rather than emit an unsound copy.
    if (!plan.storageIsFullyMapped()) {
        std::ostringstream reason;
        reason << "[OpenACCCodeGen] not offloading loop: no derivable extent for "
               << plan.unmappable().size() << " mapped variable(s): ";
        for (size_t i = 0; i < plan.unmappable().size(); ++i) {
            if (i) reason << ", ";
            reason << plan.unmappable()[i];
        }
        std::cerr << reason.str() << " -- leaving loop sequential" << std::endl;
        return;
    }

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
        pragmaText << " private(" << plan.privateList() << ")";
    }

    std::string reductionText = plan.reductionClause();
    if (!reductionText.empty()) {
        pragmaText << " " << reductionText;
    }

    // OpenACC's reduction() already implies the data transfer, so reduction
    // variables are deliberately not folded into the copy clause here (unlike
    // the OpenMP backend).
    const std::vector<loomX::MappedEntry>& entries = plan.mappedEntries();
    if (!entries.empty()) {
        pragmaText << " " << buildCopyClause(entries);
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


std::string OpenACCCodeGen::buildCopyClause(
    const std::vector<loomX::MappedEntry>& entries) {
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
    for (const loomX::MappedEntry& entry : entries) {
        if (entry.name.empty()) continue;
        std::string accDir = toOpenACCDirection(entry.direction);
        auto it = directionByName.find(entry.name);
        if (it == directionByName.end() ||
            directionRank(accDir) > directionRank(it->second)) {
            directionByName[entry.name] = accDir;
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
    return ParallelForPlan::renderVarList(vars);
}

std::string OpenACCCodeGen::buildReductionClause(
    const std::vector<loomX::ReductionInfo>& reductionDetails) {
    return ParallelForPlan::renderReductionClause(reductionDetails);
}

void OpenACCCodeGen::postProcessSource(std::string& source) {
    if (source.find("#include <openacc.h>") == std::string::npos &&
        source.find("#include \"openacc.h\"") == std::string::npos) {
        source = std::string("#include <openacc.h>\n\n") + source;
    }
}
