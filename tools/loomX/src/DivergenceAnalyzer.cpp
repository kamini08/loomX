#include "DivergenceAnalyzer.h"
#include "LoopAnalysisUtil.h"

namespace loomX {

DivergenceResult DivergenceAnalyzer::analyze(SgForStatement* loop) {
    DivergenceResult result;
    if (!loop) {
        result.description = "null loop";
        return result;
    }

    SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
    SgStatement* body = loop->get_loop_body();
    if (!body) {
        result.description = "empty loop body";
        return result;
    }

    // Early exits (break/continue/return inside loop) are strongly divergent.
    if (hasEarlyExit(loop)) {
        result.isDivergent = true;
        result.kind = DivergenceKind::EARLY_EXIT;
        result.description = "Loop contains break/continue/return";
        return result;
    }

    // Inner loops cause thread divergence in a straight GPU offload.
    if (hasInnerLoop(loop)) {
        result.isDivergent = true;
        result.kind = DivergenceKind::INNER_LOOP;
        result.description = "Loop contains nested loop";
        return result;
    }

    // Check if-statements for data-dependent conditions.
    Rose_STL_Container<SgNode*> ifStmts =
        NodeQuery::querySubTree(body, V_SgIfStmt);
    for (SgNode* node : ifStmts) {
        SgIfStmt* ifStmt = isSgIfStmt(node);
        if (!ifStmt) continue;

        SgStatement* condStmt = ifStmt->get_conditional();
        if (!condStmt) continue;

        SgExpression* condition = nullptr;
        if (SgExprStatement* es = isSgExprStatement(condStmt)) {
            condition = es->get_expression();
        }
        if (!condition) continue;

        if (conditionDependsOnData(condition, loopVar)) {
            result.isDivergent = true;
            result.kind = DivergenceKind::DATA_DEPENDENT_IF;
            result.source = ifStmt;
            result.description = "Data-dependent if-statement: " + condition->unparseToString();
            return result;
        }
    }

    // Check switch statements for data-dependent selectors.
    Rose_STL_Container<SgNode*> switchStmts =
        NodeQuery::querySubTree(body, V_SgSwitchStatement);
    for (SgNode* node : switchStmts) {
        SgSwitchStatement* switchStmt = isSgSwitchStatement(node);
        if (!switchStmt) continue;

        SgStatement* selectorStmt = switchStmt->get_item_selector();
        if (!selectorStmt) continue;

        SgExpression* selector = nullptr;
        if (SgExprStatement* es = isSgExprStatement(selectorStmt)) {
            selector = es->get_expression();
        }
        if (selector && conditionDependsOnData(selector, loopVar)) {
            result.isDivergent = true;
            result.kind = DivergenceKind::DATA_DEPENDENT_SWITCH;
            result.source = switchStmt;
            result.description = "Data-dependent switch selector";
            return result;
        }
    }

    // Function calls in the body may hide divergent control flow.
    // We treat this as mild divergence (still parallelizable, but note it).
    if (containsFunctionCallInBody(loop)) {
        result.isDivergent = true;
        result.kind = DivergenceKind::FUNCTION_CALL;
        result.description = "Loop body contains function calls (possible hidden divergence)";
        return result;
    }

    result.isDivergent = false;
    result.kind = DivergenceKind::NONE;
    result.description = "No divergent control flow detected";
    return result;
}

bool DivergenceAnalyzer::conditionDependsOnData(SgExpression* condition,
                                                 SgInitializedName* loopVar) {
    if (!condition) return false;

    // A condition is data-dependent if it reads an array element or pointer-deref.
    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(condition, V_SgPntrArrRefExp);
    if (!arrRefs.empty()) return true;

    Rose_STL_Container<SgNode*> derefRefs =
        NodeQuery::querySubTree(condition, V_SgPointerDerefExp);
    if (!derefRefs.empty()) return true;

    // Conditions that only depend on the loop iterator are uniform across threads
    // (each thread sees a different iterator value, but the branch outcome is
    // predictable per thread and does not cause intra-warp divergence).
    Rose_STL_Container<SgNode*> varRefs =
        NodeQuery::querySubTree(condition, V_SgVarRefExp);
    for (SgNode* node : varRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(node);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (var && var != loopVar) {
            // Reading a scalar other than the loop variable makes the condition
            // potentially data-dependent (e.g., if (flag) where flag is set elsewhere).
            return true;
        }
    }

    return false;
}

bool DivergenceAnalyzer::hasEarlyExit(SgForStatement* loop) {
    if (!loop) return false;
    SgStatement* body = loop->get_loop_body();
    if (!body) return false;

    Rose_STL_Container<SgNode*> exits =
        NodeQuery::querySubTree(body, V_SgBreakStmt);
    if (!exits.empty()) return true;

    exits = NodeQuery::querySubTree(body, V_SgContinueStmt);
    if (!exits.empty()) return true;

    exits = NodeQuery::querySubTree(body, V_SgReturnStmt);
    if (!exits.empty()) return true;

    return false;
}

bool DivergenceAnalyzer::hasInnerLoop(SgForStatement* loop) {
    if (!loop) return false;
    Rose_STL_Container<SgNode*> innerLoops =
        NodeQuery::querySubTree(loop->get_loop_body(), V_SgForStatement);
    return !innerLoops.empty();
}

bool DivergenceAnalyzer::containsFunctionCallInBody(SgForStatement* loop) {
    if (!loop) return false;
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(loop->get_loop_body(), V_SgFunctionCallExp);
    return !calls.empty();
}

} // namespace loomX
