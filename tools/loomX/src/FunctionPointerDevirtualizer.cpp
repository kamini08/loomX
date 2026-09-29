#include "FunctionPointerDevirtualizer.h"
#include <iostream>
#include <map>
#include <set>

namespace loomX {

namespace {

// Extract a function declaration from an expression that may be a bare
// function reference or an address-of operation.
static SgFunctionDeclaration* functionFromExpr(SgExpression* e) {
    if (!e) return nullptr;
    if (SgFunctionRefExp* ref = isSgFunctionRefExp(e)) {
        return ref->getAssociatedFunctionDeclaration();
    }
    if (SgAddressOfOp* addr = isSgAddressOfOp(e)) {
        return functionFromExpr(addr->get_operand());
    }
    if (SgCastExp* c = isSgCastExp(e)) {
        return functionFromExpr(c->get_operand());
    }
    return nullptr;
}

// True if e is a reference to the variable var.
static bool refersToVar(SgExpression* e, SgInitializedName* var) {
    if (!e || !var) return false;
    e = isSgCastExp(e) ? isSgCastExp(e)->get_operand() : e;
    SgVarRefExp* ref = isSgVarRefExp(e);
    if (!ref) return false;
    return ref->get_symbol()->get_declaration() == var;
}

// Check whether a variable is a function-pointer type.
static bool isFunctionPointerType(SgType* t) {
    if (!t) return false;
    t = t->stripType(SgType::STRIP_TYPEDEF_TYPE);
    return isSgPointerType(t) != nullptr &&
           isSgFunctionType(t->findBaseType()) != nullptr;
}

// Find candidate function-pointer variables assigned a conditional of two
// functions, with no other assignments to the variable between the initializer
// and its uses.
struct DevirtCandidate {
    SgInitializedName* var = nullptr;
    SgExpression* condition = nullptr;
    SgFunctionDeclaration* funcA = nullptr;
    SgFunctionDeclaration* funcB = nullptr;
    SgStatement* definingStmt = nullptr;
};

static std::vector<DevirtCandidate> findCandidates(SgProject* project) {
    std::vector<DevirtCandidate> candidates;
    Rose_STL_Container<SgNode*> vars =
        NodeQuery::querySubTree(project, V_SgInitializedName);
    for (SgNode* node : vars) {
        SgInitializedName* var = isSgInitializedName(node);
        if (!var || !isFunctionPointerType(var->get_type())) continue;

        SgExpression* init = var->get_initializer();
        if (!init) {
            // Try assignment in declaration statement.
            SgVariableDeclaration* decl = isSgVariableDeclaration(var->get_declaration());
            if (decl && !decl->get_variables().empty()) {
                init = decl->get_variables().front()->get_initializer();
            }
        }
        if (!init) continue;

        // Unwrap SgAssignInitializer if present.
        if (SgAssignInitializer* ai = isSgAssignInitializer(init)) {
            init = ai->get_operand();
        }

        // Strip casts from the initializer.
        while (SgCastExp* c = isSgCastExp(init)) init = c->get_operand();

        SgConditionalExp* cond = isSgConditionalExp(init);
        if (!cond) continue;

        SgFunctionDeclaration* fa = functionFromExpr(cond->get_true_exp());
        SgFunctionDeclaration* fb = functionFromExpr(cond->get_false_exp());
        if (!fa || !fb) continue;

        // Reject trivial same-target conditionals.
        if (fa == fb) continue;

        DevirtCandidate cand;
        cand.var = var;
        cand.condition = cond->get_conditional_exp();
        cand.funcA = fa;
        cand.funcB = fb;
        cand.definingStmt = SageInterface::getEnclosingStatement(var);
        candidates.push_back(cand);
    }
    return candidates;
}

} // anonymous namespace

std::size_t devirtualizeFunctionPointers(SgProject* project) {
    if (!project) return 0;

    std::vector<DevirtCandidate> candidates = findCandidates(project);
    if (candidates.empty()) return 0;

    std::size_t transformed = 0;
    for (const DevirtCandidate& cand : candidates) {
        // Find all call sites through this variable.
        Rose_STL_Container<SgNode*> calls =
            NodeQuery::querySubTree(project, V_SgFunctionCallExp);
        for (SgNode* node : calls) {
            SgFunctionCallExp* call = isSgFunctionCallExp(node);
            if (!call) continue;

            SgExpression* funcExpr = call->get_function();
            if (!refersToVar(funcExpr, cand.var)) continue;

            // Deep-copy condition and arguments so each call site is independent.
            SgExpression* condCopy = isSgExpression(SageInterface::deepCopy(cand.condition));
            if (!condCopy) continue;

            SgExprListExp* args = isSgExprListExp(
                SageInterface::deepCopy(call->get_args()));
            if (!args) continue;

            SgFunctionSymbol* symA = isSgFunctionSymbol(
                cand.funcA->search_for_symbol_from_symbol_table());
            SgFunctionSymbol* symB = isSgFunctionSymbol(
                cand.funcB->search_for_symbol_from_symbol_table());
            if (!symA || !symB) continue;

            SgFunctionRefExp* refA = SageBuilder::buildFunctionRefExp(symA);
            SgFunctionRefExp* refB = SageBuilder::buildFunctionRefExp(symB);
            SgFunctionCallExp* callA = SageBuilder::buildFunctionCallExp(refA, args);
            // Need a second copy of args for callB.
            SgExprListExp* argsB = isSgExprListExp(
                SageInterface::deepCopy(call->get_args()));
            if (!argsB) continue;
            SgFunctionCallExp* callB = SageBuilder::buildFunctionCallExp(refB, argsB);

            SgConditionalExp* replacement =
                SageBuilder::buildConditionalExp(condCopy, callA, callB);

            SageInterface::replaceExpression(call, replacement, false);
            ++transformed;
        }
    }

    if (transformed) {
        std::cout << "[FunctionPointerDevirtualizer] transformed " << transformed
                  << " call site(s)\n";
    }
    return transformed;
}

} // namespace loomX
