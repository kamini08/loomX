#include "ReductionHelperInliner.h"
#include "LoopAnalysisTypes.h"
#include "LoopAnalysisUtil.h"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace loomX {

namespace {

// Information about a function that is a candidate for reduction-helper inlining.
struct ReductionHelperInfo {
    SgFunctionDeclaration* decl = nullptr;
    SgFunctionDefinition* def = nullptr;
    int reductionParamIdx = -1;          // pointer parameter being reduced into (-1 for global)
    SgInitializedName* reducedGlobal = nullptr; // global variable being reduced into
    bool isGlobalReduction = false;
    ReductionOp op = ReductionOp::UNKNOWN;
    SgExpression* rhsExpr = nullptr;     // expression on the RHS, in terms of parameters
};

// Strip casts from an expression.
static SgExpression* skipCastsExpr(SgExpression* e) {
    while (SgCastExp* c = isSgCastExp(e)) {
        e = c->get_operand();
    }
    return e;
}

// True if n is a write to a variable declared outside any function (global/file-static).
static bool isGlobalVariable(SgInitializedName* n) {
    if (!n) return false;
    SgScopeStatement* scope = n->get_scope();
    return isSgGlobal(scope) != nullptr;
}

// Check whether the function body is a single statement of the form
//   *param += expr;   or   *param = *param + expr;
// If so, record the pointer parameter index, the reduction operator, and the RHS.
static bool isReductionHelperFunction(SgFunctionDefinition* def, ReductionHelperInfo& info) {
    if (!def) return false;
    SgBasicBlock* body = def->get_body();
    if (!body) return false;

    SgStatementPtrList& stmts = body->get_statements();
    // Allow declarations before the reduction statement (e.g. local temporaries).
    SgExprStatement* reductionStmt = nullptr;
    for (SgStatement* s : stmts) {
        if (isSgExprStatement(s)) {
            if (reductionStmt) return false; // more than one executable statement
            reductionStmt = isSgExprStatement(s);
        } else if (isSgVariableDeclaration(s)) {
            // declarations are OK if they have no side effects
            continue;
        } else {
            return false;
        }
    }
    if (!reductionStmt) return false;

    SgExpression* expr = reductionStmt->get_expression();
    if (!expr) return false;

    SgInitializedName* reducedParam = nullptr;
    SgInitializedName* reducedGlobalVar = nullptr;
    SgExpression* rhs = nullptr;
    ReductionOp op = ReductionOp::UNKNOWN;
    bool isGlobal = false;

    // Try to identify the reduction target from a compound or simple assignment.
    auto setOpFromCompound = [&](SgCompoundAssignOp* compound) -> bool {
        if (isSgPlusAssignOp(compound)) { op = ReductionOp::ADD; return true; }
        if (isSgMultAssignOp(compound)) { op = ReductionOp::MUL; return true; }
        if (isSgMinusAssignOp(compound)) { op = ReductionOp::SUB; return true; }
        if (isSgAndAssignOp(compound)) { op = ReductionOp::BIT_AND; return true; }
        if (isSgIorAssignOp(compound)) { op = ReductionOp::BIT_OR; return true; }
        if (isSgXorAssignOp(compound)) { op = ReductionOp::BIT_XOR; return true; }
        return false;
    };

    // Case 1: compound assignment (*p += expr) or (global += expr)
    if (SgCompoundAssignOp* compound = isSgCompoundAssignOp(expr)) {
        SgExpression* lhs = skipCastsExpr(compound->get_lhs_operand());
        if (SgPointerDerefExp* deref = isSgPointerDerefExp(lhs)) {
            SgVarRefExp* baseVar = isSgVarRefExp(skipCastsExpr(deref->get_operand()));
            if (!baseVar) return false;
            reducedParam = baseVar->get_symbol()->get_declaration();
        } else if (SgVarRefExp* varRef = isSgVarRefExp(lhs)) {
            SgInitializedName* var = varRef->get_symbol()->get_declaration();
            if (!isGlobalVariable(var)) return false;
            reducedGlobalVar = var;
            isGlobal = true;
        } else {
            return false;
        }
        if (!setOpFromCompound(compound)) return false;
        rhs = compound->get_rhs_operand();
    }
    // Case 2: simple assignment (*p = *p + expr) or (global = global + expr)
    else if (SgAssignOp* assign = isSgAssignOp(expr)) {
        SgExpression* lhs = skipCastsExpr(assign->get_lhs_operand());
        SgInitializedName* reducedVar = nullptr;
        if (SgPointerDerefExp* deref = isSgPointerDerefExp(lhs)) {
            SgVarRefExp* baseVar = isSgVarRefExp(skipCastsExpr(deref->get_operand()));
            if (!baseVar) return false;
            reducedVar = reducedParam = baseVar->get_symbol()->get_declaration();
        } else if (SgVarRefExp* varRef = isSgVarRefExp(lhs)) {
            SgInitializedName* var = varRef->get_symbol()->get_declaration();
            if (!isGlobalVariable(var)) return false;
            reducedVar = reducedGlobalVar = var;
            isGlobal = true;
        } else {
            return false;
        }

        SgExpression* rhsExpr = assign->get_rhs_operand();
        // Look for one of the reduction forms.
        auto detect = [&](SgExpression* e) -> std::pair<bool, SgExpression*> {
            e = skipCastsExpr(e);
            if (SgAddOp* add = isSgAddOp(e)) {
                SgExpression* l = skipCastsExpr(add->get_lhs_operand());
                SgExpression* r = skipCastsExpr(add->get_rhs_operand());
                SgVarRefExp* lv = isSgVarRefExp(l);
                SgVarRefExp* rv = isSgVarRefExp(r);
                if (lv && lv->get_symbol()->get_declaration() == reducedVar && rv) {
                    op = ReductionOp::ADD; return {true, r};
                }
                if (rv && rv->get_symbol()->get_declaration() == reducedVar && lv) {
                    op = ReductionOp::ADD; return {true, l};
                }
            } else if (SgMultiplyOp* mul = isSgMultiplyOp(e)) {
                SgExpression* l = skipCastsExpr(mul->get_lhs_operand());
                SgExpression* r = skipCastsExpr(mul->get_rhs_operand());
                SgVarRefExp* lv = isSgVarRefExp(l);
                SgVarRefExp* rv = isSgVarRefExp(r);
                if (lv && lv->get_symbol()->get_declaration() == reducedVar && rv) {
                    op = ReductionOp::MUL; return {true, r};
                }
                if (rv && rv->get_symbol()->get_declaration() == reducedVar && lv) {
                    op = ReductionOp::MUL; return {true, l};
                }
            }
            return {false, nullptr};
        };
        auto result = detect(rhsExpr);
        if (!result.first) return false;
        rhs = result.second;
    }

    if ((!reducedParam && !reducedGlobalVar) || !rhs || op == ReductionOp::UNKNOWN) return false;

    SgFunctionDeclaration* decl = def->get_declaration();
    if (!decl) return false;
    SgInitializedNamePtrList& params = decl->get_args();

    int reductionIdx = -1;
    if (!isGlobal) {
        // The reduced variable must be a pointer parameter of the function.
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i] == reducedParam) {
                reductionIdx = static_cast<int>(i);
                break;
            }
        }
        if (reductionIdx < 0) return false;
        // The reduced parameter must have pointer or array type.
        if (!isPointerOrArrayType(reducedParam->get_type())) return false;
    }

    // Reject functions with global writes, IO, or other side effects.
    // A quick conservative check: any write to a non-parameter, non-local variable.
    {
        std::vector<SgNode*> readRefs, writeRefs;
        SageInterface::collectReadWriteRefs(body, readRefs, writeRefs);
        SgInitializedNamePtrList& p = decl->get_args();
        std::set<SgInitializedName*> paramSet(p.begin(), p.end());

        for (SgNode* ref : writeRefs) {
            SgVarRefExp* varRef = isSgVarRefExp(ref);
            if (!varRef) continue;
            SgInitializedName* var = varRef->get_symbol()->get_declaration();
            if (!var) continue;
            if (paramSet.count(var)) continue;
            if (var->get_scope() == body) continue;
            if (isGlobal && var == reducedGlobalVar) continue; // the reduction target is allowed
            return false; // write to global or outer scope
        }

        // Also reject function calls inside the helper (besides standard library).
        Rose_STL_Container<SgNode*> calls =
            NodeQuery::querySubTree(body, V_SgFunctionCallExp);
        for (SgNode* node : calls) {
            SgFunctionCallExp* call = isSgFunctionCallExp(node);
            if (!call) continue;
            SgFunctionDeclaration* callee = call->getAssociatedFunctionDeclaration();
            if (!callee) return false;
            std::string name = callee->get_name().getString();
            if (name == "printf" || name == "fprintf" || name == "scanf") return false;
            // pure math functions are OK
            if (name == "sqrt" || name == "sin" || name == "cos" || name == "exp" || name == "log") continue;
            // anything else is a side effect we don't want to inline blindly
            return false;
        }
    }

    info.decl = decl;
    info.def = def;
    info.reductionParamIdx = reductionIdx;
    info.reducedGlobal = reducedGlobalVar;
    info.isGlobalReduction = isGlobal;
    info.op = op;
    info.rhsExpr = rhs;
    return true;
}

// Deep-copy an expression and replace parameter references with argument expressions.
static SgExpression* substituteAndCopy(
    SgExpression* expr,
    const std::map<SgInitializedName*, SgExpression*>& paramToArg) {

    if (!expr) return nullptr;
    SgExpression* copy = isSgExpression(SageInterface::deepCopy(expr));
    if (!copy) return nullptr;

    Rose_STL_Container<SgNode*> refs =
        NodeQuery::querySubTree(copy, V_SgVarRefExp);
    for (SgNode* node : refs) {
        SgVarRefExp* varRef = isSgVarRefExp(node);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        auto it = paramToArg.find(var);
        if (it == paramToArg.end()) continue;

        SgExpression* arg = it->second;
        if (!arg) continue;
        // Deep copy the argument so each substitution is independent.
        SgExpression* argCopy = isSgExpression(SageInterface::deepCopy(arg));
        if (!argCopy) continue;

        if (varRef == copy) {
            copy = argCopy;
        } else {
            SageInterface::replaceExpression(varRef, argCopy, false);
        }
    }
    return copy;
}

// Extract a scalar variable from an address-of expression (&var).
static SgInitializedName* variableFromAddressOf(SgExpression* expr) {
    if (!expr) return nullptr;
    expr = skipCastsExpr(expr);
    SgAddressOfOp* addr = isSgAddressOfOp(expr);
    if (!addr) return nullptr;
    SgVarRefExp* varRef = isSgVarRefExp(skipCastsExpr(addr->get_operand()));
    if (!varRef) return nullptr;
    return varRef->get_symbol()->get_declaration();
}

} // anonymous namespace

std::size_t inlineReductionHelpers(SgProject* project) {
    if (!project) return 0;

    // Pass 1: identify candidate helper functions.
    std::map<std::string, ReductionHelperInfo> helpers;
    Rose_STL_Container<SgNode*> funcDefs =
        NodeQuery::querySubTree(project, V_SgFunctionDefinition);
    for (SgNode* node : funcDefs) {
        SgFunctionDefinition* def = isSgFunctionDefinition(node);
        if (!def) continue;
        ReductionHelperInfo info;
        if (isReductionHelperFunction(def, info)) {
            std::string name = def->get_declaration()->get_name().getString();
            helpers[name] = info;
        }
    }

    if (helpers.empty()) return 0;

    std::cout << "[ReductionHelperInliner] found " << helpers.size()
              << " reduction-helper candidate(s)\n";

    // Pass 2: transform call sites.
    std::size_t inlined = 0;
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(project, V_SgFunctionCallExp);

    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;
        SgFunctionDeclaration* calleeDecl = call->getAssociatedFunctionDeclaration();
        if (!calleeDecl) continue;
        std::string calleeName = calleeDecl->get_name().getString();
        auto it = helpers.find(calleeName);
        if (it == helpers.end()) continue;

        const ReductionHelperInfo& info = it->second;

        SgExprListExp* argsExpr = call->get_args();
        if (!argsExpr) continue;
        std::vector<SgExpression*> args(argsExpr->get_expressions().begin(),
                                         argsExpr->get_expressions().end());

        SgInitializedName* reducedVar = nullptr;
        SgInitializedNamePtrList& params = info.decl->get_args();
        std::map<SgInitializedName*, SgExpression*> paramToArg;

        if (info.isGlobalReduction) {
            reducedVar = info.reducedGlobal;
            if (!reducedVar) continue;
            // Global reduction: all arguments feed the RHS expression.
            for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
                paramToArg[params[i]] = args[i];
            }
        } else {
            if (info.reductionParamIdx >= static_cast<int>(args.size())) continue;
            reducedVar = variableFromAddressOf(args[info.reductionParamIdx]);
            if (!reducedVar) continue;
            // Build param -> arg map, skipping the reduced pointer parameter.
            for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
                if (static_cast<int>(i) == info.reductionParamIdx) continue;
                paramToArg[params[i]] = args[i];
            }
        }

        // Substitute parameters into the RHS expression.
        SgExpression* newRhs = substituteAndCopy(info.rhsExpr, paramToArg);
        if (!newRhs) continue;

        // Build the replacement statement: reducedVar += newRhs.
        SgVarRefExp* lhsVar = SageBuilder::buildVarRefExp(reducedVar);
        SgExpression* newAssign = nullptr;
        switch (info.op) {
            case ReductionOp::ADD: newAssign = SageBuilder::buildPlusAssignOp(lhsVar, newRhs); break;
            case ReductionOp::MUL: newAssign = SageBuilder::buildMultAssignOp(lhsVar, newRhs); break;
            case ReductionOp::SUB: newAssign = SageBuilder::buildMinusAssignOp(lhsVar, newRhs); break;
            case ReductionOp::BIT_AND: newAssign = SageBuilder::buildAndAssignOp(lhsVar, newRhs); break;
            case ReductionOp::BIT_OR: newAssign = SageBuilder::buildIorAssignOp(lhsVar, newRhs); break;
            case ReductionOp::BIT_XOR: newAssign = SageBuilder::buildXorAssignOp(lhsVar, newRhs); break;
            default: continue;
        }
        if (!newAssign) continue;

        SgStatement* replacement = SageBuilder::buildExprStatement(newAssign);
        if (!replacement) continue;

        // The call must be the entire statement (not embedded in a larger expression).
        SgStatement* enclosingStmt = SageInterface::getEnclosingStatement(call);
        if (!enclosingStmt) continue;
        SgExprStatement* callStmt = isSgExprStatement(enclosingStmt);
        if (!callStmt) continue;
        if (skipCastsExpr(callStmt->get_expression()) != skipCastsExpr(call)) continue;

        SageInterface::replaceStatement(callStmt, replacement, false);
        inlined++;
    }

    std::cout << "[ReductionHelperInliner] inlined " << inlined
              << " reduction-helper call site(s)\n";
    return inlined;
}

} // namespace loomX
