#include "PureFunctionInliner.h"
#include <iostream>
#include <map>
#include <set>

namespace loomX {

namespace {

// Strip casts from an expression.
static SgExpression* skipCastsExpr(SgExpression* e) {
    while (SgCastExp* c = isSgCastExp(e)) {
        e = c->get_operand();
    }
    return e;
}

// Deep-copy an expression and replace parameter references with argument
// expressions.
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

// Count AST nodes under the given node (rough size estimate).
static std::size_t astNodeCount(SgNode* node) {
    if (!node) return 0;
    std::size_t count = 1;
    for (SgNode* child : node->get_traversalSuccessorContainer()) {
        count += astNodeCount(child);
    }
    return count;
}

// True if the call is the entire expression statement (not embedded in a
// larger expression).
static bool isTopLevelStatementCall(SgFunctionCallExp* call) {
    if (!call) return false;
    SgStatement* stmt = SageInterface::getEnclosingStatement(call);
    if (!stmt) return false;
    SgExprStatement* exprStmt = isSgExprStatement(stmt);
    if (!exprStmt) return false;
    return skipCastsExpr(exprStmt->get_expression()) == skipCastsExpr(call);
}

} // anonymous namespace

std::size_t inlinePureHelperFunctions(SgProject* project,
                                      const InterproceduralAnalysis& ipa) {
    if (!project) return 0;

    // Phase 1: identify candidate helper functions.
    std::set<std::string> candidates;
    Rose_STL_Container<SgNode*> funcDefs =
        NodeQuery::querySubTree(project, V_SgFunctionDefinition);
    for (SgNode* node : funcDefs) {
        SgFunctionDefinition* def = isSgFunctionDefinition(node);
        if (!def) continue;
        SgFunctionDeclaration* decl = def->get_declaration();
        if (!decl) continue;

        // Only inline void functions for now.
        SgType* retType = decl->get_type()->get_return_type();
        if (!retType) continue;
        SgType* bareRet = retType->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
        if (!isSgTypeVoid(bareRet)) continue;

        std::string name = decl->get_name().getString();
        const FunctionSummary* summary = ipa.getSummary(name);
        if (!summary) continue;
        if (!summary->hasDefinition) continue;
        if (summary->inRecursiveCycle) continue;
        // Pointer writes through parameters are OK to inline: after
        // substitution the caller's dependence analysis sees them in context.
        // Reject only real side effects (I/O, globals).
        if (summary->hasIOSideEffects) continue;
        if (summary->hasGlobalWrites) continue;
        // For the first version, inline only leaf helpers to avoid the
        // ordering problem of helper A calling helper B.
        if (!summary->isLeaf) continue;

        // Reject functions that are too large to inline blindly.
        if (astNodeCount(def) > 200) continue;

        candidates.insert(name);
    }

    if (candidates.empty()) return 0;

    std::cout << "[PureFunctionInliner] found " << candidates.size()
              << " candidate helper function(s)\n";

    // Phase 2: find and transform call sites inside loops.
    std::size_t inlined = 0;
    Rose_STL_Container<SgNode*> loops =
        NodeQuery::querySubTree(project, V_SgForStatement);

    for (SgNode* loopNode : loops) {
        SgForStatement* loop = isSgForStatement(loopNode);
        if (!loop) continue;

        // Find calls inside this loop.  We copy the list because we will
        // mutate the AST as we iterate.
        Rose_STL_Container<SgNode*> calls =
            NodeQuery::querySubTree(loop, V_SgFunctionCallExp);
        for (SgNode* callNode : calls) {
            SgFunctionCallExp* call = isSgFunctionCallExp(callNode);
            if (!call) continue;
            if (!isTopLevelStatementCall(call)) continue;

            SgFunctionDeclaration* calleeDecl = call->getAssociatedFunctionDeclaration();
            if (!calleeDecl) continue;
            std::string calleeName = calleeDecl->get_name().getString();
            if (!candidates.count(calleeName)) continue;

            SgFunctionDefinition* calleeDef = calleeDecl->get_definition();
            if (!calleeDef) {
                // Try to find the definition elsewhere in the project.
                Rose_STL_Container<SgNode*> defs =
                    NodeQuery::querySubTree(project, V_SgFunctionDefinition);
                for (SgNode* d : defs) {
                    SgFunctionDefinition* fd = isSgFunctionDefinition(d);
                    if (!fd) continue;
                    SgFunctionDeclaration* ddecl = fd->get_declaration();
                    if (ddecl && ddecl->get_name().getString() == calleeName) {
                        calleeDef = fd;
                        break;
                    }
                }
            }
            if (!calleeDef) continue;

            SgFunctionDeclaration* actualDecl = calleeDef->get_declaration();
            if (!actualDecl) continue;

            SgInitializedNamePtrList& params = actualDecl->get_args();
            SgExprListExp* argsExpr = call->get_args();
            if (!argsExpr) continue;
            std::vector<SgExpression*> args(argsExpr->get_expressions().begin(),
                                            argsExpr->get_expressions().end());
            if (args.size() != params.size()) continue;

            // Build parameter -> argument map.
            std::map<SgInitializedName*, SgExpression*> paramToArg;
            bool argsOk = true;
            for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
                if (!params[i]) { argsOk = false; break; }
                paramToArg[params[i]] = args[i];
            }
            if (!argsOk) continue;

            // Deep-copy the function body into a new compound statement so
            // that local variable declarations do not collide with the caller
            // or with other inlined call sites.
            SgBasicBlock* body = calleeDef->get_body();
            if (!body) continue;

            SgBasicBlock* inlineBlock = SageBuilder::buildBasicBlock();
            for (SgStatement* stmt : body->get_statements()) {
                SgStatement* copy = isSgStatement(SageInterface::deepCopy(stmt));
                if (!copy) { argsOk = false; break; }

                // Substitute parameter references in the copied statement.
                Rose_STL_Container<SgNode*> refs =
                    NodeQuery::querySubTree(copy, V_SgVarRefExp);
                for (SgNode* refNode : refs) {
                    SgVarRefExp* varRef = isSgVarRefExp(refNode);
                    if (!varRef) continue;
                    SgInitializedName* var = varRef->get_symbol()->get_declaration();
                    auto it = paramToArg.find(var);
                    if (it == paramToArg.end()) continue;
                    SgExpression* argCopy = isSgExpression(
                        SageInterface::deepCopy(it->second));
                    if (!argCopy) continue;
                    SageInterface::replaceExpression(varRef, argCopy, false);
                }
                inlineBlock->append_statement(copy);
            }
            if (!argsOk) continue;

            // Replace the call statement with the inlined body block.
            SgStatement* enclosingStmt = SageInterface::getEnclosingStatement(call);
            if (!enclosingStmt) continue;

            SageInterface::replaceStatement(enclosingStmt, inlineBlock, false);
            ++inlined;
        }
    }

    std::cout << "[PureFunctionInliner] inlined " << inlined
              << " helper call site(s)\n";
    return inlined;
}

} // namespace loomX
