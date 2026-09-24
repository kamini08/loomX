#include "KernelCodeGen.h"
#include "LoopAnalysisUtil.h"
#include "LoopCanonicalChecker.h"
#include <algorithm>
#include <cctype>
#include <sstream>

using namespace loomX;

namespace {

// ---------------------------------------------------------------------------
// Small text/collection utilities
// ---------------------------------------------------------------------------

// The first statement of a loop body if the body is exactly one for-loop,
// otherwise null.  Used to walk a perfect collapse nest.
SgForStatement* onlyLoopInBody(SgStatement* body) {
    if (!body) return nullptr;
    if (SgBasicBlock* bb = isSgBasicBlock(body)) {
        if (bb->get_statements().size() != 1) return nullptr;
        return isSgForStatement(bb->get_statements()[0]);
    }
    return isSgForStatement(body);
}

// Collect the perfect collapse nest starting at `loop`, outermost first.
std::vector<SgForStatement*> collectLoopChain(SgForStatement* loop,
                                              int collapseDepth) {
    std::vector<SgForStatement*> chain;
    SgForStatement* cur = loop;
    chain.push_back(cur);
    while (static_cast<int>(chain.size()) < collapseDepth) {
        SgForStatement* inner = onlyLoopInBody(cur->get_loop_body());
        if (!inner) break;
        chain.push_back(inner);
        cur = inner;
    }
    return chain;
}

// Send a single-statement body to a vector.
std::vector<SgStatement*> flattenBody(SgStatement* body) {
    std::vector<SgStatement*> stmts;
    if (!body) return stmts;
    if (SgBasicBlock* bb = isSgBasicBlock(body)) {
        stmts = bb->get_statements();
    } else {
        stmts.push_back(body);
    }
    return stmts;
}

// Collection of the variable declaration of every VarRef in a subtree.
void collectVarRefs(SgNode* root, std::set<SgInitializedName*>& out) {
    if (!root) return;
    Rose_STL_Container<SgNode*> refs =
        NodeQuery::querySubTree(root, V_SgVarRefExp);
    for (SgNode* n : refs) {
        SgVarRefExp* vr = isSgVarRefExp(n);
        if (!vr) continue;
        if (SgInitializedName* v =
                vr->get_symbol() ? vr->get_symbol()->get_declaration() : nullptr) {
            out.insert(v);
        }
    }
}

bool exprReferencesAny(SgExpression* expr, const std::set<SgInitializedName*>& vars) {
    if (!expr) return false;
    std::set<SgInitializedName*> refs;
    collectVarRefs(expr, refs);
    for (SgInitializedName* v : refs) {
        if (vars.count(v)) return true;
    }
    return false;
}

std::string stripType(SgType* type) {
    if (!type) return "void";
    type = type->stripType(SgType::STRIP_TYPEDEF_TYPE | SgType::STRIP_MODIFIER_TYPE);
    return type->unparseToString();
}

// Innermost element type name of an array type (or the type itself if not an
// array).  Returns "" for unsupported types.
std::string elementTypeName(SgType* type) {
    if (!type) return "";
    type = type->stripType(SgType::STRIP_TYPEDEF_TYPE | SgType::STRIP_MODIFIER_TYPE);
    while (SgArrayType* arr = isSgArrayType(type)) {
        type = arr->get_base_type();
        type = type->stripType(SgType::STRIP_TYPEDEF_TYPE | SgType::STRIP_MODIFIER_TYPE);
    }
    if (isSgPointerType(type) || isSgReferenceType(type)) return "";
    return type->unparseToString();
}

// Collect dimension expressions of an array type, outermost first.  Returns
// false when any dimension is unknown (incomplete type / pure pointer).
bool arrayDims(SgType* type, std::vector<SgExpression*>& dims) {
    dims.clear();
    if (!type) return false;
    type = type->stripType(SgType::STRIP_TYPEDEF_TYPE | SgType::STRIP_MODIFIER_TYPE);
    while (SgArrayType* arr = isSgArrayType(type)) {
        SgExpression* index = arr->get_index();
        if (!index) return false;
        dims.push_back(index);
        type = arr->get_base_type();
        type = type->stripType(SgType::STRIP_TYPEDEF_TYPE | SgType::STRIP_MODIFIER_TYPE);
    }
    if (isSgPointerType(type) || isSgReferenceType(type)) return false;
    // A scalar (or otherwise non-array) type is not an array.
    return !dims.empty();
}

std::string arraySizeExpr(const std::string& elemType,
                          const std::vector<SgExpression*>& dims) {
    std::ostringstream oss;
    oss << "sizeof(" << elemType << ")";
    for (SgExpression* dim : dims) {
        oss << " * (" << dim->unparseToString() << ")";
    }
    return oss.str();
}

// Build a multi-line C string literal so arbitrary text can be embedded into
// the generated host code (used for OpenCL kernel sources).
std::string toCStringLiteral(const std::string& text) {
    std::string lit = "\"";
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == '\\' || c == '"') {
            lit += '\\';
            lit += c;
        } else if (c == '\n') {
            lit += "\\n\"\n\"";
        } else if (c == '\r' || c == '\t') {
            lit += ' ';
        } else {
            lit += c;
        }
    }
    lit += "\"";
    return lit;
}

bool isAllowedDeviceCallee(const std::string& name) {
    static const std::set<std::string> allowed = {
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
        "sqrt", "cbrt", "pow", "hypot", "erf", "erfc", "tgamma", "lgamma",
        "fabs", "fmin", "fmax", "fmod", "floor", "ceil", "round", "trunc",
        "fma"
    };
    return allowed.count(name) > 0;
}

// A scalar is "live out" when it is written in the loop body and read later in
// the enclosing function, unless it is a reduction variable (whose final value
// is reconstructed by the copy-back).
bool hasLiveOutScalar(SgForStatement* loop,
                      const loomX::LoopSummary& summary) {
    SgStatement* body = loop->get_loop_body();
    if (!body) return false;

    std::set<SgInitializedName*> reductionVars = summary.getReductionVariables();

    // Every loop index variable inside this loop (the parallelized loop itself
    // plus any nested/collapsed loops) is re-declared per thread in the kernel,
    // so writes to them are not live-out scalar writes.
    std::set<SgInitializedName*> indexVars;
    Rose_STL_Container<SgNode*> innerLoops =
        NodeQuery::querySubTree(loop, V_SgForStatement);
    for (SgNode* n : innerLoops) {
        if (SgForStatement* L = isSgForStatement(n)) {
            if (SgInitializedName* iv = SageInterface::getLoopIndexVariable(L)) {
                indexVars.insert(iv);
            }
        }
    }

    std::set<SgInitializedName*> writtenScalars;
    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(body, readRefs, writeRefs);
    for (SgNode* n : writeRefs) {
        SgVarRefExp* vr = isSgVarRefExp(n);
        if (!vr) continue;
        SgInitializedName* v = vr->get_symbol()->get_declaration();
        if (!v || indexVars.count(v) || reductionVars.count(v)) continue;
        SgType* t = v->get_type();
        if (!t) continue;
        t = t->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
        if (isSgPointerType(t) || isSgArrayType(t)) continue;
        SgScopeStatement* scope = v->get_scope();
        bool inside = false;
        for (SgNode* cur = scope; cur && !isSgFunctionDefinition(cur);
             cur = cur->get_parent()) {
            if (cur == body) {
                inside = true;
                break;
            }
        }
        if (!inside) writtenScalars.insert(v);
    }
    if (writtenScalars.empty()) return false;

    long loopEnd = loop->get_file_info() ? loop->get_file_info()->get_line() : 0;
    Rose_STL_Container<SgNode*> loopNodes = NodeQuery::querySubTree(loop, V_SgNode);
    for (SgNode* node : loopNodes) {
        if (node->get_file_info()) {
            loopEnd = std::max(loopEnd,
                               static_cast<long>(node->get_file_info()->get_line()));
        }
    }

    SgFunctionDefinition* def = getEnclosingFunctionDefinition(loop);
    if (!def) return false;
    std::vector<SgNode*> fReads, fWrites;
    SageInterface::collectReadWriteRefs(def->get_body(), fReads, fWrites);
    for (SgNode* n : fReads) {
        SgVarRefExp* vr = isSgVarRefExp(n);
        if (!vr) continue;
        SgInitializedName* v = vr->get_symbol()->get_declaration();
        if (!v || !writtenScalars.count(v)) continue;
        if (vr->get_file_info() &&
            vr->get_file_info()->get_line() > loopEnd) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// AST rewrites on the (about-to-be-deleted) loop body
// ---------------------------------------------------------------------------

// The reductions we atomically support and their element types.
struct ReductionTarget {
    SgInitializedName* var;
    std::string elemType;   // "int", "float" or "double"
    std::string opKind;     // "add" or "sub" resolved from statement shape
};

// Validate that every write to a reduction variable has a supported shape and
// infer the operator used.  Also refuses array-element reductions and
// non-atomic-supported scalar types.  Fills `targets` for the rewrite pass.
bool resolveReductions(
    SgStatement* body,
    const std::vector<loomX::ReductionInfo>& reductions,
    std::map<SgInitializedName*, ReductionTarget>& targets) {
    targets.clear();
    for (const loomX::ReductionInfo& r : reductions) {
        if (!r.variable) return false;
        if (r.isArrayElement) return false;  // v1: no array-element reductions
        ReductionOp op = r.op;
        if (op != ReductionOp::ADD && op != ReductionOp::SUB) return false;
        std::string elem = elementTypeName(r.variable->get_type());
        if (elem != "int" && elem != "float" && elem != "double") return false;
        ReductionTarget t;
        t.var = r.variable;
        t.elemType = elem;
        t.opKind = (op == ReductionOp::SUB) ? "sub" : "add";
        targets[r.variable] = t;
    }
    if (targets.empty()) return true;

    Rose_STL_Container<SgNode*> all = NodeQuery::querySubTree(body, V_SgNode);
    auto isTargetVar = [&](SgExpression* e) -> SgInitializedName* {
        SgVarRefExp* vr = isSgVarRefExp(skipCasts(e));
        if (!vr) return nullptr;
        SgInitializedName* v = vr->get_symbol()->get_declaration();
        return targets.count(v) ? v : nullptr;
    };

    // First pass: classify every assignment whose LHS is a reduction variable.
    // Accepted statements (x += e, x -= e, x = x + e / e + x, x = x - e) are
    // remembered so their own variable references are not treated as stray
    // writes in the second pass.
    std::set<const SgNode*> accepted;
    for (SgNode* node : all) {
        if (SgCompoundAssignOp* cop = isSgCompoundAssignOp(node)) {
            SgInitializedName* v = isTargetVar(cop->get_lhs_operand());
            if (!v) continue;
            if (!isSgPlusAssignOp(cop) && !isSgMinusAssignOp(cop)) return false;
            accepted.insert(cop);
        } else if (SgAssignOp* aop = isSgAssignOp(node)) {
            SgInitializedName* v = isTargetVar(aop->get_lhs_operand());
            if (!v) continue;
            SgExpression* rhs = skipCasts(aop->get_rhs_operand());
            bool ok = false;
            if (SgAddOp* add = isSgAddOp(rhs)) {
                SgVarRefExp* l = isSgVarRefExp(skipCasts(add->get_lhs_operand()));
                SgVarRefExp* rr = isSgVarRefExp(skipCasts(add->get_rhs_operand()));
                SgInitializedName* lv = l ? l->get_symbol()->get_declaration() : nullptr;
                SgInitializedName* rv = rr ? rr->get_symbol()->get_declaration() : nullptr;
                if ((lv == v && rv != v) || (rv == v && lv != v)) ok = true;
            } else if (SgSubtractOp* sub = isSgSubtractOp(rhs)) {
                SgVarRefExp* l = isSgVarRefExp(skipCasts(sub->get_lhs_operand()));
                if (l && l->get_symbol()->get_declaration() == v) ok = true;
            }
            if (!ok) return false;
            accepted.insert(aop);
        }
    }

    // Second pass: any remaining access to a reduction variable must be a plain
    // read.  Writes (LHS of an assignment, ++/--) outside an accepted statement
    // are unsupported.  opKind was already fixed from the ReductionInfo and is
    // only informational; the rewrite pass re-derives add/sub per statement.
    for (SgNode* node : all) {
        SgVarRefExp* vr = isSgVarRefExp(node);
        if (!vr) continue;
        SgInitializedName* v = vr->get_symbol()->get_declaration();
        if (!v || !targets.count(v)) continue;
        if (isLhsOfAssignment(vr)) {
            bool inside = false;
            for (SgNode* c = vr->get_parent();
                 c && !isSgStatement(c); c = c->get_parent()) {
                if (accepted.count(c)) { inside = true; break; }
            }
            if (inside) continue;
            return false;
        }
    }

    return true;
}

// Rewrite reduction statements into atomic accumulation calls.  Called only
// after resolveReductions() has verified every statement's shape.
bool rewriteReductions(SgStatement* body,
                       CodeGenBackend backend,
                       const std::map<SgInitializedName*, ReductionTarget>& targets) {
    if (targets.empty()) return true;

    auto isTargetVar = [&](SgExpression* e) -> SgInitializedName* {
        SgVarRefExp* vr = isSgVarRefExp(skipCasts(e));
        if (!vr) return nullptr;
        SgInitializedName* v = vr->get_symbol()->get_declaration();
        return targets.count(v) ? v : nullptr;
    };

    // Determine the atomic function name for (backend, element type).
    auto atomicName = [&](const ReductionTarget& t) -> std::string {
        if (backend == CodeGenBackend::CUDA) return "atomicAdd";
        if (t.elemType == "int") return "loomXAtomicAddInt";
        if (t.elemType == "float") return "loomXAtomicAddFloat";
        return "loomXAtomicAddDouble";
    };

    // Classify and collect the statements to rewrite BEFORE touching the tree,
    // so no candidate node is invalidated by replacing another one first (nested
    // reduction assignments inside larger expressions are left intact).
    struct RedRewrite {
        SgExpression* node;
        SgInitializedName* target;
        SgExpression* rhs;
        bool negate;
    };
    std::vector<RedRewrite> rewrites;

    auto isNestedInAssignLike = [](SgExpression* e) {
        for (SgNode* p = e->get_parent(); p && !isSgExprStatement(p);
             p = p->get_parent()) {
            if (isSgAssignOp(p) || isSgCompoundAssignOp(p)) return true;
        }
        return false;
    };

    Rose_STL_Container<SgNode*> nodes = NodeQuery::querySubTree(body, V_SgExpression);
    for (SgNode* node : nodes) {
        SgInitializedName* target = nullptr;
        bool negate = false;
        SgExpression* rhs = nullptr;

        if (SgPlusAssignOp* op = isSgPlusAssignOp(node)) {
            target = isTargetVar(op->get_lhs_operand());
            if (!target) continue;
            rhs = op->get_rhs_operand();
            negate = false;
        } else if (SgMinusAssignOp* op = isSgMinusAssignOp(node)) {
            target = isTargetVar(op->get_lhs_operand());
            if (!target) continue;
            rhs = op->get_rhs_operand();
            negate = true;
        } else if (SgAssignOp* op = isSgAssignOp(node)) {
            target = isTargetVar(op->get_lhs_operand());
            if (!target) continue;
            SgExpression* r = skipCasts(op->get_rhs_operand());
            SgExpression* other = nullptr;
            char shape = 0;
            if (SgAddOp* add = isSgAddOp(r)) {
                SgVarRefExp* l = isSgVarRefExp(skipCasts(add->get_lhs_operand()));
                SgVarRefExp* rr = isSgVarRefExp(skipCasts(add->get_rhs_operand()));
                SgInitializedName* lv = l ? l->get_symbol()->get_declaration() : nullptr;
                SgInitializedName* rv = rr ? rr->get_symbol()->get_declaration() : nullptr;
                if (lv == target && rv != target) { other = add->get_rhs_operand(); shape = 'a'; }
                else if (rv == target && lv != target) { other = add->get_lhs_operand(); shape = 'a'; }
            } else if (SgSubtractOp* sub = isSgSubtractOp(r)) {
                SgVarRefExp* l = isSgVarRefExp(skipCasts(sub->get_lhs_operand()));
                if (l && l->get_symbol()->get_declaration() == target) {
                    other = sub->get_rhs_operand();
                    shape = 's';
                }
            }
            if (shape == 'a') { rhs = other; negate = false; }
            else if (shape == 's') { rhs = other; negate = true; }
            else continue;
        } else {
            continue;
        }

        if (target && !isNestedInAssignLike(isSgExpression(node))) {
            rewrites.push_back({isSgExpression(node), target, rhs, negate});
        }
    }

    for (const RedRewrite& rw : rewrites) {
        const ReductionTarget& t = targets.at(rw.target);
        std::string fnName = atomicName(t);

        SgExpression* value = SageInterface::deepCopy(rw.rhs);
        if (rw.negate) {
            value = SageBuilder::buildMinusOp(value, SgUnaryOp::prefix);
        }
        SgExprListExp* args = SageBuilder::buildExprListExp();
        SgVarRefExp* varRef = SageBuilder::buildVarRefExp(t.var);
        args->append_expression(SageBuilder::buildAddressOfOp(varRef));
        args->append_expression(value);

        SgFunctionCallExp* call = SageBuilder::buildFunctionCallExp(
            SgName(fnName.c_str()), SageBuilder::buildVoidType(), args,
            t.var->get_scope());
        if (!call) return false;
        SageInterface::replaceExpression(rw.node, call, true);
    }
    return true;
}

// Flatten the subscripts of every multi-dimensional array access of a
// kernel-passed array into a single flat pointer reference, e.g.
//   A[i][j]  ->  A[(i) * (M) + (j)]
// so the device kernel can take a flat element pointer parameter.
bool linearizeSubscripts(SgStatement* body,
                         const std::set<SgInitializedName*>& arrayVars) {
    if (arrayVars.empty()) return true;

    Rose_STL_Container<SgNode*> refs =
        NodeQuery::querySubTree(body, V_SgPntrArrRefExp);
    // Only the outermost reference of each chain (parent not an array ref).
    std::vector<SgPntrArrRefExp*> tops;
    for (SgNode* n : refs) {
        SgPntrArrRefExp* ref = isSgPntrArrRefExp(n);
        if (!ref) continue;
        if (isSgPntrArrRefExp(ref->get_parent())) continue;
        tops.push_back(ref);
    }

    for (SgPntrArrRefExp* top : tops) {
        SgVarRefExp* baseRef = nullptr;
        SgExpression* chain = top->get_lhs_operand();
        SgExpression* base = chain;
        while (SgPntrArrRefExp* inner = isSgPntrArrRefExp(base)) {
            base = inner->get_lhs_operand();
        }
        SgVarRefExp* bv = isSgVarRefExp(skipCasts(base));
        if (!bv) continue;
        SgInitializedName* var = bv->get_symbol()->get_declaration();
        if (!var) continue;

        // The base variable is treated as a flat pointer only if it is one of
        // the kernel-passed arrays.
        std::vector<SgExpression*> dims;
        if (!arrayDims(var->get_type(), dims)) continue;
        if (dims.size() < 2) continue;  // 1-D already flat

        // Collect subscripts in source order: A[i][j] -> {i, j}, which is also
        // outermost-dim-first, matching the order produced by arrayDims().
        std::vector<SgExpression*> indices;
        SgExpression* e = chain;
        while (SgPntrArrRefExp* inner = isSgPntrArrRefExp(e)) {
            indices.push_back(inner->get_rhs_operand());
            e = inner->get_lhs_operand();
        }
        // Append the final rhs (of `top`).
        indices.push_back(top->get_rhs_operand());

        // For arrayVars with `dims.size() < indices.size()` typical arrays...
        // Guard: ranks must match.
        if (indices.size() != dims.size()) continue;

        // flat = sum_k indices[k] * prod_{m>k} dims[m]
        SgExpression* flat = nullptr;
        for (size_t k = 0; k < indices.size(); ++k) {
            SgExpression* term = SageInterface::deepCopy(indices[k]);
            for (size_t m = k + 1; m < indices.size(); ++m) {
                SgExpression* stride = SageInterface::deepCopy(dims[m]);
                term = SageBuilder::buildMultiplyOp(term, stride);
            }
            flat = flat ? SageBuilder::buildAddOp(flat, term) : term;
        }
        if (!flat) continue;

        SgExpression* newRef = SageBuilder::buildPntrArrRefExp(
            SageBuilder::buildVarRefExp(var), flat);
        SageInterface::replaceExpression(top, newRef, true);
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// KernelCodeGen
// ---------------------------------------------------------------------------

bool KernelCodeGen::generate(SgForStatement* loop,
                             const loomX::LoopSummary& summary) {
    if (!loop) return false;
    if (summary.target != ParallelTarget::GPU_OFFLOAD) return false;

    int collapseDepth = summary.collapseDepth > 0 ? summary.collapseDepth : 1;
    std::vector<SgForStatement*> chain = collectLoopChain(loop, collapseDepth);
    if (chain.empty()) return false;

    LoopCanonicalChecker canon;
    for (SgForStatement* L : chain) {
        if (canon.analyze(L).form != CanonicalForm::CANONICAL) return false;
    }

    SgForStatement* innermost = chain.back();
    SgStatement* body = innermost->get_loop_body();
    std::vector<SgStatement*> bodyStmts = flattenBody(body);
    if (bodyStmts.empty()) return false;

    // --- 1. Control flow sanity -------------------------------------------
    for (SgStatement* s : bodyStmts) {
        Rose_STL_Container<SgNode*> breaks =
            NodeQuery::querySubTree(s, V_SgBreakStmt);
        Rose_STL_Container<SgNode*> continues =
            NodeQuery::querySubTree(s, V_SgContinueStmt);
        Rose_STL_Container<SgNode*> returns =
            NodeQuery::querySubTree(s, V_SgReturnStmt);
        if (!breaks.empty() || !continues.empty() || !returns.empty()) {
            return false;
        }
    }

    // --- 2. Function-call whitelist ----------------------------------------
    for (SgStatement* s : bodyStmts) {
        Rose_STL_Container<SgNode*> calls =
            NodeQuery::querySubTree(s, V_SgFunctionCallExp);
        for (SgNode* n : calls) {
            SgFunctionCallExp* call = isSgFunctionCallExp(n);
            if (!call) continue;
            if (SgFunctionDeclaration* decl =
                    call->getAssociatedFunctionDeclaration()) {
                std::string name = decl->get_name().getString();
                if (!name.empty() && !isAllowedDeviceCallee(name)) return false;
            }
        }
    }

    // --- 3. Scalar live-out check ------------------------------------------
    if (hasLiveOutScalar(loop, summary)) return false;

    // --- 4. Reduction resolution -------------------------------------------
    std::map<SgInitializedName*, ReductionTarget> targets;
    // resolveReductions() validates shapes on the body; do it once.
    for (SgStatement* s : bodyStmts) {
        std::map<SgInitializedName*, ReductionTarget> t;
        if (!resolveReductions(s, summary.reductions, t)) return false;
        targets.insert(t.begin(), t.end());
    }

    // --- 5. Classify the kernel-passed variables ---------------------------
    std::set<SgInitializedName*> indexVars;
    for (SgForStatement* L : chain) {
        if (SgInitializedName* iv = SageInterface::getLoopIndexVariable(L)) {
            indexVars.insert(iv);
        }
    }
    // Include the indices of any loops nested deeper in the body (a
    // non-collapsed outer loop can still contain inner for-loops); they are
    // re-declared inside the kernel, never passed as parameters.
    Rose_STL_Container<SgNode*> innerLoops =
        NodeQuery::querySubTree(loop, V_SgForStatement);
    for (SgNode* n : innerLoops) {
        if (SgForStatement* L = isSgForStatement(n)) {
            if (SgInitializedName* iv = SageInterface::getLoopIndexVariable(L)) {
                indexVars.insert(iv);
            }
        }
    }

    // Collect all references from the body plus the loop bounds.
    std::set<SgInitializedName*> allVars;
    for (SgStatement* s : bodyStmts) collectVarRefs(s, allVars);
    for (SgForStatement* L : chain) {
        const CanonicalResult& c = canon.analyze(L);
        if (c.lowerBound) collectVarRefs(c.lowerBound, allVars);
        if (c.upperBound) collectVarRefs(c.upperBound, allVars);
        if (c.stride) collectVarRefs(c.stride, allVars);
    }

    // A variable declared inside the loop body stays local in the kernel.
    auto declaredInsideLoopBody = [&](SgInitializedName* var) {
        for (SgNode* cur = var->get_scope(); cur && !isSgFunctionDefinition(cur);
             cur = cur->get_parent()) {
            if (cur == body) return true;
        }
        return false;
    };

    // Variable-length array dimensions reference host variables that the kernel
    // needs as plain scalar parameters (e.g. `double A[n][m]`).  Grow allVars
    // with every variable appearing in a kernel-passed array's extent so the
    // classification below passes them by value.
    bool grew = true;
    while (grew) {
        grew = false;
        for (SgInitializedName* var : allVars) {
            if (!var || indexVars.count(var)) continue;
            std::vector<SgExpression*> dims;
            if (!arrayDims(var->get_type(), dims)) continue;
            for (SgExpression* dim : dims) {
                std::set<SgInitializedName*> refs;
                collectVarRefs(dim, refs);
                for (SgInitializedName* v : refs) {
                    if (!v || indexVars.count(v) || declaredInsideLoopBody(v)) continue;
                    if (allVars.insert(v).second) grew = true;
                }
            }
        }
    }

    std::vector<std::pair<KernelParam, SgInitializedName*>> params;
    std::set<SgInitializedName*> arrayVars;
    for (SgInitializedName* var : allVars) {
        if (!var) continue;
        if (indexVars.count(var)) continue;

        // Loop-body-locals stay local in the kernel.
        if (declaredInsideLoopBody(var)) continue;

        SgType* type = var->get_type();
        std::vector<SgExpression*> dims;
        bool isArray = arrayDims(type, dims);
        if (isArray) {
            if (targets.count(var)) return false;  // reductions on arrays: v1 no
            KernelParam kp;
            kp.kind = KernelParamKind::ARRAY;
            kp.name = var->get_name().getString();
            kp.typeName = elementTypeName(type);
            if (kp.typeName.empty()) return false;
            kp.sizeExpr = arraySizeExpr(kp.typeName, dims);
            // Map direction from the summary.
            kp.mappedTo = true;
            kp.mappedFrom = true;
            for (const auto& p : summary.mapClauses) {
                if (p.first != var) continue;
                if (p.second == "to") { kp.mappedTo = true; kp.mappedFrom = false; }
                else if (p.second == "from") { kp.mappedTo = false; kp.mappedFrom = true; }
                else { kp.mappedTo = true; kp.mappedFrom = true; }
            }
            params.push_back({kp, var});
            arrayVars.insert(var);
        } else if (isSgPointerType(type)) {
            // Pure pointer: extent unknown at compile time.
            return false;
        } else if (targets.count(var)) {
            KernelParam kp;
            kp.kind = KernelParamKind::REDUCTION;
            kp.name = var->get_name().getString();
            kp.typeName = targets[var].elemType;
            kp.sizeExpr = "sizeof(" + kp.typeName + ")";
            params.push_back({kp, var});
        } else {
            KernelParam kp;
            kp.kind = KernelParamKind::SCALAR;
            kp.name = var->get_name().getString();
            kp.typeName = stripType(type);
            params.push_back({kp, var});
        }
    }

    // Deterministic ordering: arrays, then scalars, then reductions, each by name.
    auto paramRank = [](KernelParamKind k) {
        if (k == KernelParamKind::ARRAY) return 0;
        if (k == KernelParamKind::SCALAR) return 1;
        return 2;
    };
    std::stable_sort(params.begin(), params.end(),
                     [&](const std::pair<KernelParam, SgInitializedName*>& a,
                         const std::pair<KernelParam, SgInitializedName*>& b) {
                         if (paramRank(a.first.kind) != paramRank(b.first.kind))
                             return paramRank(a.first.kind) < paramRank(b.first.kind);
                         return a.first.name < b.first.name;
                     });

    // --- 6. Collapsed dims / index invariance ------------------------------
    struct DimInfo {
        std::string lower;
        std::string upper;
        bool upperExclusive = false;  // true when the upper bound is `<`, not `<=`
        std::string indexName;
        std::string typeName;
    };
    std::vector<DimInfo> dims;
    for (SgForStatement* L : chain) {
        const CanonicalResult& c = canon.analyze(L);
        if (!c.lowerBound || !c.upperBound) return false;
        // Inner collapsed loops must have bounds that do not depend on any
        // collapsed index (the decomposition divides by their trip counts).
        if (L != chain.front() &&
            (exprReferencesAny(c.lowerBound, indexVars) ||
             exprReferencesAny(c.upperBound, indexVars))) {
            return false;
        }
        DimInfo d;
        d.lower = c.lowerBound->unparseToString();
        d.upper = c.upperBound->unparseToString();
        d.upperExclusive = c.isUpperExclusive;
        d.indexName = c.indexVar ? c.indexVar->get_name().getString() : "";
        d.typeName = c.indexVar ? stripType(c.indexVar->get_type()) : "int";
        if (d.indexName.empty()) return false;
        dims.push_back(d);
    }

    // --- 7. Mutate the AST (only now that everything is validated) ---------
    if (!rewriteReductions(body, backend_, targets)) return false;
    if (!linearizeSubscripts(body, arrayVars)) return false;

    // --- 8. Assemble per-dimension trip expressions ------------------------
    // trip_i = ((upper) - (lower))   (or +1 when inclusive)
    auto tripText = [](const DimInfo& d) {
        std::string t = "((" + d.upper + ") - (" + d.lower + "))";
        if (!d.upperExclusive) t += " + 1";
        return t;
    };
    std::string totalTrip = tripText(dims[0]);
    for (size_t i = 1; i < dims.size(); ++i) {
        totalTrip = "(" + totalTrip + ") * (" + tripText(dims[i]) + ")";
    }

    // --- 9. Kernel body text ------------------------------------------------
    std::string bodyText;
    for (SgStatement* s : bodyStmts) {
        std::string t = s->unparseToString();
        // Trim trailing whitespace/newlines.
        size_t end = t.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) t.erase(end + 1);
        if (!bodyText.empty()) bodyText += "\n";
        bodyText += t;
    }

    // --- 10. Kernel prologue -------------------------------------------------
    GeneratedKernel gen;
    gen.id = nextId_++;
    gen.backend = backend_;
    gen.hostName = "loomx_" + std::to_string(gen.id);
    gen.kernelName = gen.hostName + "_kernel";
    gen.totalTripText = totalTrip;
    for (const DimInfo& d : dims) {
        GeneratedKernel::Dim gd;
        gd.lowerText = d.lower;
        gd.upperText = d.upper;
        gd.upperExclusive = d.upperExclusive;
        gd.indexName = d.indexName;
        gen.dims.push_back(gd);
    }

    const char* linExpr = (backend_ == CodeGenBackend::OPENCL)
        ? "(size_t)get_global_id(0)"
        : "blockIdx.x * blockDim.x + threadIdx.x";
    gen.prologue.push_back("long __loo = " + std::string(linExpr) + ";");
    gen.prologue.push_back("long __tot = " + totalTrip + ";");
    gen.prologue.push_back("if (__loo >= __tot) return;");

    // Decompose linear id into the loop indices, innermost first.
    if (dims.size() == 1) {
        const DimInfo& d = dims[0];
        gen.prologue.push_back(d.typeName + " " + d.indexName +
                               " = ((" + d.lower + ")) + __loo;");
    } else {
        for (size_t k = dims.size() - 1; k > 0; --k) {
            const DimInfo& d = dims[k];
            std::string step = "__loo";
            std::string trip = tripText(d);
            gen.prologue.push_back(d.typeName + " " + d.indexName +
                                   " = ((" + d.lower + ")) + (__loo % (" +
                                   trip + ")); __loo /= (" + trip + ");");
        }
        const DimInfo& d = dims[0];
        gen.prologue.push_back(d.typeName + " " + d.indexName +
                               " = ((" + d.lower + ")) + __loo;");
    }

    // Declarations for non-collapsed nested loop induction variables that live
    // in an outer scope (so the kernel body's `for (j ...)` statements resolve).
    if (dims.size() == 1) {
        for (SgStatement* s : bodyStmts) {
            Rose_STL_Container<SgNode*> loops =
                NodeQuery::querySubTree(s, V_SgForStatement);
            for (SgNode* n : loops) {
                SgForStatement* inner = isSgForStatement(n);
                if (!inner) continue;
                // Inner loops of the original nest are only a concern when the
                // outer loop was NOT collapsed.
                SgInitializedName* iv = SageInterface::getLoopIndexVariable(inner);
                if (!iv || indexVars.count(iv)) continue;
                SgScopeStatement* scope = iv->get_scope();
                bool local = false;
                for (SgNode* c = scope; c && !isSgFunctionDefinition(c);
                     c = c->get_parent()) {
                    if (c == inner || c == body) {
                        local = true;
                        break;
                    }
                }
                if (!local) {
                    gen.prologue.push_back(stripType(iv->get_type()) + " " +
                                           iv->get_name().getString() + ";");
                }
            }
        }
    }

    // Reduction bindings: CUDA uses references, OpenCL uses preprocessor macros.
    gen.needsReductionHelpers = !targets.empty();
    if (backend_ == CodeGenBackend::CUDA) {
        for (const auto& t : targets) {
            std::string pname = "__loomx_red_" + std::to_string(gen.id) + "_" +
                                t.second.var->get_name().getString();
            gen.prologue.push_back(t.second.elemType + "& " +
                                   t.second.var->get_name().getString() +
                                   " = *" + pname + ";");
        }
    }

    // --- 11. Store params -----------------------------------------------------
    for (const auto& p : params) gen.params.push_back(p.first);

    // --- 12. Build text --------------------------------------------------------
    // Kernel parameter list.
    std::vector<std::string> kernelParams;
    for (const KernelParam& p : gen.params) {
        std::string name = p.name;
        if (p.kind == KernelParamKind::ARRAY) {
            if (backend_ == CodeGenBackend::OPENCL)
                kernelParams.push_back("__global " + p.typeName + "* " + name);
            else
                kernelParams.push_back(p.typeName + "* " + name);
        } else if (p.kind == KernelParamKind::REDUCTION) {
            std::string rn = "__loomx_red_" + std::to_string(gen.id) + "_" + name;
            if (backend_ == CodeGenBackend::OPENCL)
                kernelParams.push_back("__global " + p.typeName + "* " + rn);
            else
                kernelParams.push_back(p.typeName + "* " + rn);
        } else {
            kernelParams.push_back(p.typeName + " " + name);
        }
    }

    // Host launcher parameter list.
    std::vector<std::string> hostParams;
    for (const KernelParam& p : gen.params) {
        if (p.kind == KernelParamKind::ARRAY)
            hostParams.push_back("void* " + p.name);
        else if (p.kind == KernelParamKind::REDUCTION)
            hostParams.push_back(p.typeName + "* " + p.name);
        else
            hostParams.push_back(p.typeName + " " + p.name);
    }

    std::string kpText;
    for (size_t i = 0; i < kernelParams.size(); ++i) {
        if (i) kpText += ", ";
        kpText += kernelParams[i];
    }
    std::string hpText;
    for (size_t i = 0; i < hostParams.size(); ++i) {
        if (i) hpText += ", ";
        hpText += hostParams[i];
    }

    gen.hostSignature = "static void " + gen.hostName + "(" + hpText + ")";

    // Prologue text.
    std::string prologueText;
    for (const std::string& p : gen.prologue) {
        prologueText += "  " + p + "\n";
    }

    if (backend_ == CodeGenBackend::CUDA) {
        std::ostringstream def;
        def << "__global__ void " << gen.kernelName << "(" << kpText << ") {\n"
            << prologueText
            << bodyText << "\n}\n\n"
            << buildHostStubCUDA(gen, kernelParams, hpText, totalTrip);
        gen.definitionText = def.str();
    } else {
        gen.openclSource = buildOpenCLKernel(gen, kernelParams, bodyText,
                                             prologueText, totalTrip);
    }

    // --- 13. Replace the loop with a call to the host stub --------------------
    SgExprListExp* args = SageBuilder::buildExprListExp();
    for (auto& p : params) {
        KernelParam& kp = p.first;
        SgInitializedName* var = p.second;
        SgVarRefExp* vr = SageBuilder::buildVarRefExp(var);
        if (kp.kind == KernelParamKind::REDUCTION) {
            args->append_expression(SageBuilder::buildAddressOfOp(vr));
        } else {
            args->append_expression(vr);
        }
    }
    SgFunctionCallExp* call = SageBuilder::buildFunctionCallExp(
        SgName(gen.hostName.c_str()), SageBuilder::buildVoidType(), args,
        SageInterface::getScope(loop));
    SgExprStatement* callStmt = SageBuilder::buildExprStatement(call);
    SageInterface::insertStatementBefore(loop, callStmt);
    SageInterface::removeStatement(loop);

    kernels_.push_back(gen);
    return true;
}

// Build the CUDA host launcher stub referencing an already-built kernel.
std::string KernelCodeGen::buildHostStubCUDA(
    const GeneratedKernel& gen,
    const std::vector<std::string>& kernelParams,
    const std::string& hostParamsText,
    const std::string& totalTrip) {
    (void)kernelParams;
    std::ostringstream s;
    s << "static void " << gen.hostName << "(" << hostParamsText << ") {\n";

    // Device buffer declarations + allocations + H2D copies.
    for (const KernelParam& p : gen.params) {
        s << "  " << p.typeName << "* __loomx_d_" << p.name << " = 0;\n";
    }
    for (const KernelParam& p : gen.params) {
        std::string d = "__loomx_d_" + p.name;
        if (p.kind == KernelParamKind::ARRAY) {
            s << "  if (cudaMalloc((void**)&" << d << ", (unsigned long)("
              << p.sizeExpr << ")) != cudaSuccess) return;\n";
            if (p.mappedTo) {
                s << "  if (cudaMemcpy(" << d << ", " << p.name
                  << ", (unsigned long)(" << p.sizeExpr
                  << "), cudaMemcpyHostToDevice) != cudaSuccess) return;\n";
            }
        } else if (p.kind == KernelParamKind::REDUCTION) {
            s << "  if (cudaMalloc((void**)&" << d << ", " << p.sizeExpr
              << ") != cudaSuccess) return;\n";
            s << "  if (cudaMemcpy(" << d << ", " << p.name << ", " << p.sizeExpr
              << ", cudaMemcpyHostToDevice) != cudaSuccess) return;\n";
        }
    }

    s << "  long __tot = " << totalTrip << ";\n";
    s << "  if (__tot > 0) {\n";
    s << "    const int __block = 256;\n";
    s << "    int __grid = (int)((__tot + __block - 1) / __block);\n";
    s << "    if (__grid < 1) __grid = 1;\n";
    // Launch args in the same order as the kernel parameters.
    std::vector<std::string> launchArgs;
    for (const KernelParam& p : gen.params) {
        if (p.kind == KernelParamKind::ARRAY) {
            launchArgs.push_back("__loomx_d_" + p.name);
        } else if (p.kind == KernelParamKind::REDUCTION) {
            launchArgs.push_back("__loomx_d_" + p.name);
        } else {
            launchArgs.push_back(p.name);
        }
    }
    std::string laText;
    for (size_t i = 0; i < launchArgs.size(); ++i) {
        if (i) laText += ", ";
        laText += launchArgs[i];
    }
    s << "    " << gen.kernelName << "<<<__grid, __block>>>(" << laText << ");\n";
    s << "    cudaDeviceSynchronize();\n";
    s << "  }\n";

    // D2H copies.
    for (const KernelParam& p : gen.params) {
        std::string d = "__loomx_d_" + p.name;
        if (p.kind == KernelParamKind::ARRAY) {
            if (p.mappedFrom) {
                s << "  if (cudaMemcpy(" << p.name << ", " << d
                  << ", (unsigned long)(" << p.sizeExpr
                  << "), cudaMemcpyDeviceToHost) != cudaSuccess) return;\n";
            }
        } else if (p.kind == KernelParamKind::REDUCTION) {
            s << "  if (cudaMemcpy(" << p.name << ", " << d << ", " << p.sizeExpr
              << ", cudaMemcpyDeviceToHost) != cudaSuccess) return;\n";
        }
    }
    for (const KernelParam& p : gen.params) {
        s << "  cudaFree(__loomx_d_" << p.name << ");\n";
    }
    s << "}\n";
    return s.str();
}

// Build the OpenCL kernel source (kernel function + helpers + reduction macros)
// for one kernel.  %%LOOMX_DEFINES%% is replaced later by the file's #defines.
std::string KernelCodeGen::buildOpenCLKernel(
    const GeneratedKernel& gen,
    const std::vector<std::string>& kernelParams,
    const std::string& bodyText,
    const std::string& prologueText,
    const std::string& totalTrip) {
    std::ostringstream s;
    s << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    s << "%%LOOMX_DEFINES%%\n";
    if (gen.needsReductionHelpers) {
        s <<
            "/* atomic accumulation helpers for reductions */\n"
            "static int loomXAtomicAddInt(volatile __global int* p, int v) {\n"
            "  return atomic_add(p, v);\n"
            "}\n"
            "static float loomXAtomicAddFloat(volatile __global float* p, float v) {\n"
            "  union { unsigned int u; float f; } oldv, newv;\n"
            "  do { oldv.u = *(volatile __global unsigned int*)p; newv.f = oldv.f + v; }\n"
            "  while (atomic_cmpxchg((volatile __global unsigned int*)p, oldv.u, newv.u) != oldv.u);\n"
            "  return newv.f;\n"
            "}\n"
            "static double loomXAtomicAddDouble(volatile __global double* p, double v) {\n"
            "  union { unsigned long u; double f; } oldv, newv;\n"
            "  do { oldv.u = *(volatile __global unsigned long*)p; newv.f = oldv.f + v; }\n"
            "  while (atomic_cmpxchg((volatile __global unsigned long*)p, oldv.u, newv.u) != oldv.u);\n"
            "  return newv.f;\n"
            "}\n";
    }
    std::string kpText;
    for (size_t i = 0; i < kernelParams.size(); ++i) {
        if (i) kpText += ", ";
        kpText += kernelParams[i];
    }
    s << "__kernel void " << gen.kernelName << "(" << kpText << ") {\n";
    s << prologueText;
    // Reduction macros for OpenCL: `#define sum (*(__loomx_red_0_sum))`.
    for (size_t i = 0; i < gen.params.size(); ++i) {
        const KernelParam& p = gen.params[i];
        if (p.kind != KernelParamKind::REDUCTION) continue;
        s << "  #define " << p.name << " (*(__loomx_red_" << gen.id << "_"
          << p.name << "))\n";
    }
    s << bodyText << "\n";
    s << "}\n";
    return s.str();
}

// Splice prototypes near the top and definitions at the bottom of the emitted
// file, and add the required device runtime include.
void KernelCodeGen::finalizeOutput(std::string& source, bool ompUsed) const {
    if (kernels_.empty()) return;

    std::string header;
    if (backend_ == CodeGenBackend::CUDA) header = "#include <cuda_runtime.h>\n";
    else if (backend_ == CodeGenBackend::OPENCL) header = "#include <CL/cl.h>\n";
    else return;
    if (ompUsed) header += "#include <omp.h>\n";
    if (source.find(header) == std::string::npos) source = header + "\n" + source;

    std::string prototypes;
    for (const GeneratedKernel& g : kernels_) {
        if (backend_ == CodeGenBackend::CUDA) {
            prototypes += g.hostSignature + ";\n";
        } else {
            prototypes += g.hostSignature + ";\n";
        }
    }

    std::string definitions;
    if (backend_ == CodeGenBackend::CUDA) {
        for (const GeneratedKernel& g : kernels_) {
            definitions += g.definitionText + "\n";
        }
    } else {
        // Collect single-line #define preprocessor lines from the original file.
        std::string defines;
        {
            std::istringstream iss(source);
            std::string line;
            while (std::getline(iss, line)) {
                std::string trimmed = line;
                size_t b = trimmed.find_first_not_of(" \t");
                if (b == std::string::npos) continue;
                trimmed.erase(0, b);
                if (trimmed.rfind("#define", 0) == 0 &&
                    !trimmed.empty() && trimmed.back() != '\\') {
                    defines += line + "\n";
                }
            }
        }
        for (const GeneratedKernel& g : kernels_) {
            std::string src = g.openclSource;
            size_t slot = src.find("%%LOOMX_DEFINES%%");
            if (slot != std::string::npos) {
                src.replace(slot, std::string("%%LOOMX_DEFINES%%").size(), defines);
            }
            std::string lit = toCStringLiteral(src);

            std::ostringstream s;
            // Shared context/queue statics are per-stub for v1 simplicity.
            s << "static void " << g.hostName << "("
              << hostLaunchSignature(g) << ") {\n";
            s << "  cl_int __err = CL_SUCCESS;\n";
            s << "  static cl_context __ctx = NULL;\n";
            s << "  static cl_command_queue __q = NULL;\n";
            s << "  static cl_program __prog = NULL;\n";
            s << "  static cl_kernel __kern = NULL;\n";
            s << "  static const char* __src = " << lit << ";\n";
            s << "  if (!__ctx) {\n";
            s << "    cl_platform_id __plat = NULL;\n";
            s << "    char __clver[64] = {0};\n";
            s << "    if (clGetPlatformIDs(1, &__plat, NULL) != CL_SUCCESS) return;\n";
            s << "    clGetPlatformInfo(__plat, CL_PLATFORM_VERSION, sizeof(__clver), __clver, NULL);\n";
            s << "    if (__clver[0] >= '2') {\n";  // OpenCL >= 2.0: can embed in a kernel-arg-free context
            s << "      cl_context_properties __props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)__plat, 0 };\n";
            s << "      __ctx = clCreateContextFromType(__props, CL_DEVICE_TYPE_GPU, NULL, NULL, &__err);\n";
            s << "    } else {\n";
            s << "      __ctx = clCreateContextFromType(NULL, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, &__err);\n";
            s << "    }\n";
            s << "    if (!__ctx) return;\n";
            s << "    size_t __devSz = 0;\n";
            s << "    cl_device_id __dev = 0;\n";
            s << "    clGetContextInfo(__ctx, CL_CONTEXT_DEVICES, 0, NULL, &__devSz);\n";
            s << "    if (__devSz == 0) return;\n";
            s << "    clGetContextInfo(__ctx, CL_CONTEXT_DEVICES, sizeof(__dev), &__dev, NULL);\n";
            s << "    __q = clCreateCommandQueue(__ctx, __dev, 0, &__err);\n";
            s << "    if (!__q) return;\n";
            s << "  }\n";
            s << "  if (!__prog) {\n";
            s << "    __prog = clCreateProgramWithSource(__ctx, 1, &__src, NULL, &__err);\n";
            s << "    if (!__prog) return;\n";
            s << "    if (clBuildProgram(__prog, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) return;\n";
            s << "    __kern = clCreateKernel(__prog, \"" << g.kernelName << "\", &__err);\n";
            s << "    if (!__kern) return;\n";
            s << "  }\n";
            // Buffers + writes.
            for (const KernelParam& p : g.params) {
                if (p.kind == KernelParamKind::ARRAY) {
                    s << "  cl_mem __loomx_m_" << p.name
                      << " = clCreateBuffer(__ctx, CL_MEM_READ_WRITE, "
                      << "(size_t)(" << p.sizeExpr << "), NULL, &__err);\n";
                    s << "  if (!__loomx_m_" << p.name << ") return;\n";
                    if (p.mappedTo) {
                        s << "  if (clEnqueueWriteBuffer(__q, __loomx_m_" << p.name
                          << ", CL_TRUE, 0, (size_t)(" << p.sizeExpr << "), "
                          << p.name << ", 0, NULL, NULL) != CL_SUCCESS) return;\n";
                    }
                } else if (p.kind == KernelParamKind::REDUCTION) {
                    s << "  cl_mem __loomx_m_" << p.name
                      << " = clCreateBuffer(__ctx, CL_MEM_READ_WRITE, " << p.sizeExpr
                      << ", NULL, &__err);\n";
                    s << "  if (!__loomx_m_" << p.name << ") return;\n";
                    s << "  if (clEnqueueWriteBuffer(__q, __loomx_m_" << p.name
                      << ", CL_TRUE, 0, " << p.sizeExpr << ", " << p.name
                      << ", 0, NULL, NULL) != CL_SUCCESS) return;\n";
                }
            }
            s << "  long __tot = " << g.totalTripText << ";\n";
            s << "  if (__tot > 0) {\n";
            s << "    int __ai = 0;\n";
            for (const KernelParam& p : g.params) {
                if (p.kind == KernelParamKind::ARRAY) {
                    s << "    clSetKernelArg(__kern, __ai++, sizeof(cl_mem), &__loomx_m_"
                      << p.name << ");\n";
                } else if (p.kind == KernelParamKind::REDUCTION) {
                    s << "    clSetKernelArg(__kern, __ai++, sizeof(cl_mem), &__loomx_m_"
                      << p.name << ");\n";
                } else {
                    s << "    clSetKernelArg(__kern, __ai++, sizeof(" << p.typeName
                      << "), &" << p.name << ");\n";
                }
            }
            s << "    size_t __gs = (size_t)__tot;\n";
            s << "    if (clEnqueueNDRangeKernel(__q, __kern, 1, NULL, &__gs, NULL, 0, NULL, NULL) != CL_SUCCESS) return;\n";
            s << "    clFinish(__q);\n";
            s << "  }\n";
            // Reads back.
            for (const KernelParam& p : g.params) {
                if (p.kind == KernelParamKind::ARRAY && p.mappedFrom) {
                    s << "  if (clEnqueueReadBuffer(__q, __loomx_m_" << p.name
                      << ", CL_TRUE, 0, (size_t)(" << p.sizeExpr << "), " << p.name
                      << ", 0, NULL, NULL) != CL_SUCCESS) return;\n";
                } else if (p.kind == KernelParamKind::REDUCTION) {
                    s << "  if (clEnqueueReadBuffer(__q, __loomx_m_" << p.name
                      << ", CL_TRUE, 0, " << p.sizeExpr << ", " << p.name
                      << ", 0, NULL, NULL) != CL_SUCCESS) return;\n";
                }
            }
            for (const KernelParam& p : g.params) {
                if (p.kind == KernelParamKind::ARRAY ||
                    p.kind == KernelParamKind::REDUCTION) {
                    s << "  clReleaseMemObject(__loomx_m_" << p.name << ");\n";
                }
            }
            s << "}\n";
            definitions += s.str() + "\n";
        }
    }

    source += "\n" + definitions;

    // Insert the launcher prototypes just below the header include (not before
    // it — prototypes must come after the device-runtime include).
    size_t headerEnd = source.find(header);
    if (headerEnd != std::string::npos) {
        headerEnd += header.size();
        source.insert(headerEnd, prototypes + "\n");
    } else {
        source = prototypes + "\n" + source;
    }
}

// Host launcher parameter list for an OpenCL stub.
std::string KernelCodeGen::hostLaunchSignature(const GeneratedKernel& g) const {
    std::vector<std::string> hp;
    for (const KernelParam& p : g.params) {
        if (p.kind == KernelParamKind::ARRAY) hp.push_back("void* " + p.name);
        else if (p.kind == KernelParamKind::REDUCTION) hp.push_back(p.typeName + "* " + p.name);
        else hp.push_back(p.typeName + " " + p.name);
    }
    std::string s;
    for (size_t i = 0; i < hp.size(); ++i) {
        if (i) s += ", ";
        s += hp[i];
    }
    return s;
}