#include "ParallelForPlan.h"
#include "LoopAnalysisUtil.h"
#include <algorithm>
#include <sstream>

namespace loomX {

namespace {

// Strip to the underlying type for classification.
static SgType* strip(SgType* type) {
    if (!type) return nullptr;
    return type->stripType(SgType::STRIP_MODIFIER_TYPE |
                           SgType::STRIP_TYPEDEF_TYPE);
}

// Follow casts and nested subscripts down to the variable an array access
// ultimately reads through: for `a[i]`, `a[i][j]`, `(double*)p` this yields the
// `a` / `p` declaration.
static SgInitializedName* rootVariableOf(SgExpression* expr) {
    expr = skipCasts(expr);
    if (!expr) return nullptr;
    if (SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(expr)) {
        return rootVariableOf(arrRef->get_lhs_operand());
    }
    if (SgVarRefExp* varRef = isSgVarRefExp(expr)) {
        return varRef->get_symbol()->get_declaration();
    }
    if (SgUnaryOp* un = isSgUnaryOp(expr)) {
        return rootVariableOf(un->get_operand());
    }
    return nullptr;
}

// True when `expr` is exactly the canonical induction variable, e.g. `i`.
static bool isExactlyIndexVar(SgExpression* expr, SgInitializedName* indexVar) {
    if (!expr || !indexVar) return false;
    if (SgVarRefExp* varRef = isSgVarRefExp(skipCasts(expr))) {
        return varRef->get_symbol()->get_declaration() == indexVar;
    }
    return false;
}

// Index expressions used to subscript `var` anywhere inside `scope`. Every
// SgPntrArrRefExp whose root variable is `var` contributes its index, so a
// two-dimensional access `a[i][j]` contributes both.
static void collectSubscriptIndices(SgNode* scope, SgInitializedName* var,
                                    std::vector<SgExpression*>& out) {
    if (!scope || !var) return;
    Rose_STL_Container<SgNode*> nodes =
        NodeQuery::querySubTree(scope, V_SgPntrArrRefExp);
    for (SgNode* node : nodes) {
        SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(node);
        if (!arrRef) continue;
        if (rootVariableOf(arrRef->get_lhs_operand()) == var) {
            out.push_back(arrRef->get_rhs_operand());
        }
    }
}

} // namespace

std::string ParallelForPlan::renderVarList(
    const std::set<SgInitializedName*>& vars) {
    std::ostringstream oss;
    bool first = true;
    for (SgInitializedName* var : vars) {
        if (!var) continue;
        if (!first) oss << ", ";
        first = false;
        oss << var->get_name().getString();
    }
    return oss.str();
}

std::string ParallelForPlan::renderReductionClause(
    const std::vector<ReductionInfo>& reductionDetails) {
    // Group variables by operator string so `reduction(+:a, b)` comes out of
    // one clause rather than several.
    std::map<std::string, std::vector<std::string>> groups;
    for (const ReductionInfo& info : reductionDetails) {
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

bool ParallelForPlan::resolveMappedName(SgInitializedName* var,
                                        std::string& out) const {
    if (!var) return true;  // nothing to render; caller keeps its own default

    std::string name = var->get_name().getString();
    SgType* type = strip(var->get_type());
    if (!type) {
        out = name;
        return true;
    }

    // ---- Real array types -------------------------------------------------
    if (SgArrayType* arrType = isSgArrayType(type)) {
        // A parameter array is a bare pointer inside the callee, so it needs
        // explicit per-dimension sections; a local array of known size maps
        // correctly under its bare name. (See the note in the .h about
        // mis-sized whole-array sections tripping libomptarget's "explicit
        // extension not allowed" check, which is why locals stay bare.)
        if (isSgFunctionParameterList(var->get_parent()) == nullptr) {
            out = name;
            return true;
        }
        std::string section;
        while (arrType) {
            SgExpression* index = arrType->get_index();
            if (!index) {
                out = name;
                return true;
            }
            std::string dim = index->unparseToString();
            if (dim.empty()) {
                out = name;
                return true;
            }
            section += "[0:" + dim + "]";
            SgType* baseType = strip(arrType->get_base_type());
            arrType = isSgArrayType(baseType);
        }
        out = section.empty() ? name : name + section;
        return true;
    }

    // ---- Pointer types ----------------------------------------------------
    if (isSgPointerType(type)) {
        std::vector<SgExpression*> indices;
        if (summary_.loop && summary_.loop->get_loop_body()) {
            collectSubscriptIndices(summary_.loop->get_loop_body(), var, indices);
        }

        // Never subscripted in this loop: the pointer is being used as an
        // opaque value (linked-list cursor, FILE*, allocation handle). Mapping
        // the value itself is correct.
        if (indices.empty()) {
            out = name;
            return true;
        }

        // Element-like storage. Deriving the extent requires a canonical loop
        // running 0..ub with unit stride, and every subscript being exactly the
        // induction variable. Anything else (a shifted base such as &a[k], a
        // strided access, an inner index of a collapsed nest, a non-zero lower
        // bound) is reported as unresolvable so the caller refuses to offload
        // rather than transferring a host pointer to the device.
        const CanonicalResult& canon = summary_.canonical;
        if (canon.indexVar && canon.upperBound && canon.lowerBound &&
            canon.stride && canon.isUpperExclusive) {
            std::string lb = canon.lowerBound->unparseToString();
            std::string stride = canon.stride->unparseToString();
            bool unitStride = (stride == "1");
            bool allIndicesAreLoopVar =
                std::all_of(indices.begin(), indices.end(),
                            [&](SgExpression* e) {
                                return isExactlyIndexVar(e, canon.indexVar);
                            });
            if (lb == "0" && unitStride && allIndicesAreLoopVar) {
                std::string ub = canon.upperBound->unparseToString();
                if (!ub.empty()) {
                    out = name + "[0:" + ub + "]";
                    return true;
                }
            }
        }
        return false;
    }

    // ---- Scalars and everything else -------------------------------------
    out = name;
    return true;
}

ParallelForPlan::ParallelForPlan(const LoopSummary& summary)
    : summary_(summary) {
    for (const auto& [var, direction] : summary.mapClauses) {
        MappedEntry entry;
        entry.var = var;
        entry.direction = direction;
        std::string rendered;
        if (resolveMappedName(var, rendered)) {
            entry.name = rendered;
            entries_.push_back(entry);
        } else {
            unmappable_.push_back(var->get_name().getString());
        }
    }
}

std::string ParallelForPlan::privateList() const {
    return renderVarList(summary_.privateVars);
}

std::string ParallelForPlan::reductionClause() const {
    return renderReductionClause(summary_.reductions);
}

std::vector<MappedEntry> ParallelForPlan::entriesIncludingReductions() const {
    // Backends differ on whether reduction variables belong in the data clause:
    // OpenMP's target construct needs them present for the result to be written
    // back, while OpenACC's reduction() already implies the transfer. The
    // canonical summary keeps them out of mapClauses, so this is applied on
    // request by the dialect that wants it.
    std::vector<MappedEntry> merged = entries_;
    std::set<SgInitializedName*> reductions = summary_.getReductionVariables();
    if (reductions.empty()) return merged;

    for (SgInitializedName* redVar : reductions) {
        bool found = false;
        for (MappedEntry& entry : merged) {
            if (entry.var == redVar) {
                entry.direction = "tofrom";
                found = true;
                break;
            }
        }
        if (!found) {
            MappedEntry entry;
            entry.var = redVar;
            entry.direction = "tofrom";
            // A reduction scalar is a plain variable; it always renders as its
            // own name, so no extent resolution is needed.
            entry.name = redVar->get_name().getString();
            merged.push_back(entry);
        }
    }
    return merged;
}

} // namespace loomX
