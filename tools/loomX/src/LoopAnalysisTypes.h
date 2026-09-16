#pragma once
#include "rose.h"
#include <string>

namespace loomX {

// Canonical form classification for a for-loop.
enum class CanonicalForm {
    UNKNOWN,        // Not analyzed yet
    CANONICAL,      // Recognized canonical for-loop
    NON_CANONICAL   // Does not match canonical form
};

// Reason a loop was rejected as non-canonical.
enum class NonCanonicalReason {
    NONE,
    NO_INIT,
    NO_TEST,
    NO_INCREMENT,
    NON_INTEGER_ITERATOR,
    NON_LITERAL_OR_PARAMETER_BOUND,
    NON_UNIT_STRIDE,
    NON_IDIOMATIC_INCREMENT,
    LOOP_VAR_MODIFIED_IN_BODY,
    MULTIPLE_ITERATORS
};

// Result of canonical-form analysis.
struct CanonicalResult {
    CanonicalForm form = CanonicalForm::UNKNOWN;
    NonCanonicalReason reason = NonCanonicalReason::NONE;
    SgInitializedName* indexVar = nullptr;
    SgExpression* lowerBound = nullptr;
    SgExpression* upperBound = nullptr;
    SgExpression* stride = nullptr;
    bool isLowerInclusive = true;   // Canonical ROSE form is usually i=lb; i<ub; i++
    bool isUpperExclusive = true;
    std::string note;               // Human-readable detail
};

// Reduction operator kinds detected by the reduction analyzer.
enum class ReductionOp {
    UNKNOWN,
    ADD,
    SUB,
    MUL,
    MIN,
    MAX,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    LOGICAL_AND,
    LOGICAL_OR
};

// Description of a single reduction variable.
struct ReductionInfo {
    SgInitializedName* variable = nullptr;
    ReductionOp op = ReductionOp::UNKNOWN;
    std::string opString;  // OpenMP reduction operator string
    SgStatement* statement = nullptr;  // Statement where reduction was detected
};

// Divergence classification.
enum class DivergenceKind {
    NONE,
    DATA_DEPENDENT_IF,      // if (A[i] > 0)
    DATA_DEPENDENT_SWITCH,  // switch (A[i])
    EARLY_EXIT,             // break/continue/return inside loop
    INNER_LOOP,             // nested loop causes thread divergence
    FUNCTION_CALL           // call that may have divergent internal control flow
};

// Result of divergence analysis.
struct DivergenceResult {
    bool isDivergent = false;
    DivergenceKind kind = DivergenceKind::NONE;
    SgNode* source = nullptr;
    std::string description;
};

// Compute-intensity classification.
enum class IntensityClass {
    UNKNOWN,
    MEMORY_BOUND,    // More memory ops than compute
    BALANCED,        // Comparable compute and memory
    COMPUTE_BOUND    // Compute dominates
};

// Result of compute-intensity estimation.
struct ComputeIntensityResult {
    IntensityClass classification = IntensityClass::UNKNOWN;
    double flopsPerMemoryOp = 0.0;  // FLOPs per memory access
    int flopCount = 0;              // Static estimate of FP operations
    int memoryOpCount = 0;          // Static estimate of memory accesses
    int integerOpCount = 0;         // Static estimate of integer operations
    bool hasHeavyMath = false;      // sin/cos/sqrt/exp/log/etc.
    std::string note;
};

} // namespace loomX
