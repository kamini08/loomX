#pragma once
#include "rose.h"
#include "LoopSummary.h"
#include <set>
#include <string>
#include <vector>

namespace loomX {

// One data-mapping entry for a parallel region, with the extent already
// resolved into the rendered name.
struct MappedEntry {
    SgInitializedName* var = nullptr;
    // Rendered reference. For storage that needs an explicit extent this
    // carries the array section (e.g. "a[0:n]"); for genuine scalars it is the
    // bare name.
    std::string name;
    // "to", "from" or "tofrom" in the summary's own vocabulary. Backends
    // translate to their own keywords (OpenMP map(...), OpenACC copy(...)).
    std::string direction;
};

// Render-ready view of a LoopSummary, shared by the OpenMP and OpenACC
// backends.
//
// This exists for two reasons.
//
// First, the clause-building helpers (var list, reduction clause, mapped
// variable name) were duplicated verbatim between OpenMPCodeGen and
// OpenACCCodeGen, so a fix applied to one dialect silently did not reach the
// other. They are rendered here once.
//
// Second, and more importantly, resolving *what storage a mapped variable
// refers to* is a correctness question, not a formatting one. In a `target`
// region a pointer passed as a bare name transfers the pointer value, not the
// allocation; the device then dereferences a host address and libomptarget
// aborts with "CUDA error: an illegal memory access was encountered". Real
// array types get correct sections for free, which is why array-parameter
// kernels work and malloc'd pointer kernels do not. That resolution therefore
// belongs in one place, and a backend that cannot express an extent must be
// told to leave the loop alone rather than emit a broken map clause.
class ParallelForPlan {
public:
    explicit ParallelForPlan(const LoopSummary& summary);

    // Comma-separated list of private variables.
    std::string privateList() const;

    // reduction(...) clause text, or empty when there are no reductions. The
    // wording is shared verbatim by both dialects.
    std::string reductionClause() const;

    // Data-mapping entries with extents resolved.
    const std::vector<MappedEntry>& mappedEntries() const { return entries_; }

    // Entries with the given var promoted to `tofrom`, as a backend that needs
    // reduction variables in its data clause requires. Returned by value
    // because this is a dialect-specific policy, not a property of the loop.
    std::vector<MappedEntry> entriesIncludingReductions() const;

    // False when some mapped variable's storage could not be given an
    // explicit extent. A backend must not emit a target region in that case:
    // the only alternative map clause transfers a host pointer to the device.
    // The loop has to stay sequential instead.
    bool storageIsFullyMapped() const { return unmappable_.empty(); }

    // Human-readable names of the variables that blocked offload, for logging.
    const std::vector<std::string>& unmappable() const { return unmappable_; }

    // Render the same var list / reduction clause for an arbitrary collection.
    // Exposed because the OpenACC routine-seq path and the OpenMP post-
    // processing pass also need them.
    static std::string renderVarList(const std::set<SgInitializedName*>& vars);
    static std::string renderReductionClause(
        const std::vector<ReductionInfo>& reductions);

private:
    // Render `var` for use in a data clause. Returns false (and leaves `out`
    // untouched) when the variable is element-like storage whose extent cannot
    // be established, which means the caller must not offload.
    bool resolveMappedName(SgInitializedName* var, std::string& out) const;

    const LoopSummary& summary_;
    std::vector<MappedEntry> entries_;
    std::vector<std::string> unmappable_;
};

} // namespace loomX
