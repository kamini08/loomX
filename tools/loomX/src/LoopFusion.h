#pragma once
#include <string>

namespace loomX {

// Text-based fusion of adjacent offloaded loops.
//
// The OpenMP backend hoists consecutive offload loops into one shared
// "target data" region and strips the per-loop map clauses, so by the time
// this runs a GPU loop is a bare
//     #pragma omp target teams distribute parallel for
//     for (...) { ... }
// pair. Two such pairs that share an iteration space and where the first is a
// pure per-index producer of exactly the data the second consumes at the same
// index can therefore be merged textually, with no map-clause rewriting and no
// AST mutation.
//
// The merge is deliberately conservative. In particular it refuses a stencil
// such as b[i] = a[i-1] + a[i] + a[i+1], where fusing the fill with the stencil
// would race the fill against its neighbours, and it refuses to pass a produced
// array to a callee, whose purity loomX cannot re-verify at this level. Such
// kernels keep two loops.
//
// Returns the rewritten source; returns `source` unchanged when nothing fuses.
std::string fuseAdjacentOffloadLoops(const std::string& source);

}  // namespace loomX
