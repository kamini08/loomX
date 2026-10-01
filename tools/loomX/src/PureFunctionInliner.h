#pragma once
#include "rose.h"
#include "InterproceduralAnalysis.h"
#include <cstddef>

namespace loomX {

// Inline small, side-effect-free helper functions at call sites inside loops.
// This turns loops like:
//
//   for (...) {
//       zeroReal3(s->atoms->f[ii]);
//       ...
//   }
//
// into:
//
//   for (...) {
//       s->atoms->f[ii][0] = 0;
//       s->atoms->f[ii][1] = 0;
//       s->atoms->f[ii][2] = 0;
//       ...
//   }
//
// so that the existing loop parallelizer can see the actual memory accesses.
//
// A candidate function must have a definition, no transitive side effects
// (I/O, global writes, unanalyzable pointer writes), no recursion, and an AST
// size below a threshold.  Only void functions called as entire statements are
// inlined for now.
//
// Returns the number of call sites inlined.
std::size_t inlinePureHelperFunctions(SgProject* project,
                                      const InterproceduralAnalysis& ipa);

} // namespace loomX
