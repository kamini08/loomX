#pragma once
#include "rose.h"
#include <cstddef>

namespace loomX {

// Inline helper functions whose only side effect is a reduction through a
// pointer parameter.  The canonical shape is:
//
//   void accum(double x, double* sum) {
//       *sum += x * x;
//   }
//
// called as:
//
//   for (i = 0; i < n; i++) { accum(a[i], &sum); }
//
// After inlining the loop body becomes:
//
//   for (i = 0; i < n; i++) { sum += a[i] * a[i]; }
//
// which the existing reduction detector and GPU codegen already handle.
//
// Returns the number of call sites transformed.
std::size_t inlineReductionHelpers(SgProject* project);

} // namespace loomX
