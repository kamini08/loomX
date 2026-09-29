#pragma once
#include "rose.h"
#include <cstddef>

namespace loomX {

// Inline helper functions whose only side effect is a reduction.  Two shapes
// are supported:
//
// 1. Reduction through a pointer parameter:
//      void accum(double x, double* sum) { *sum += x * x; }
//    called as:
//      for (i = 0; i < n; i++) { accum(a[i], &sum); }
//    becomes:
//      for (i = 0; i < n; i++) { sum += a[i] * a[i]; }
//
// 2. Reduction into a file-scope/global variable:
//      static double global_state = 0.0;
//      void update_global(double x) { global_state += x; }
//    called as:
//      for (i = 0; i < n; i++) { update_global(a[i]); }
//    becomes:
//      for (i = 0; i < n; i++) { global_state += a[i]; }
//
// In both cases the existing reduction detector and GPU codegen handle the
// resulting loop body.
//
// Returns the number of call sites transformed.
std::size_t inlineReductionHelpers(SgProject* project);

} // namespace loomX
