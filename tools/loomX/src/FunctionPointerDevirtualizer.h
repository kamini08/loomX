#pragma once
#include "rose.h"
#include <cstddef>

namespace loomX {

// Replace loop-invariant function-pointer calls with conditional direct calls
// when the pointer is assigned from a small conditional expression.  This turns
//
//   unary_op_t op = (n % 2 == 0) ? scale : offset;
//   for (...) { b[i] = op(a[i]); }
//
// into
//
//   for (...) { b[i] = (n % 2 == 0) ? scale(a[i]) : offset(a[i]); }
//
// which the existing offload analysis can then parallelize and inline.
//
// Returns the number of call sites transformed.
std::size_t devirtualizeFunctionPointers(SgProject* project);

} // namespace loomX
