
#ifndef SKIP_ROSE_BUILTIN_DECLARATIONS

// Put any required C++ declarations here.

// GCC math/abs builtins used by modern libstdc++ headers. ROSE's Clang
// frontend does not predefine these, so declare them here so that standard
// C++ headers can be parsed. They are only used inside inline functions in
// system headers; the real host compiler provides the definitions at link
// time when the generated source is compiled.
extern "C" {
  long __builtin_labs(long);
  long long __builtin_llabs(long long);
  double __builtin_fabs(double);
  float __builtin_fabsf(float);
  long double __builtin_fabsl(long double);
}

#endif
