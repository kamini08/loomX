/* Minimal forward declarations so micro-benchmarks can avoid system headers.
 * ROSE's Clang frontend has trouble parsing modern glibc/clang headers that
 * expose __float128; these declarations are enough to compile the harness
 * programs with clang and libc.
 */
#ifndef BENCH_MINIMAL_H
#define BENCH_MINIMAL_H

#ifndef NULL
#define NULL ((void*)0)
#endif

typedef __builtin_va_list va_list;

extern int printf(const char* format, ...);
extern int sprintf(char* str, const char* format, ...);
#ifndef __SIZE_TYPE__
#define __SIZE_TYPE__ unsigned long
#endif
typedef __SIZE_TYPE__ size_t;
extern void* malloc(size_t size);
extern void free(void* ptr);
extern int atoi(const char* str);
extern double atof(const char* str);

#endif
