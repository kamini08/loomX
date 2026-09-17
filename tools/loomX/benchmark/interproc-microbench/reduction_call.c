/* reduction_call.c: hot loop calls a helper that accumulates into a scalar.
 * Expected: recognized as a reduction and parallelized.
 */
#include "bench_minimal.h"

static void accum(double x, double* sum) {
    *sum += x * x;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double* a = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = (double)(i % 100);
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        accum(a[i], &sum);
    }

    printf("sum: %.6f\n", sum);
    free(a);
    return 0;
}
