/* call_chain.c: hot loop calls a chain of small pure functions.
 * Expected: parallelized if interprocedural analysis follows the chain.
 */
#include "bench_minimal.h"

static double step_c(double x) {
    return x + 1.0;
}

static double step_b(double x) {
    return step_c(x * 2.0);
}

static double step_a(double x) {
    return step_b(x) - 1.0;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double* a = (double*)malloc(sizeof(double) * n);
    double* b = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = (double)i;
        b[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        b[i] = step_a(a[i]);
    }

    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        checksum += b[i];
    }
    printf("checksum: %.6f\n", checksum);
    free(a);
    free(b);
    return 0;
}
