/* leaf_call.c: hot loop calls a pure leaf function.
 * Expected: parallelized (CPU/GPU depending on profitability).
 */
#include "bench_minimal.h"

static double scale(double x) {
    return x * 2.0 + 1.0;
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
        b[i] = scale(a[i]);
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
