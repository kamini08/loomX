/* readonly_helper.c: hot loop calls a read-only helper.
 * The helper only reads memory and returns a value.
 * Expected: parallelized.
 */
#include "bench_minimal.h"

static double window_sum(const double* a, int i, int n) {
    double s = a[i];
    if (i > 0) s += a[i - 1];
    if (i + 1 < n) s += a[i + 1];
    return s;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double* a = (double*)malloc(sizeof(double) * n);
    double* b = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = (double)(i % 100);
        b[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        b[i] = window_sum(a, i, n);
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
