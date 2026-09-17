/* conditional_call.c: hot loop calls a helper only on even iterations.
 * Expected: parallelized if the call itself is safe.
 */
#include "bench_minimal.h"

static double even_op(double x) {
    return x * 0.5;
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
        if (i % 2 == 0) {
            b[i] = even_op(a[i]);
        } else {
            b[i] = a[i];
        }
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
