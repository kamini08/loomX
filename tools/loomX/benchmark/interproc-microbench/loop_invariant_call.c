/* loop_invariant_call.c: hot loop calls a pure function with loop-invariant args.
 * Expected: parallelized; the call could also be hoisted, but that is optional.
 */
#include "bench_minimal.h"

static double poly(double x) {
    return x * x * x - 2.0 * x * x + x + 1.0;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double x = (argc > 2) ? atof(argv[2]) : 1.5;
    double* b = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        b[i] = poly(x) + (double)i;
    }

    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        checksum += b[i];
    }
    printf("checksum: %.6f\n", checksum);
    free(b);
    return 0;
}
