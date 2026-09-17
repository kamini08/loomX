/* local_array.c: callee uses only its own local array.
 * Expected: parallelized because there are no cross-iteration side effects.
 */
#include "bench_minimal.h"

#define K 8

static double convolve_local(double x) {
    double kernel[K] = {0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05};
    double acc = 0.0;
    for (int j = 0; j < K; j++) {
        acc += kernel[j] * x;
    }
    return acc;
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
        b[i] = convolve_local(a[i]);
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
