/* global_writer.c: hot loop calls a function that writes a shared global.
 * Expected: rejected as unsafe because of the loop-carried dependence.
 */
#include "bench_minimal.h"

static double global_state = 0.0;

static void update_global(double x) {
    global_state += x;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double* a = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = (double)(i % 100);
    }

    for (int i = 0; i < n; i++) {
        update_global(a[i]);
    }

    printf("global_state: %.6f\n", global_state);
    free(a);
    return 0;
}
