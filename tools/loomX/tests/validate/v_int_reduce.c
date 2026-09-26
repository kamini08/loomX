// Numerical validation: integer reduction, expected bitwise-exact.
// Exercises: a reduction over int. Unlike the FP kernels, reassociating an
// integer sum is still exact, so this must match the sequential result
// BITWISE. It is the control case: if a backend perturbs even this, the
// failure is a real bug rather than tolerated FP drift.
#include <stdio.h>

#define N 1000003

int main(void) {
    long long total = 0;
    int i;

    for (i = 0; i < N; i++) {
        total = total + (long long)(i % 1000);
    }

    printf("int_reduce %lld\n", total);

    {
        long long prod = 1;
        for (i = 1; i <= 20; i++) {
            prod = prod * (long long)i;
        }
        printf("int_factorial %lld\n", prod);
    }
    return 0;
}
