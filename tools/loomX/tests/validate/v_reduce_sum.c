// Numerical validation: floating-point sum reduction.
// Exercises: reduction(+:) clause emission, and FP reassociation tolerance
// (a correct parallel reduction reorders additions, so the result differs in
// the last bits -- that must be tolerated, not flagged).
#include <stdio.h>

#define N 1000003

int main(void) {
    double total = 0.0;
    long i;

    for (i = 0; i < N; i++) {
        total = total + (double)(i % 17) * 0.5;
    }

    printf("sum_reduce %.17g\n", total);
    return 0;
}
