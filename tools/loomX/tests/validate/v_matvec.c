// Numerical validation: matrix-vector product.
// Exercises: outer loop parallelized, inner reduction privatized, and the
// dependent accumulation y[i] += ... kept correct. The y[i] = 0.0 store
// followed by an accumulation in the same iteration is the pattern most
// likely to be broken by a missing private() or a bad collapse.
#include <stdio.h>

#define M 2048
#define N 2048

static double A[M][N];
static double x[N];
static double y[M];

int main(void) {
    int i, j;
    double sum = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) / (double)(M + N);
        }
    }
    for (j = 0; j < N; j++) {
        x[j] = (double)j / (double)N;
    }

    for (i = 0; i < M; i++) {
        double acc = 0.0;
        for (j = 0; j < N; j++) {
            acc = acc + A[i][j] * x[j];
        }
        y[i] = acc;
    }

    for (i = 0; i < M; i++) {
        sum = sum + y[i];
    }

    printf("matvec_sum %.17g\n", sum);
    printf("matvec_y0 %.17g\n", y[0]);
    printf("matvec_yM1 %.17g\n", y[M - 1]);
    return 0;
}
