// Numerical validation: dense GEMM, C = alpha*A*B + beta*C.
// This is the flagship loop-nest shape: three perfectly nested loops over
// three distinct arrays with a reduction in the innermost loop. A collapse or
// privatization mistake here produces plausible-looking but wrong numbers,
// which is exactly what an exit-code check cannot see.
#include <stdio.h>

#define N 256

static double A[N][N];
static double B[N][N];
static double C[N][N];

int main(void) {
    const double alpha = 1.5;
    const double beta = 0.75;
    int i, j, k;
    double trace = 0.0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (double)((i * 7 + j * 3) % 23) * 0.25;
            B[i][j] = (double)((i * 5 + j * 11) % 19) * 0.5;
            C[i][j] = 0.0;
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double acc = 0.0;
            for (k = 0; k < N; k++) {
                acc = acc + alpha * A[i][k] * B[k][j];
            }
            C[i][j] = acc + beta * C[i][j];
        }
    }

    for (i = 0; i < N; i++) {
        trace = trace + C[i][i];
    }

    printf("gemm_trace %.17g\n", trace);
    printf("gemm_c00 %.17g\n", C[0][0]);
    printf("gemm_cNN %.17g\n", C[N - 1][N - 1]);
    return 0;
}
