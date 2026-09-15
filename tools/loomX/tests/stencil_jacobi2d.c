// 2D Jacobi stencil benchmark for loomX
// No standard headers to avoid __float128 parsing issues

#define N 512
#define ITERATIONS 100

double A[N][N];
double B[N][N];

int main() {
    int i, j, t;
    double sum = 0.0;

    // Initialize
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) / (double)(2 * N);
        }
    }

    // Jacobi iterations: each timestep is a 5-point stencil
    for (t = 0; t < ITERATIONS; t++) {
        for (i = 1; i < N - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
            }
        }
        // Swap A and B pointers conceptually; here we copy back
        for (i = 1; i < N - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                A[i][j] = B[i][j];
            }
        }
    }

    // Reduction for checksum
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = sum + A[i][j];
        }
    }

    return (int)(sum / (double)(N * N));
}
