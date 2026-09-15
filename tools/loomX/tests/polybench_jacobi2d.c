// Polybench jacobi-2d-imper benchmark for loomX
// 2D Jacobi stencil, imperative version
// No standard headers to avoid __float128 parsing issues

#define N 1024
#define TSTEPS 100

static double A[N][N];
static double B[N][N];

int main() {
    int t, i, j;
    double sum = 0.0;

    // Initialize
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = ((double)(i + j)) / (double)N;
            B[i][j] = ((double)(i + j)) / (double)N;
        }
    }

    // Jacobi iterations
    for (t = 0; t < TSTEPS; t++) {
        for (i = 1; i < N - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            }
        }
        for (i = 1; i < N - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                A[i][j] = B[i][j];
            }
        }
    }

    // Checksum for validation
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = sum + A[i][j];
        }
    }

    return (int)(sum / 1000000.0);
}
