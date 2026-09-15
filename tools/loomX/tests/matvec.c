// Matrix-vector multiplication benchmark for loomX
// No standard headers

#define M 4096
#define N 4096

double A[M][N];
double x[N];
double y[M];

int main() {
    int i, j;
    double sum = 0.0;

    // Initialize
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) / (double)(M + N);
        }
    }
    for (j = 0; j < N; j++) {
        x[j] = (double)j / (double)N;
    }

    // Matrix-vector multiply: y = A * x
    for (i = 0; i < M; i++) {
        y[i] = 0.0;
        for (j = 0; j < N; j++) {
            y[i] = y[i] + A[i][j] * x[j];
        }
    }

    // Reduction for checksum
    for (i = 0; i < M; i++) {
        sum = sum + y[i];
    }

    return (int)(sum / (double)M);
}
