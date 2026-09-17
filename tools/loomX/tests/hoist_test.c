// Simple test for target data hoisting.
#define N 100000

int A[N];
int B[N];
int C[N];

int main() {
    int i;

    for (i = 0; i < N; i++) {
        double x = (double)i;
        A[i] = (int)(x * x + x * x * x + x / (x + 1.0) + x * x + x);
    }

    for (i = 0; i < N; i++) {
        double x = (double)i;
        B[i] = (int)(x * x + x * x * x + x / (x + 1.0) + x * x + x);
    }

    for (i = 0; i < N; i++) {
        double x = (double)A[i];
        double y = (double)B[i];
        C[i] = (int)(x * x + y * y + x * y + x / (y + 1.0) + x * y * y + x * x * y + x * x * x + y * y * y + x * x * y * y);
    }

    return C[0];
}
