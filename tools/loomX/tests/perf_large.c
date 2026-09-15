// Large-scale benchmark for loomX performance measurement
// Uses global arrays to avoid stack overflow and stdlib headers
// No standard headers to avoid __float128 parsing issues with ROSE

#define N 10000000

double input[N];
double output[N];

int classify_value(double val) {
    if (val > 0.5) return 1;
    if (val < -0.5) return -1;
    return 0;
}

double compute_element(double x, int idx) {
    double result = x * 2.0 + (double)idx * 0.01;
    return result * result - result + 1.0;
}

int main() {
    int i;
    double sum = 0.0;

    // Initialize
    for (i = 0; i < N; i++) {
        input[i] = (double)i / (double)N;
    }

    // Main compute loop - should go to GPU
    for (i = 0; i < N; i++) {
        output[i] = compute_element(input[i], i);
    }

    // Reduction loop - should go to GPU with reduction clause
    for (i = 0; i < N; i++) {
        sum = sum + output[i];
    }

    // Return a checksum
    return (int)(sum / (double)N);
}
