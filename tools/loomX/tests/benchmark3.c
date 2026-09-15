// Benchmark without stdio to avoid __float128 issue
// Return sum as exit code for verification

double compute_element(double x, int idx) {
    double result = x * 2.0 + (double)idx * 0.01;
    return result * result - result + 1.0;
}

int main() {
    int N = 100000;
    double input[100000];
    double output[100000];
    int i;

    for (i = 0; i < N; i++) {
        input[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        output[i] = compute_element(input[i], i);
    }

    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum = sum + output[i];
    }

    // Return truncated sum as exit code for verification
    return (int)(sum / 100000.0);
}
