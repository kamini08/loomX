// Minimal benchmark without standard headers to avoid __float128 issue

double compute_element(double x, int idx) {
    double result = x * 2.0 + (double)idx * 0.01;
    return result * result - result + 1.0;
}

int classify_value(double val) {
    if (val > 0.5) return 1;
    if (val < -0.5) return -1;
    return 0;
}

int main() {
    int N = 100000;
    double input[100000];
    double output[100000];
    int classes[100];
    int i;

    // Initialize
    for (i = 0; i < N; i++) {
        input[i] = (double)i / (double)N;
    }

    // Loop 1: GPU-suitable - large N, regular access, compute-intensive, leaf function
    for (i = 0; i < N; i++) {
        output[i] = compute_element(input[i], i);
    }

    // Loop 2: CPU-suitable - small N, not worth GPU overhead
    for (i = 0; i < 100; i++) {
        classes[i] = classify_value(output[i]);
    }

    // Loop 3: GPU-suitable - reduction pattern
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum = sum + output[i];
    }

    return (int)sum;
}
