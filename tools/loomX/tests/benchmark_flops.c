// Compute-intensive benchmark v2: heap-allocated, anti-optimization
// Uses a 2D array pattern to prevent vectorization
// Heavy per-element arithmetic to amortize GPU transfer cost

double compute_element(double x, int idx) {
    double a = x;
    double b = (double)idx * 0.0000001;
    int j;
    // Unrolled accumulation to increase FLOPs
    for (j = 0; j < 50; j++) {
        a = a * 1.0001 + b;
        b = b * 0.9999 + a * 0.0001;
    }
    return a + b;
}

int main() {
    int N = 2000000;
    // Use a 2D flat array to simulate more realistic memory access
    double data[2000000];
    double result[2000000];
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i * 0.000001;
    }

    for (i = 0; i < N; i++) {
        result[i] = compute_element(data[i], i);
    }

    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum = sum + result[i];
    }

    return (int)(sum / 1000000.0);
}
