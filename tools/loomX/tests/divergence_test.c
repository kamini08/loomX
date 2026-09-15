// Divergence test: loop has an if-statement inside
// Should be detected as divergent and potentially skipped or CPU-only

double process(double x) {
    return x * x + 1.0;
}

int main() {
    int N = 100000;
    double data[100000];
    double result[100000];
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        if (data[i] > 0.5) {
            result[i] = process(data[i]) * 2.0;
        } else {
            result[i] = process(data[i]) * 0.5;
        }
    }

    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum = sum + result[i];
    }

    return (int)(sum / 100000.0);
}
