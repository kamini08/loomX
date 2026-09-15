// Heavy compute benchmark for loomX GPU measurement
// 5M elements, inline heavy ops, no inner loops

#define N 5000000

double input[N];
double output[N];

double heavy_compute(double x, int idx) {
    double a = x;
    double b = (double)idx * 0.0001;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    a = a * a + b; b = b * 0.5 + a * 0.3; a = a - b * 0.1; b = b + a * 0.2;
    return a + b;
}

int main() {
    int i;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        input[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        output[i] = heavy_compute(input[i], i);
    }

    for (i = 0; i < N; i++) {
        sum = sum + output[i];
    }

    return (int)(sum / (double)N);
}
