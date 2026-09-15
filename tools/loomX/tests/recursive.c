// Test: recursive function detection
// factorial is recursive and therefore conservatively marked unsafe.
// The loop calling factorial should stay sequential.

#define N 100000

double data[N];

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        data[i] = (double)factorial(i % 10);
    }

    return 0;
}
