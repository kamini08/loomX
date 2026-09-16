// Test: loop-local temporaries must not be flagged as reductions.
// Each iteration declares and updates its own accumulator; the variable is
// private, not a reduction across iterations.

#define N 1000

int main() {
    int i;
    int result[N];

    for (i = 0; i < N; i++) {
        int local_acc = 0;
        local_acc += i;
        local_acc++;
        result[i] = local_acc;
    }

    return 0;
}
