// Test: subtraction reduction must use the '-' operator, not '+'.
#define N 1000

int main() {
    int i;
    int total = 1000000;

    for (i = 0; i < N; i++) {
        total = total - i;
    }

    return total;
}
