// Numerical validation: histogram with per-iteration private bins.
// This is the hard case for automatic parallelization: the "reduction" is
// over an array, so it needs an array of thread-private copies plus a
// serial combine, not a scalar reduction clause. A backend that emits a
// scalar reduction here, or omits the privatization, loses counts.
// Expect loomX to decline this loop -- declining is correct; emitting
// anything else is a bug, and this checksum is what shows the difference.
#include <stdio.h>

#define N 1000003
#define NBINS 16
#define NOUTER 64

static int histogram[NBINS];

int main(void) {
    int outer, i;
    long long checksum = 0;

    for (i = 0; i < NBINS; i++) {
        histogram[i] = 0;
    }

    for (outer = 0; outer < NOUTER; outer++) {
        for (i = 0; i < N / NOUTER; i++) {
            long idx = (long)outer * (N / NOUTER) + i;
            int bin = (int)(idx % NBINS);
            histogram[bin] = histogram[bin] + 1;
        }
    }

    for (i = 0; i < NBINS; i++) {
        checksum = checksum + (long long)histogram[i] * (i + 1);
    }

    printf("hist_checksum %lld\n", checksum);
    for (i = 0; i < NBINS; i++) {
        printf("hist_bin%d %d\n", i, histogram[i]);
    }
    return 0;
}
