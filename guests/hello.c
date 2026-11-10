// A small C guest for shem: argv, stdio, a heap allocation large enough to need
// mmap, and 64-bit arithmetic, linked statically against musl.
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    for (int i = 0; i < argc; i++)
        printf("argv[%d] = %s\n", i, argv[i]);

    size_t n = 100000;
    unsigned long long *squares = malloc(n * sizeof *squares);
    if (!squares) {
        perror("malloc");
        return 1;
    }
    unsigned long long sum = 0;
    for (size_t i = 0; i < n; i++) {
        squares[i] = (unsigned long long)i * i;
        sum += squares[i];
    }
    free(squares);
    printf("sum of squares below %zu = %llu\n", n, sum);
    return argc;
}
