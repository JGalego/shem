// A guest that keeps a running total of the numbers it reads, one a line, and
// prints it after each. The total lives in its memory, so an image that
// `docker commit` made of a running container of it goes on from that total.
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    char line[64];
    long long total = 0;
    while (fgets(line, sizeof line, stdin)) {
        total += atoll(line);
        printf("total %lld\n", total);
        fflush(stdout);
    }
    return 0;
}
