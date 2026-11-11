// A C program for a shem image, compiled for RV32 against musl.
#include <stdio.h>

int main(int argc, char **argv) {
    printf("hello");
    for (int i = 1; i < argc; i++)
        printf(" %s", argv[i]);
    printf("\n");
    return 0;
}
