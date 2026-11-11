/* Multiply two whole numbers of any length exactly, as decimal digits:
 * what a language model cannot do reliably, and a CPU always does.
 * Plain write(2) instead of printf keeps it to a few thousand instructions. */
#include <string.h>
#include <unistd.h>

#define MAX_DIGITS 1000

static int fail(const char *message) {
    write(2, message, strlen(message));
    return 2;
}

static int whole(const char *s, size_t n) {
    if (n == 0 || n > MAX_DIGITS) return 0;
    for (size_t i = 0; i < n; i++)
        if (s[i] < '0' || s[i] > '9') return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) return fail("usage: mul A B\n");
    const char *a = argv[1], *b = argv[2];
    size_t n = strlen(a), m = strlen(b);
    if (!whole(a, n) || !whole(b, m)) return fail("mul: A and B must be whole numbers of up to 1000 digits\n");

    static unsigned char r[2 * MAX_DIGITS];
    static char out[2 * MAX_DIGITS + 1];
    memset(r, 0, n + m);
    for (size_t i = n; i-- > 0;) {
        unsigned carry = 0;
        for (size_t j = m; j-- > 0;) {
            unsigned t = r[i + j + 1] + (unsigned)(a[i] - '0') * (unsigned)(b[j] - '0') + carry;
            r[i + j + 1] = t % 10;
            carry = t / 10;
        }
        r[i] += carry;
    }
    size_t k = 0, len = 0;
    while (k + 1 < n + m && r[k] == 0) k++;
    for (; k < n + m; k++) out[len++] = (char)('0' + r[k]);
    out[len++] = '\n';
    write(1, out, len);
    return 0;
}
