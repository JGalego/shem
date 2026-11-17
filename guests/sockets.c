/* Sockets on shem: a server listens on a Unix socket and on a loopback port,
 * two children connect to them and talk, and the parent serves each, in the
 * order they come, then waits for the children. Prints what each said, and
 * exits 0 when all is as it should be.
 *   sockets */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#define PORT 7777
#define PATH "/tmp/shem.sock"

static int fail(const char *what) {
    perror(what);
    return 1;
}

static void client(int domain, const char *name) {
    int fd = socket(domain, SOCK_STREAM, 0);
    int status = -1;
    for (int tries = 0; tries < 1000 && status < 0; tries++) {
        if (domain == AF_UNIX) {
            struct sockaddr_un addr = {.sun_family = AF_UNIX};
            strcpy(addr.sun_path, PATH);
            status = connect(fd, (struct sockaddr *)&addr, sizeof addr);
        } else {
            struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(PORT)};
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            status = connect(fd, (struct sockaddr *)&addr, sizeof addr);
        }
        if (status < 0)
            sched_yield();
    }
    if (status < 0)
        _exit(2);
    char hello[64];
    int n = snprintf(hello, sizeof hello, "hello from %s", name);
    write(fd, hello, n);
    char reply[64];
    n = read(fd, reply, sizeof reply - 1);
    _exit(n > 0 && memcmp(reply, "ok ", 3) == 0 ? 0 : 3);
}

int main(void) {
    int unix_server = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un un = {.sun_family = AF_UNIX};
    strcpy(un.sun_path, PATH);
    if (bind(unix_server, (struct sockaddr *)&un, sizeof un) < 0 || listen(unix_server, 4) < 0)
        return fail("unix listen");

    int inet_server = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(inet_server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in in = {.sin_family = AF_INET, .sin_port = htons(PORT)};
    in.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(inet_server, (struct sockaddr *)&in, sizeof in) < 0 || listen(inet_server, 4) < 0)
        return fail("inet listen");

    if (fork() == 0)
        client(AF_UNIX, "unix");
    if (fork() == 0)
        client(AF_INET, "inet");

    for (int served = 0; served < 2; served++) {
        int server = served == 0 ? unix_server : inet_server;
        int fd = accept(server, NULL, NULL);
        if (fd < 0)
            return fail("accept");
        char said[64];
        int n = read(fd, said, sizeof said - 1);
        if (n <= 0)
            return fail("read");
        said[n] = 0;
        printf("%s\n", said);
        char reply[80];
        n = snprintf(reply, sizeof reply, "ok %s", said);
        write(fd, reply, n);
        close(fd);
    }
    int status, bad = 0;
    while (wait(&status) > 0)
        bad |= !WIFEXITED(status) || WEXITSTATUS(status) != 0;
    return bad;
}
