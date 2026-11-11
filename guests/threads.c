/* Threads on shem: four threads add to shared totals, under a mutex and with
 * atomics, and count in thread-local storage; the main thread joins them and
 * checks every total. Exits 0 when they are all right.
 *   threads [ROUNDS]    each thread's rounds, 1000 by default */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS 4

static int rounds = 1000;

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static long locked_total;
static atomic_long atomic_total;
static _Thread_local int mine;

static void *work(void *arg) {
    long id = (long)arg;
    for (int i = 0; i < rounds; i++) {
        pthread_mutex_lock(&lock);
        locked_total += id;
        pthread_mutex_unlock(&lock);
        atomic_fetch_add(&atomic_total, 1);
        mine++;
    }
    return (void *)(long)mine;
}

int main(int argc, char **argv) {
    if (argc > 1) rounds = atoi(argv[1]);
    pthread_t threads[THREADS];
    for (long t = 0; t < THREADS; t++)
        if (pthread_create(&threads[t], NULL, work, (void *)(t + 1)) != 0) return 1;
    long tls_total = 0;
    for (int t = 0; t < THREADS; t++) {
        void *count;
        pthread_join(threads[t], &count);
        tls_total += (long)count;
    }
    printf("locked %ld atomic %ld thread-local %ld main %d\n", locked_total, (long)atomic_total, tls_total, mine);
    int right = locked_total == rounds * 10 && atomic_total == THREADS * rounds && tls_total == THREADS * rounds && mine == 0;
    return right ? 0 : 2;
}
