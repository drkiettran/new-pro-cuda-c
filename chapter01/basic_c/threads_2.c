#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

//
// to compile & run:
// gcc -pthread -g -o mthreads_v2 mthreads_v2.c
// ./mthreads_v2
//
struct thread_info {
    int count;
    const char* name;
    struct timespec *ts;
};

void *thread_1(void *vargp) {
    struct thread_info *ti = (struct thread_info*)vargp;

    for (int i = 0; i < ti->count; i++) {
        nanosleep(ti->ts, NULL);
        printf("%s: %d.awake ...\n", ti->name, i);
    }

    return NULL;
}

int main()
{
    struct timespec ts = {0, 800000000};

    struct thread_info ti_1 = {8, "thread_1 #1", &ts};
    struct thread_info ti_2 = {10, "thread_1 #2", &ts};
    struct thread_info ti_3 = {12, "thread_1 #3", &ts};

    pthread_t thread_id_1, thread_id_2, thread_id_3;
    printf("Main: Before Thread\n");
    pthread_create(&thread_id_1, NULL, thread_1, &ti_1);
    pthread_create(&thread_id_2, NULL, thread_1, &ti_2);
    pthread_create(&thread_id_3, NULL, thread_1, &ti_3);

    for (int i = 0; i < 6; i++) {
        nanosleep(&ts, NULL);
        printf("    Main: %d.awake ...\n", i);
    }

    pthread_join(thread_id_1, NULL);
    pthread_join(thread_id_2, NULL);
    pthread_join(thread_id_3, NULL);
    printf("Main: After Thread\n");
    exit(0);
}