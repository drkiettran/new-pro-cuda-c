#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

// A normal C function that is executed as a thread
// when its name is specified in pthread_create()
//
// to compile & run:
// gcc -pthread -g -o mthreads mthreads.c
// ./mthreads
//
void *myThreadFun(void *vargp)
{
        sleep(1);
        printf("Printing GeeksQuiz from Thread \n");
        return NULL;
}

int main()
{
        pthread_t thread_id;
        printf("Before Thread\n");
        pthread_create(&thread_id, NULL, myThreadFun, NULL);
        pthread_join(thread_id, NULL);
        printf("After Thread\n");

        exit(0);
}