#include "common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

int main(int argc, char** argv)
{
    const char* hello_msg = "Hello World from CPU!\n";
    printf("%s", hello_msg);

    helloFromGPU << <1, 10 >> > ();
    CHECK(cudaDeviceReset());
    return 0;
}


