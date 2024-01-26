
#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

/*
 * Demonstrate defining the dimensions of a block of threads and a grid of
 * blocks from the host.
 */

int main(int argc, char** argv)
{
    // define total data element
    int nElem = 1024;
    if (argc > 1) {
        nElem = atoi(argv[1]);
    }

    // define grid and block structure
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return(0);
}


