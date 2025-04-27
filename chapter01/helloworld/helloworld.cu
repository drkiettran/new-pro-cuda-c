#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */

__global__ void checkIndex(int* A, int n)
{
    // printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    // printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    // printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
        "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            printf("A[%d] = %d\n", i, A[i]);
        }
    }
}

int main(int argc, char** argv)
{
    // define total data element
    int nElem = 6;
    if (argc > 1) {
        nElem = atoi(argv[1]);
    }

    int* h_A = (int*)malloc(nElem * sizeof(int));
    int* d_A = NULL;

    for (int i = 0; i < nElem; i++) {
        h_A[i] = i;
    }

    int cudaStatus = cudaMalloc((void**)&d_A, nElem * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(-1);
    }

    cudaStatus = cudaMemcpy(d_A, h_A, nElem * sizeof(int), cudaMemcpyHostToDevice);

    // define grid and block structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex <<<grid, block >>> (d_A, nElem);

    // reset device before you leave
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        exit(-2);
    }

    if (h_A != NULL) free(h_A);
    if (d_A != NULL) cudaFree(d_A);

    return cudaStatus;
}
