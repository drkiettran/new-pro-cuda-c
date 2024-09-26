#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"


/*
 * simpleDivergence demonstrates divergent code on the GPU and its impact on
 * performance and CUDA metrics.
 */

__global__ void mathKernel1(float* c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel2(float* c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    // if ((tid >> 5) % 2 == 0)
    // if (((tid >> 5) & 0x01) == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel3(float* c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        ia = 100.0f;
    }

    if (!ipred)
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel4(float* c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void warmingup(float* c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}


int main(int argc, char** argv)
{
    std::chrono::steady_clock::time_point begin;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 64;
    int blocksize = 64;

    if (argc > 1) blocksize = atoi(argv[1]);

    if (argc > 2) size = atoi(argv[2]);

    printf("Data size %d ", size);

    // set up execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float* d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // run a warmup kernel to remove overhead
    CHECK(cudaDeviceSynchronize());

    begin = StartTimer();
    warmingup << <grid, block >> > (d_C);
    CHECK(cudaDeviceSynchronize());
    std::cout << "- grid.x: " << grid.x << std::endl << "- block.x: " << block.x << std::endl;
    std::cout << "Warming up on GPU: " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;

    CHECK(cudaGetLastError());

    // run kernel 1
    begin = StartTimer();
    mathKernel1 << <grid, block >> > (d_C);
    std::cout << "Running mathKernel1 on GPU: " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // run kernel 2
    begin = StartTimer();
    mathKernel2 << <grid, block >> > (d_C);
    std::cout << "Running mathKernel2 on GPU: " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // run kernel 3
    begin = StartTimer();
    mathKernel3 << <grid, block >> > (d_C);
    std::cout << "Running mathKernel3 on GPU: " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;
  
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // run kernel 4
    begin = StartTimer();
    mathKernel4 << <grid, block >> > (d_C);
    std::cout << "Running mathKernel4 on GPU: " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // free gpu memory and reset divece
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}