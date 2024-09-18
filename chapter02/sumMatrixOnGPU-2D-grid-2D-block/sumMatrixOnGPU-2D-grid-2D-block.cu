﻿#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include <tclap/CmdLine.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. A 2D thread block and 2D grid are used. sumArraysOnHost sequentially
 * iterates through vector elements on the host.
 */

void initialData(float* ip, const int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
        // ip[i] = (float)i;
    }

    return;
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx,
    const int ny)
{
    float* ia = A;
    float* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}


void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float* MatA, float* MatB, float* MatC, int nx,
    int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];

    //printf("thread_id (%02d,%02d) block_id (%02d,%02d) coordinate (%02d,%02d) global index"
    //    " %d ival %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
    //    ix, iy, idx, (int)MatA[idx]);
}

void getArgs(int argc, char** argv, int& x, int& y) {
    try {
        TCLAP::CmdLine cmd("MyProgram - A sample C++ program", ' ', "1.0");

        TCLAP::ValueArg<int> xArg("x", "power-of-x", "Number of columns: 2 to the x", false, 14, "int");
        TCLAP::ValueArg<int> yArg("y", "power-of-y", "Number of rows: 2 to the y", false, 14, "int");
        cmd.add(xArg);
        cmd.add(yArg);
        cmd.parse(argc, argv);
        x = xArg.getValue();
        y = yArg.getValue();
    }
    catch (TCLAP::ArgException& e) {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    }
}


int main(int argc, char** argv)
{
    int x, y;
    printf("%s Starting...\n", argv[0]);
    std::chrono::steady_clock::time_point begin;
    getArgs(argc, argv, x, y);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << x;
    int ny = 1 << y;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    begin = StartTimer();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    std::cout << "Initialize Matrices on Host: " << GetDurationInMilliSeconds(begin, StopTimer()) << " ms" << std::endl;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    begin = StartTimer();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    std::cout << "Sum Matrices on Host: " << GetDurationInMilliSeconds(begin, StopTimer()) << " ms" << std::endl;

    // malloc device global memory
    float* d_MatA, * d_MatB, * d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 512;
    int dimy = 1024/dimx; // max threads per block = 1,024!
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    begin = StartTimer();
    sumMatrixOnGPU2D << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
    std::cout << "Sum Matrices on GPU: " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
