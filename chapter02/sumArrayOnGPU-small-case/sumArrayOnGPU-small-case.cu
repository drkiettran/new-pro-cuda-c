
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include <tclap/CmdLine.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 */

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}


void initialData(float* ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}


void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N)
{
    int i = threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

void getArgs(int argc, char** argv, int& n) {
    try {
        TCLAP::CmdLine cmd("MyProgram - A sample C++ program", ' ', "1.0");

        TCLAP::ValueArg<int> nArg("n", "num-elements", "Number of data elements", false, 32, "int");
        cmd.add(nArg);
        cmd.parse(argc, argv);
        n = nArg.getValue();
    }
    catch (TCLAP::ArgException& e) {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    }
}

int main(int argc, char** argv)
{
    int n;

    printf("%s Starting...\n", argv[0]);
    std::chrono::steady_clock::time_point begin;
    getArgs(argc, argv, n);
    // set up device
    int dev = 0;
    int cudaStatus = cudaSetDevice(dev);

    // set up data size of vectors
    printf("Vector size %d\n", n);

    // malloc host memory
    size_t nBytes = n * sizeof(float);

    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, n);
    initialData(h_B, n);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    dim3 block(n);
    dim3 grid(1);


    begin = StartTimer();
    sumArraysOnGPU << <grid, block >> > (d_A, d_B, d_C, n);
    printf("Execution configure <<<%d, %d>>>\n", grid.x, block.x);
    std::cout << "Sum Arrays on GPU: " << GetDurationInMilliSeconds(begin, StopTimer()) << " ms" << std::endl;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, n);

    // check device results
    checkResult(hostRef, gpuRef, n);

    // free device global memory
    cudaStatus = cudaFree(d_A);
    cudaStatus = cudaFree(d_B);
    cudaStatus = cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaStatus = cudaDeviceReset();
    return(0);
}
