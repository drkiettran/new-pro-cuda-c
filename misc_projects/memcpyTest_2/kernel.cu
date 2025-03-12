#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

const int MAXDATASIZE = 1024 * 1024;

// Kernel for accessing mapped memory
__global__ void accessMappedMemoryKernel(int* mappedData, int dataSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < dataSize) {
        mappedData[idx] += 1;
    }
}

void testMemoryCopySpeed(int iter, int step, const char* filename) {
    cudaStream_t str;
    int* h_data_pageable, * h_data_pinned, * h_data_mapped;
    int* d_data;
    int i, dataSize;
    cudaEvent_t startT, endT;
    float duration;

    // Allocate pageable memory
    h_data_pageable = (int*)malloc(sizeof(int) * MAXDATASIZE);

    // Allocate pinned memory
    cudaMallocHost((void**)&h_data_pinned, sizeof(int) * MAXDATASIZE);

    // Allocate mapped memory
    cudaHostAlloc((void**)&h_data_mapped, sizeof(int) * MAXDATASIZE, cudaHostAllocMapped);

    // Allocate device memory
    cudaMalloc((void**)&d_data, sizeof(int) * MAXDATASIZE);

    // Initialize host data
    for (i = 0; i < MAXDATASIZE; i++) {
        h_data_pageable[i] = h_data_pinned[i] = h_data_mapped[i] = i;
    }

    // Create CUDA events and stream
    cudaEventCreate(&startT);
    cudaEventCreate(&endT);
    cudaStreamCreate(&str);

    std::ofstream outFile(filename);
    outFile << "DataSize(Bytes),Pageable_H2D(MB/s),Pageable_D2H(MB/s),Pinned_H2D(MB/s),Pinned_D2H(MB/s),Mapped_H2D(MB/s),Mapped_D2H(MB/s)\n";

    for (dataSize = step; dataSize <= MAXDATASIZE; dataSize += step) {
        float pageableH2D, pageableD2H;
        float pinnedH2D, pinnedD2H;
        float mappedH2D, mappedD2H;

        // Test pageable H2D
        cudaEventRecord(startT, str);
        for (i = 0; i < iter; i++) {
            cudaMemcpyAsync(d_data, h_data_pageable, sizeof(int) * dataSize, cudaMemcpyHostToDevice, str);
        }
        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&duration, startT, endT);
        pageableH2D = (dataSize * sizeof(int) * iter / (duration / 1e3)) / (1024 * 1024);

        // Test pageable D2H
        cudaEventRecord(startT, str);
        for (i = 0; i < iter; i++) {
            cudaMemcpyAsync(h_data_pageable, d_data, sizeof(int) * dataSize, cudaMemcpyDeviceToHost, str);
        }
        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&duration, startT, endT);
        pageableD2H = (dataSize * sizeof(int) * iter / (duration / 1e3)) / (1024 * 1024);

        // Test pinned H2D
        cudaEventRecord(startT, str);
        for (i = 0; i < iter; i++) {
            cudaMemcpyAsync(d_data, h_data_pinned, sizeof(int) * dataSize, cudaMemcpyHostToDevice, str);
        }
        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&duration, startT, endT);
        pinnedH2D = (dataSize * sizeof(int) * iter / (duration / 1e3)) / (1024 * 1024);

        // Test pinned D2H
        cudaEventRecord(startT, str);
        for (i = 0; i < iter; i++) {
            cudaMemcpyAsync(h_data_pinned, d_data, sizeof(int) * dataSize, cudaMemcpyDeviceToHost, str);
        }
        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&duration, startT, endT);
        pinnedD2H = (dataSize * sizeof(int) * iter / (duration / 1e3)) / (1024 * 1024);

        // Test mapped H2D (access by kernel)
        int* d_mapped;
        cudaHostGetDevicePointer((void**)&d_mapped, h_data_mapped, 0);
        cudaEventRecord(startT, str);
        for (i = 0; i < iter; i++) {
            accessMappedMemoryKernel << <(dataSize + 255) / 256, 256, 0, str >> > (d_mapped, dataSize);
        }
        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&duration, startT, endT);
        mappedH2D = (dataSize * sizeof(int) * iter / (duration / 1e3)) / (1024 * 1024);

        // Test mapped D2H (access by kernel)
        cudaEventRecord(startT, str);
        for (i = 0; i < iter; i++) {
            accessMappedMemoryKernel << <(dataSize + 255) / 256, 256, 0, str >> > (d_mapped, dataSize);
        }
        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&duration, startT, endT);
        mappedD2H = (dataSize * sizeof(int) * iter / (duration / 1e3)) / (1024 * 1024);

        // Write results to file
        outFile << (dataSize * sizeof(int)) << "," << pageableH2D << "," << pageableD2H << "," << pinnedH2D << "," << pinnedD2H << "," << mappedH2D << "," << mappedD2H << "\n";
    }

    outFile.close();

    // Clean up
    cudaStreamDestroy(str);
    cudaEventDestroy(startT);
    cudaEventDestroy(endT);
    free(h_data_pageable);
    cudaFreeHost(h_data_pinned);
    cudaFreeHost(h_data_mapped);
    cudaFree(d_data);
    cudaDeviceReset();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <iterations> <step size>\n", argv[0]);
        return -1;
    }

    int iter = atoi(argv[1]);
    int step = atoi(argv[2]);

    testMemoryCopySpeed(iter, step, "memory_copy_results.csv");

    printf("Memory copy speed test completed. Results saved to memory_copy_results.csv.\n");
    return 0;
}
