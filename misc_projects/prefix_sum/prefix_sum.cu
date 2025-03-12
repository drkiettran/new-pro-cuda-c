#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"


// CUDA kernel for performing prefix sum (exclusive scan)
__global__ void prefixSumKernel(float* input, float* output, int n) {
    extern __shared__ float sharedMem[]; // Shared memory for block
    int threadIdxInBlock = threadIdx.x;
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load input into shared memory
    if (globalIdx < n) {
        sharedMem[threadIdxInBlock] = input[globalIdx];
    }
    else {
        sharedMem[threadIdxInBlock] = 0.0f; // Handle out-of-bounds threads
    }
    __syncthreads();

    // Perform the up-sweep (reduce) phase
    for (int step = 1; step < blockDim.x; step *= 2) {
        if (threadIdxInBlock % (2 * step) == 0 && threadIdxInBlock + step < blockDim.x) {
            sharedMem[threadIdxInBlock + 2 * step - 1] += sharedMem[threadIdxInBlock + step - 1];
        }
        __syncthreads();
    }

    // Set the last element in the block to 0 (exclusive scan requires it)
    if (threadIdxInBlock == blockDim.x - 1) {
        sharedMem[threadIdxInBlock] = 0.0f;
    }
    __syncthreads();

    // Perform the down-sweep phase
    for (int step = blockDim.x / 2; step >= 1; step /= 2) {
        if (threadIdxInBlock % (2 * step) == 0 && threadIdxInBlock + step < blockDim.x) {
            float temp = sharedMem[threadIdxInBlock + step - 1];
            sharedMem[threadIdxInBlock + step - 1] = sharedMem[threadIdxInBlock + 2 * step - 1];
            sharedMem[threadIdxInBlock + 2 * step - 1] += temp;
        }
        __syncthreads();
    }

    // Write the results from shared memory back to the global memory
    if (globalIdx < n) {
        output[globalIdx] = sharedMem[threadIdxInBlock];
    }
}

// Utility function to initialize an array
void initializeArray(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<float>(i + 1);
    }
}

// Host function to perform prefix sum
void prefixSum(float* input, float* output, int n) {
    float* d_input, * d_output;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 1024; // Adjust based on your GPU's capabilities
    int numBlocks = (n + blockSize - 1) / blockSize;
    prefixSumKernel << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_input, d_output, n);

    // Copy results back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// Main function to test the kernel
int main() {
    const int n = 16; // Size of the input array
    float h_input[n], h_output[n];

    initializeArray(h_input, n);

    std::cout << "Input Array: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    prefixSum(h_input, h_output, n);

    std::cout << "Output Array (Prefix-Sum): ";
    for (int i = 0; i < n; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
