#include <iostream>
#include <cuda_runtime.h>

__global__ void warpShuffleReduction(int* data) {
    int laneId = threadIdx.x & 31;  // Get thread ID within the warp
    int value = data[threadIdx.x];
    printf("%d. value: %d, threadIdx.x: %d\n\n", laneId, value, threadIdx.x);

    // Reduce within the warp using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        printf("%d. value: %d, off-set: %d {\n", laneId, value, offset);
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
        printf("%d. value: %d, off-set: %d }\n", laneId, value, offset);
    }

    // Store results from thread 0 of each warp
    if (laneId == 0) {
        data[threadIdx.x] = value;
        printf("\nlaneId 0, value: %d\n", value);
    }
}

int main() {
    const int N = 32;
    int h_data[N];
    int* d_data;

    // Initialize data
    for (int i = 0; i < N; i++) h_data[i] = i + 1;

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    warpShuffleReduction << <1, N >> > (d_data);

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    std::cout << "Warp Reduction Result: " << h_data[0] << std::endl;
    return 0;
}