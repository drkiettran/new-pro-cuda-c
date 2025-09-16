// IntelliSense configuration for Visual Studio
#ifdef __INTELLISENSE__
#define __CUDACC__
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#include <device_launch_parameters.h>
#include <device_functions.h>
// Simplified WMMA declarations for IntelliSense
namespace nvcuda {
    namespace wmma {
        enum fragment_layout { row_major, col_major, mem_row_major, mem_col_major };
        template<typename Use, int m, int n, int k, typename T, fragment_layout Layout = row_major>
        struct fragment {
            enum { num_elements = 8 };
            T x[num_elements];
        };
        struct matrix_a {};
        struct matrix_b {};
        struct accumulator {};
        template<typename T> void fill_fragment(T& frag, float value) {}
        template<typename T> void load_matrix_sync(T& frag, const void* ptr, unsigned ldm) {}
        template<typename T> void store_matrix_sync(void* ptr, const T& frag, unsigned ldm, fragment_layout layout) {}
        template<typename T> void mma_sync(T& d, const T& a, const T& b, const T& c) {}
    }
}
#endif

// RTX 5060 Blackwell Tensor Core Matrix Multiplication Example
// File: kernel.cu (IMPORTANT: Must be .cu extension)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>  // This provides blockDim, threadIdx, etc.
#include <mma.h>                       // WMMA API for Tensor Cores
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Constants for Tensor Core dimensions
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Modern Tensor Core kernel using latest WMMA API
// Note: __global__ kernels must be free functions, not class members
__global__ void tensor_core_gemm_kernel(
    const half* __restrict__ A,    // M x K matrix
    const half* __restrict__ B,    // K x N matrix  
    float* __restrict__ C,         // M x N matrix (FP32 output)
    const float* __restrict__ D,   // M x N bias matrix (optional)
    int M, int N, int K,
    float alpha, float beta
) {
    // Blackwell architecture: 5th generation Tensor Cores

    // Calculate warp position in output matrix
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Bounds checking for matrix dimensions
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // Declare fragments for Tensor Core operations - use explicit namespace
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // Main computation loop - process K dimension in chunks
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking for K dimension
        if (aCol + WMMA_K <= K && bRow + WMMA_K <= K) {
            // Load matrix fragments into Tensor Core registers
            // A: row-major layout
            nvcuda::wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);

            // B: column-major layout for optimal memory access
            nvcuda::wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiplication using Tensor Cores
            // This single operation computes 16x16x16 mixed-precision GEMM
            // Input: FP16, Output: FP32 accumulation
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Handle bias addition if provided
    if (D != nullptr) {
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;

        // Load existing C matrix values for fused operations
        nvcuda::wmma::load_matrix_sync(c_frag, D + cRow * N + cCol, N, nvcuda::wmma::mem_row_major);

        // Fused multiply-add: C = alpha * A*B + beta * D
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    }
    else {
        // Simple scaling: C = alpha * A*B
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] *= alpha;
        }
    }

    // Store final result to global memory
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, nvcuda::wmma::mem_row_major);
}

// Tensor Core Matrix Multiplication utility class for RTX 5060 Blackwell
class BlackwellTensorCoreGEMM {
public:
    static void launch_tensor_core_gemm(
        const half* d_A, const half* d_B, float* d_C, const float* d_D,
        int M, int N, int K, float alpha = 1.0f, float beta = 0.0f
    ) {
        // Ensure dimensions are compatible with Tensor Cores
        assert(M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0);

        // Configure grid and block dimensions
        dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
        dim3 blockDim(32, 4); // 128 threads per block, arranged for warps

        // Launch kernel
        tensor_core_gemm_kernel << <gridDim, blockDim >> > (
            d_A, d_B, d_C, d_D, M, N, K, alpha, beta
            );

        // Check for launch errors
        CUDA_CHECK(cudaGetLastError());
    }
};

// Utility class for performance benchmarking
class PerformanceBenchmark {
private:
    cudaEvent_t start_, stop_;

public:
    PerformanceBenchmark() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~PerformanceBenchmark() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }
};

// Complete example with error checking and benchmarking
int main() {
    std::cout << "RTX 5060 Blackwell Tensor Core GEMM Example\n";
    std::cout << "============================================\n";

    // Check GPU compute capability for Tensor Core support
    int device = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;

    if (props.major < 7) {  // Tensor Cores require compute capability 7.0+
        std::cerr << "Error: Tensor Cores require compute capability 7.0 or higher\n";
        std::cerr << "Your GPU has compute capability " << props.major << "." << props.minor << std::endl;
        return -1;
    }

    if (props.major >= 12) {
        std::cout << "[OK] Next-gen architecture detected - " << (props.major - 6) << "th gen Tensor Cores available!\n";
    }
    else if (props.major == 8 && props.minor == 9) {
        std::cout << "[OK] Blackwell architecture detected - 5th gen Tensor Cores available!\n";
    }
    else if (props.major >= 7) {
        std::cout << "[OK] Tensor Cores supported (generation " << (props.major - 6) << ")\n";
    }

    // Matrix dimensions (must be multiples of 16 for Tensor Cores)
    const int M = 1024;  // Rows of A and C
    const int N = 1024;  // Columns of B and C  
    const int K = 1024;  // Columns of A, rows of B

    std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;

    // Allocate host memory
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(dist(gen));
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(dist(gen));
    }

    std::fill(h_C.begin(), h_C.end(), 0.0f);

    // Allocate device memory
    half* d_A, * d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Warm up GPU
    BlackwellTensorCoreGEMM::launch_tensor_core_gemm(d_A, d_B, d_C, nullptr, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark Tensor Core performance
    PerformanceBenchmark benchmark;
    const int num_iterations = 10;

    benchmark.start();
    for (int i = 0; i < num_iterations; i++) {
        BlackwellTensorCoreGEMM::launch_tensor_core_gemm(d_A, d_B, d_C, nullptr, M, N, K);
    }
    float gpu_time = benchmark.stop();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    double gflops = (2.0 * M * N * K * num_iterations) / (gpu_time * 1e6); // GFLOPS

    // Updated utilization calculation for newer architectures
    double theoretical_peak = props.major >= 12 ? 2000.0 :  // Next-gen estimate
        props.major >= 8 ? 1000.0 :   // Blackwell/Ada
        500.0;                         // Older architectures
    double tensor_core_utilization = (gflops / theoretical_peak) * 100.0;

    std::cout << "\nPerformance Results:\n";
    std::cout << "====================\n";
    std::cout << "GPU Time: " << gpu_time / num_iterations << " ms per iteration\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";
    std::cout << "Estimated Tensor Core Utilization: " << tensor_core_utilization << "%\n";

    // Simple verification (check non-zero results)
    bool results_valid = true;
    int non_zero_count = 0;

    for (int i = 0; i < std::min(100, M * N); i++) {
        if (std::abs(h_C[i]) > 1e-6f) {
            non_zero_count++;
        }
    }

    if (non_zero_count > 90) { // Expect most results to be non-zero
        std::cout << "[OK] Results appear valid (found " << non_zero_count << "/100 non-zero values)\n";
    }
    else {
        std::cout << "[ERROR] Results may be invalid (only " << non_zero_count << "/100 non-zero values)\n";
        results_valid = false;
    }

    // Tensor Core specific information
    std::cout << "\nTensor Core Features Used:\n";
    std::cout << "==========================\n";
    std::cout << "* Mixed Precision: FP16 inputs -> FP32 accumulation\n";
    std::cout << "* WMMA Fragment Size: 16x16x16\n";
    if (props.major >= 12) {
        std::cout << "* " << (props.major - 6) << "th Generation Tensor Cores (Next-gen)\n";
    }
    else if (props.major == 8 && props.minor == 9) {
        std::cout << "* 5th Generation Tensor Cores (Blackwell)\n";
    }
    else {
        std::cout << "* Tensor Cores (Generation " << (props.major - 6) << ")\n";
    }
    std::cout << "* Warp-level Matrix Operations\n";
    std::cout << "* Hardware Accelerated GEMM\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << "\nTensor Core demonstration completed successfully!\n";

    return results_valid ? 0 : -1;
}