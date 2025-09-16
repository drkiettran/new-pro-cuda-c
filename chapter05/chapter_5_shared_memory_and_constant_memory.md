# CUDA Programming: Shared Memory and Constant Memory

## Table of Contents
1. [Introduction](#introduction)
2. [Shared Memory Fundamentals](#shared-memory-fundamentals)
3. [Data Arrangement in Shared Memory](#data-arrangement-in-shared-memory)
4. [Index Conversion: 2D to Linear Memory](#index-conversion-2d-to-linear-memory)
5. [Bank Conflicts and Resolution](#bank-conflicts-and-resolution)
6. [Caching Data in Shared Memory](#caching-data-in-shared-memory)
7. [Avoiding Non-Coalesced Global Memory Access](#avoiding-non-coalesced-global-memory-access)
8. [Constant Memory and Cache Types](#constant-memory-and-cache-types)
9. [Warp Shuffle Instructions](#warp-shuffle-instructions)
10. [Practical Examples](#practical-examples)
11. [Performance Optimization Guidelines](#performance-optimization-guidelines)

---

## Introduction

CUDA's memory hierarchy provides several types of memory with different characteristics. This tutorial focuses on **Shared Memory** and **Constant Memory**, two crucial memory types for high-performance GPU programming.

### Memory Hierarchy Overview
- **Global Memory**: Large, high-latency, accessible by all threads
- **Shared Memory**: Small, low-latency, accessible within a thread block
- **Constant Memory**: Read-only, cached, broadcast to all threads
- **Local Memory**: Thread-private, stored in global memory space
- **Registers**: Fastest, thread-private storage

---

## Shared Memory Fundamentals

### What is Shared Memory?
Shared memory is a programmable cache that can be accessed by all threads within a thread block. It's much faster than global memory but has limited capacity.

### Key Characteristics:
- **Size**: Typically 48KB or 64KB per SM (depending on GPU architecture)
- **Latency**: ~100x faster than global memory
- **Scope**: Shared within a thread block
- **Lifetime**: Exists for the duration of the thread block

### Declaration Syntax:
```c
// Static allocation
__shared__ int shared_array[256];

// Dynamic allocation
extern __shared__ int dynamic_shared[];

// In kernel launch
kernel<<<blocks, threads, shared_memory_size>>>();
```

---

## Data Arrangement in Shared Memory

### Memory Layout
Shared memory is organized as a linear array of 32-bit words, divided into 32 banks. Each bank can service one memory request per clock cycle.

```c
// Example: 2D array mapping to shared memory
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Memory layout visualization:
// Bank 0: [0,0] [0,4] [0,8] [0,12] ...
// Bank 1: [0,1] [0,5] [0,9] [0,13] ...
// Bank 2: [0,2] [0,6] [0,10] [0,14] ...
// Bank 3: [0,3] [0,7] [0,11] [0,15] ...
```

### Bank Assignment Formula
For a 4-byte element at address `addr`:
```
bank_id = (addr / 4) % 32
```

### Example: Understanding Bank Assignment
```c
__global__ void demonstrate_bank_assignment() {
    __shared__ int data[128];
    
    int tid = threadIdx.x;
    
    // These accesses go to different banks (no conflict)
    data[tid] = tid;  // tid = 0,1,2,3... → banks 0,1,2,3...
    
    // These accesses cause bank conflicts
    data[tid * 2] = tid;  // tid = 0,1,2,3... → banks 0,2,4,6...
}
```

---

## Index Conversion: 2D to Linear Memory

### Problem Statement
Converting between 2D thread indices and linear memory addresses is crucial for efficient shared memory usage.

### Basic Conversion Formulas

#### 2D to Linear (Row-Major)
```c
int linear_index = row * width + col;
```

#### 2D to Linear (Column-Major)
```c
int linear_index = col * height + row;
```

### Practical Example: Matrix Tiling
```c
#define TILE_SIZE 16

__global__ void matrix_multiply_tiled(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Global indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load tiles into shared memory
        if (row < N && k * TILE_SIZE + tx < N)
            tile_A[ty][tx] = A[row * N + k * TILE_SIZE + tx];
        else
            tile_A[ty][tx] = 0.0f;
            
        if (col < N && k * TILE_SIZE + ty < N)
            tile_B[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
        else
            tile_B[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[ty][i] * tile_B[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Advanced Index Conversion
```c
// Convert between different memory layouts
__device__ int convert_2d_to_linear(int row, int col, int width, bool row_major) {
    if (row_major) {
        return row * width + col;
    } else {
        return col * width + row;  // Column-major
    }
}

// Reverse conversion
__device__ void convert_linear_to_2d(int linear_idx, int width, 
                                    int *row, int *col, bool row_major) {
    if (row_major) {
        *row = linear_idx / width;
        *col = linear_idx % width;
    } else {
        *col = linear_idx / width;
        *row = linear_idx % width;
    }
}
```

---

## Bank Conflicts and Resolution

### Understanding Bank Conflicts
A bank conflict occurs when multiple threads in a warp access the same bank simultaneously, causing serialization.

### Types of Access Patterns

#### 1. No Conflict (Ideal)
```c
__shared__ int data[32];
int tid = threadIdx.x;
data[tid] = value;  // Each thread accesses different bank
```

#### 2. Bank Conflict (Problematic)
```c
__shared__ int data[32];
int tid = threadIdx.x;
data[tid * 2] = value;  // Multiple threads access same banks
```

#### 3. Broadcast (Acceptable)
```c
__shared__ int data[32];
int tid = threadIdx.x;
int value = data[0];  // All threads read same address (broadcast)
```

### Bank Conflict Resolution Strategies

#### Strategy 1: Padding
```c
// Problematic: Bank conflicts in column access
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Solution: Add padding to avoid conflicts
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
```

#### Strategy 2: Offset Access
```c
__global__ void resolve_conflicts_with_offset() {
    __shared__ int data[32 * 32];
    int tid = threadIdx.x;
    
    // Problematic access pattern
    // int idx = tid * 32;
    
    // Better: Add offset to distribute across banks
    int offset = tid / 32;
    int idx = tid * 32 + offset;
    
    data[idx] = tid;
}
```

#### Strategy 3: Transpose Pattern
```c
__global__ void transpose_shared_memory() {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 for padding
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load with coalesced access
    tile[ty][tx] = input[...];
    __syncthreads();
    
    // Store with transposed pattern
    output[...] = tile[tx][ty];
}
```

### Bank Conflict Analysis Tool
```c
__device__ void analyze_bank_conflicts(int access_pattern[], int num_accesses) {
    int bank_count[32] = {0};  // Track accesses per bank
    
    for (int i = 0; i < num_accesses; i++) {
        int bank = (access_pattern[i] / 4) % 32;
        bank_count[bank]++;
    }
    
    int max_conflicts = 0;
    for (int i = 0; i < 32; i++) {
        if (bank_count[i] > max_conflicts) {
            max_conflicts = bank_count[i];
        }
    }
    
    // max_conflicts = 1: No conflicts
    // max_conflicts > 1: Conflicts present
}
```

---

## Caching Data in Shared Memory

### Benefits of Shared Memory Caching
1. **Reduced Global Memory Bandwidth**: Multiple reuse of data
2. **Lower Latency**: ~100x faster than global memory
3. **Improved Throughput**: Better memory coalescing

### Caching Strategies

#### 1. Tile-Based Caching
```c
#define TILE_SIZE 16
#define FILTER_SIZE 5

__global__ void convolution_with_caching(float *input, float *output, 
                                       float *filter, int width, int height) {
    __shared__ float tile[TILE_SIZE + FILTER_SIZE - 1][TILE_SIZE + FILTER_SIZE - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate global position
    int gx = bx * TILE_SIZE + tx;
    int gy = by * TILE_SIZE + ty;
    
    // Load tile with halo (boundary handling)
    for (int dy = 0; dy < TILE_SIZE + FILTER_SIZE - 1; dy += blockDim.y) {
        for (int dx = 0; dx < TILE_SIZE + FILTER_SIZE - 1; dx += blockDim.x) {
            int sx = tx + dx;
            int sy = ty + dy;
            int gx_load = bx * TILE_SIZE + sx - FILTER_SIZE / 2;
            int gy_load = by * TILE_SIZE + sy - FILTER_SIZE / 2;
            
            if (sx < TILE_SIZE + FILTER_SIZE - 1 && sy < TILE_SIZE + FILTER_SIZE - 1) {
                if (gx_load >= 0 && gx_load < width && gy_load >= 0 && gy_load < height) {
                    tile[sy][sx] = input[gy_load * width + gx_load];
                } else {
                    tile[sy][sx] = 0.0f;  // Boundary condition
                }
            }
        }
    }
    
    __syncthreads();
    
    // Perform convolution using cached data
    if (gx < width && gy < height) {
        float sum = 0.0f;
        for (int fy = 0; fy < FILTER_SIZE; fy++) {
            for (int fx = 0; fx < FILTER_SIZE; fx++) {
                sum += tile[ty + fy][tx + fx] * filter[fy * FILTER_SIZE + fx];
            }
        }
        output[gy * width + gx] = sum;
    }
}
```

#### 2. Cooperative Caching
```c
__global__ void cooperative_caching_example() {
    __shared__ float cache[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Cooperative loading: each thread loads multiple elements
    for (int i = tid; i < 256; i += blockDim.x) {
        cache[i] = global_data[bid * 256 + i];
    }
    
    __syncthreads();
    
    // All threads can now access cached data efficiently
    float result = 0.0f;
    for (int i = 0; i < 256; i++) {
        result += cache[i] * some_function(i);
    }
}
```

### Cache Efficiency Metrics
```c
// Calculate cache hit ratio
float cache_hit_ratio = (float)cache_hits / (cache_hits + cache_misses);

// Effective bandwidth utilization
float effective_bandwidth = (bytes_transferred / total_time) / peak_bandwidth;
```

---

## Avoiding Non-Coalesced Global Memory Access

### Understanding Memory Coalescing
Memory coalescing occurs when threads in a warp access consecutive memory locations, allowing the hardware to combine multiple accesses into fewer transactions.

### Coalescing Rules
1. **Alignment**: Starting address should be aligned to cache line boundary
2. **Consecutive Access**: Threads should access consecutive addresses
3. **Stride**: Access pattern should have unit stride

### Common Non-Coalesced Patterns and Solutions

#### Problem 1: Strided Access
```c
// Problematic: Non-coalesced access
__global__ void non_coalesced_access(float *data, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float value = data[tid * stride];  // Non-coalesced if stride > 1
}

// Solution: Use shared memory for reordering
__global__ void coalesced_with_shared_memory(float *data, int stride) {
    __shared__ float temp[256];
    
    int tid = threadIdx.x;
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Coalesced load
    temp[tid] = data[global_tid];
    __syncthreads();
    
    // Access with desired stride pattern locally
    float value = temp[(tid * stride) % 256];
}
```

#### Problem 2: Matrix Transpose
```c
// Problematic: Non-coalesced writes
__global__ void naive_transpose(float *input, float *output, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx < width && idy < height) {
        output[idx * height + idy] = input[idy * width + idx];  // Non-coalesced write
    }
}

// Solution: Shared memory transpose
__global__ void optimized_transpose(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int idx = bx * TILE_SIZE + tx;
    int idy = by * TILE_SIZE + ty;
    
    // Coalesced read
    if (idx < width && idy < height) {
        tile[ty][tx] = input[idy * width + idx];
    }
    
    __syncthreads();
    
    // Calculate transposed indices
    int out_idx = by * TILE_SIZE + tx;
    int out_idy = bx * TILE_SIZE + ty;
    
    // Coalesced write
    if (out_idx < height && out_idy < width) {
        output[out_idy * height + out_idx] = tile[tx][ty];
    }
}
```

### Coalescing Analysis
```c
// Check memory access pattern for coalescing
__device__ bool is_coalesced_access(int thread_id, int *addresses, int num_threads) {
    if (num_threads != 32) return false;  // Must be full warp
    
    // Check if addresses are consecutive
    for (int i = 1; i < num_threads; i++) {
        if (addresses[i] != addresses[i-1] + sizeof(float)) {
            return false;
        }
    }
    
    // Check alignment
    return (addresses[0] % 128) == 0;  // 128-byte alignment
}
```

---

## Constant Memory and Cache Types

### Constant Memory Overview
Constant memory is a read-only memory space that's cached and can broadcast the same value to all threads in a warp efficiently.

### Characteristics
- **Size**: 64KB per GPU
- **Access**: Read-only from kernels
- **Caching**: Cached in constant cache
- **Broadcast**: Efficient for same-value access across warp

### Declaration and Usage
```c
// Declare constant memory
__constant__ float const_data[1024];

// Host code: Copy to constant memory
cudaMemcpyToSymbol(const_data, host_data, size);

// Device code: Access constant memory
__global__ void use_constant_memory() {
    int tid = threadIdx.x;
    float value = const_data[tid % 1024];  // Cached access
}
```

### Constant Cache vs Read-Only Cache

#### Constant Cache
- **Purpose**: Designed for constant memory access
- **Size**: 8KB per SM
- **Optimization**: Best for uniform access patterns
- **Usage**: Automatic for `__constant__` variables

#### Read-Only Cache (Texture Cache)
- **Purpose**: General-purpose read-only cache
- **Size**: Varies by architecture (e.g., 48KB on newer GPUs)
- **Optimization**: Good for spatial locality
- **Usage**: Explicit with `__ldg()` intrinsic

### Comparison Example
```c
// Using constant memory
__constant__ float filter_const[25];

__global__ void convolution_constant(float *input, float *output, int width) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // All threads access same filter values - efficient broadcast
    for (int i = 0; i < 25; i++) {
        output[tid] += input[tid + i] * filter_const[i];
    }
}

// Using read-only cache
__global__ void convolution_readonly(float *input, float *output, 
                                   const float* __restrict__ filter, int width) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Compiler may use read-only cache for filter access
    for (int i = 0; i < 25; i++) {
        output[tid] += input[tid + i] * __ldg(&filter[i]);
    }
}
```

### Performance Considerations
```c
// Optimal constant memory usage
__constant__ float lookup_table[256];

__global__ void optimal_constant_usage() {
    int tid = threadIdx.x;
    
    // Good: All threads access same value
    float common_value = lookup_table[0];
    
    // Suboptimal: Different threads access different values
    float different_value = lookup_table[tid];  // May cause cache misses
}
```

---

## Warp Shuffle Instructions

### Introduction to Warp Shuffle
Warp shuffle instructions allow direct exchange of data between threads in a warp without using shared memory.

### Types of Shuffle Operations

#### 1. `__shfl_sync()` - Indexed Shuffle
```c
// Exchange data between specific lanes
__global__ void indexed_shuffle_example() {
    int tid = threadIdx.x;
    int value = tid;
    
    // Get value from lane 0
    int broadcasted = __shfl_sync(0xFFFFFFFF, value, 0);
    
    // Get value from lane (tid + 1) % 32
    int next_value = __shfl_sync(0xFFFFFFFF, value, (tid + 1) % 32);
}
```

#### 2. `__shfl_up_sync()` - Shift Up
```c
__global__ void shift_up_example() {
    int tid = threadIdx.x;
    int value = tid;
    
    // Get value from lane (tid - 1), if tid == 0, use own value
    int prev_value = __shfl_up_sync(0xFFFFFFFF, value, 1);
}
```

#### 3. `__shfl_down_sync()` - Shift Down
```c
__global__ void shift_down_example() {
    int tid = threadIdx.x;
    int value = tid;
    
    // Get value from lane (tid + 1), if tid == 31, use own value
    int next_value = __shfl_down_sync(0xFFFFFFFF, value, 1);
}
```

#### 4. `__shfl_xor_sync()` - Butterfly Exchange
```c
__global__ void butterfly_exchange() {
    int tid = threadIdx.x;
    int value = tid;
    
    // Exchange with thread whose ID differs by XOR with 1
    int exchanged = __shfl_xor_sync(0xFFFFFFFF, value, 1);
}
```

### Practical Applications

#### 1. Warp-Level Reduction
```c
__device__ float warp_reduce_sum(float value) {
    // Butterfly reduction pattern
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

__global__ void parallel_reduction(float *input, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    float value = (tid < n) ? input[tid] : 0.0f;
    
    // Warp-level reduction
    value = warp_reduce_sum(value);
    
    // Only lane 0 writes result
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, value);
    }
}
```

#### 2. Prefix Sum (Scan)
```c
__device__ float warp_prefix_sum(float value) {
    // Inclusive prefix sum within warp
    for (int offset = 1; offset < 32; offset <<= 1) {
        float temp = __shfl_up_sync(0xFFFFFFFF, value, offset);
        if (threadIdx.x >= offset) {
            value += temp;
        }
    }
    return value;
}
```

#### 3. Matrix Transpose with Shuffle
```c
__global__ void transpose_with_shuffle(float *input, float *output, int width) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Load 32 elements per warp
    float value = input[tid];
    
    // Transpose pattern using shuffle
    for (int i = 0; i < 32; i++) {
        float transposed = __shfl_sync(0xFFFFFFFF, value, i);
        if (lane_id == i) {
            output[warp_id * 32 + i] = transposed;
        }
    }
}
```

### Performance Benefits
```c
// Comparison: Shared memory vs Shuffle
__global__ void reduction_shared_memory(float *data, float *result) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    sdata[tid] = data[tid];
    __syncthreads();
    
    // Reduction using shared memory
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

__global__ void reduction_shuffle(float *data, float *result) {
    int tid = threadIdx.x;
    float value = data[tid];
    
    // Warp-level reduction with shuffle
    value = warp_reduce_sum(value);
    
    // Block-level reduction
    __shared__ float warp_results[8];  // 256/32 = 8 warps
    
    if (tid % 32 == 0) {
        warp_results[tid / 32] = value;
    }
    __syncthreads();
    
    if (tid < 8) {
        float final_value = warp_results[tid];
        final_value = warp_reduce_sum(final_value);
        if (tid == 0) result[blockIdx.x] = final_value;
    }
}
```

---

## Practical Examples

### Example 1: Optimized Matrix Multiplication
```c
#define TILE_SIZE 16

__global__ void optimized_matrix_multiply(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];  // Padding to avoid bank conflicts
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Coalesced loading with boundary checks
        if (row < N && k * TILE_SIZE + tx < N) {
            tile_A[ty][tx] = A[row * N + k * TILE_SIZE + tx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        if (col < N && k * TILE_SIZE + ty < N) {
            tile_B[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute using cached data
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[ty][i] * tile_B[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Example 2: Stencil Computation with Shared Memory
```c
#define RADIUS 3
#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE + 2 * RADIUS)

__constant__ float stencil_coeffs[2 * RADIUS + 1];

__global__ void stencil_1d(float *input, float *output, int n) {
    __shared__ float tile[BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int gx = blockIdx.x * TILE_SIZE + tx;
    
    // Load data with halo
    int left_halo = gx - RADIUS;
    if (left_halo >= 0 && left_halo < n) {
        tile[tx] = input[left_halo];
    } else {
        tile[tx] = 0.0f;  // Boundary condition
    }
    
    __syncthreads();
    
    // Compute stencil for valid threads
    if (tx >= RADIUS && tx < TILE_SIZE + RADIUS && gx < n) {
        float result = 0.0f;
        
        #pragma unroll
        for (int i = -RADIUS; i <= RADIUS; i++) {
            result += tile[tx + i] * stencil_coeffs[i + RADIUS];
        }
        
        output[gx] = result;
    }
}
```

### Example 3: Histogram with Shared Memory
```c
#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256

__global__ void histogram_shared(unsigned char *input, int *output, int n) {
    __shared__ int hist[HISTOGRAM_SIZE];
    
    int tx = threadIdx.x;
    int gx = blockIdx.x * blockDim.x + tx;
    
    // Initialize shared histogram
    if (tx < HISTOGRAM_SIZE) {
        hist[tx] = 0;
    }
    __syncthreads();
    
    // Compute histogram
    while (gx < n) {
        atomicAdd(&hist[input[gx]], 1);
        gx += blockDim.x * gridDim.x;
    }
    
    __syncthreads();
    
    // Write results back
    if (tx < HISTOGRAM_SIZE) {
        atomicAdd(&output[tx], hist[tx]);
    }
}
```

---

## Performance Optimization Guidelines

### 1. Shared Memory Optimization
- **Minimize Bank Conflicts**: Use padding or offset access patterns
- **Maximize Occupancy**: Balance shared memory usage with thread blocks
- **Optimize Data Layout**: Choose row-major vs column-major based on access patterns

### 2. Memory Coalescing
- **Align Data Structures**: Use proper alignment for optimal memory transactions
- **Sequential Access**: Ensure consecutive threads access consecutive memory
- **Avoid Irregular Patterns**: Use shared memory to regularize access patterns

### 3. Constant Memory Usage
- **Uniform Access**: Use constant memory when all threads access same data
- **Size Limitations**: Keep constant data under 64KB
- **Cache Efficiency**: Minimize divergent access patterns

### 4. Warp Shuffle Optimization
- **Reduce Shared Memory**: Use shuffle instead of shared memory for simple operations
- **Minimize Synchronization**: Shuffle operations are synchronous within warp
- **Pattern Matching**: Use appropriate shuffle variant for your data exchange pattern

### Performance Analysis Tools
```c
// Measure memory bandwidth
__global__ void measure_bandwidth(float *data, int n, int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        if (tid < n) {
            data[tid] = data[tid] * 2.0f;
        }
    }
    
    clock_t end = clock();
    
    if (tid == 0) {
        float elapsed = (float)(end - start);
        printf("Bandwidth: %f GB/s\n", (n * sizeof(float) * iterations) / elapsed / 1e9);
    }
}
```

### Debugging Techniques
```c
// Shared memory debugging
__global__ void debug_shared_memory() {
    __shared__ int debug_data[256];
    
    int tid = threadIdx.x;
    
    // Initialize with thread ID
    debug_data[tid] = tid;
    __syncthreads();
    
    // Check for expected values
    if (tid == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            if (debug_data[i] != i) {
                printf("Error: Expected %d, got %d at index %d\n", i, debug_data[i], i);
            }
        }
    }
}

// Bank conflict detection
__global__ void detect_bank_conflicts() {
    __shared__ int test_data[1024];
    
    int tid = threadIdx.x;
    
    // Test different access patterns
    clock_t start, end;
    
    // Pattern 1: No conflicts
    start = clock();
    test_data[tid] = tid;
    __syncthreads();
    end = clock();
    
    if (tid == 0) {
        printf("No conflict pattern time: %d\n", (int)(end - start));
    }
    
    // Pattern 2: With conflicts
    start = clock();
    test_data[tid * 2] = tid;
    __syncthreads();
    end = clock();
    
    if (tid == 0) {
        printf("Conflict pattern time: %d\n", (int)(end - start));
    }
}
```

---

## Advanced Topics and Best Practices

### 1. Dynamic Shared Memory Management
```c
// Dynamically allocated shared memory
__global__ void dynamic_shared_memory(float *input, float *output, int n) {
    extern __shared__ float dynamic_data[];
    
    // Partition dynamic shared memory
    float *cache = dynamic_data;
    float *temp = &dynamic_data[blockDim.x];
    int *counters = (int*)&dynamic_data[2 * blockDim.x];
    
    int tid = threadIdx.x;
    
    // Use partitioned memory
    cache[tid] = input[tid];
    temp[tid] = cache[tid] * 2.0f;
    counters[tid] = tid;
    
    __syncthreads();
    
    output[tid] = temp[tid];
}

// Launch with dynamic shared memory
int shared_size = 2 * sizeof(float) * block_size + sizeof(int) * block_size;
dynamic_shared_memory<<<grid_size, block_size, shared_size>>>(input, output, n);
```

### 2. Multi-Level Memory Hierarchy Optimization
```c
__global__ void multi_level_optimization(float *global_data, int n) {
    // Level 1: Registers (automatic)
    float reg_data = 0.0f;
    
    // Level 2: Shared memory
    __shared__ float shared_cache[256];
    
    // Level 3: Constant memory
    __constant__ float const_params[16];
    
    // Level 4: Texture memory (read-only cache)
    // Accessed via __ldg() intrinsic
    
    int tid = threadIdx.x;
    
    // Optimize data movement through hierarchy
    shared_cache[tid] = __ldg(&global_data[tid]);  // Use read-only cache
    __syncthreads();
    
    // Computation using register and shared memory
    reg_data = shared_cache[tid] * const_params[tid % 16];
    
    // Write back to global memory
    global_data[tid] = reg_data;
}
```

### 3. Memory Access Pattern Analysis
```c
// Memory access pattern analyzer
__device__ void analyze_access_pattern(int *addresses, int num_accesses) {
    // Check for coalescing
    bool is_coalesced = true;
    for (int i = 1; i < num_accesses && i < 32; i++) {
        if (addresses[i] != addresses[i-1] + sizeof(int)) {
            is_coalesced = false;
            break;
        }
    }
    
    // Check for bank conflicts in shared memory
    int bank_histogram[32] = {0};
    for (int i = 0; i < num_accesses; i++) {
        int bank = (addresses[i] / sizeof(int)) % 32;
        bank_histogram[bank]++;
    }
    
    int max_bank_conflicts = 0;
    for (int i = 0; i < 32; i++) {
        if (bank_histogram[i] > max_bank_conflicts) {
            max_bank_conflicts = bank_histogram[i];
        }
    }
    
    if (threadIdx.x == 0) {
        printf("Coalesced: %s, Max bank conflicts: %d\n", 
               is_coalesced ? "Yes" : "No", max_bank_conflicts);
    }
}
```

### 4. Warp-Level Primitives Library
```c
// Comprehensive warp-level operations
__device__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

__device__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
    }
    return value;
}

__device__ float warp_scan_inclusive(float value) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float temp = __shfl_up_sync(0xFFFFFFFF, value, offset);
        if (threadIdx.x >= offset) {
            value += temp;
        }
    }
    return value;
}

__device__ float warp_broadcast(float value, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, value, src_lane);
}

__device__ void warp_sort_32(float *data) {
    // Bitonic sort for 32 elements using shuffle
    int tid = threadIdx.x % 32;
    float value = data[tid];
    
    // Bitonic sort implementation
    for (int size = 2; size <= 32; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, value, stride);
            bool ascending = ((tid & size) == 0);
            
            if ((value > other) == ascending) {
                value = other;
            }
        }
    }
    
    data[tid] = value;
}
```

---

## Performance Benchmarking and Profiling

### 1. Memory Bandwidth Benchmarks
```c
// Global memory bandwidth test
__global__ void benchmark_global_memory(float *data, int n, int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        float value = data[tid];
        
        for (int i = 0; i < iterations; i++) {
            value = value * 1.01f + 0.01f;  // Prevent optimization
        }
        
        data[tid] = value;
    }
}

// Shared memory bandwidth test
__global__ void benchmark_shared_memory(float *output, int iterations) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    sdata[tid] = tid;
    
    float value = sdata[tid];
    
    for (int i = 0; i < iterations; i++) {
        __syncthreads();
        sdata[tid] = value * 1.01f + 0.01f;
        __syncthreads();
        value = sdata[tid];
    }
    
    output[tid] = value;
}
```

### 2. Occupancy Calculator
```c
// Calculate theoretical occupancy
__host__ void calculate_occupancy(int block_size, int shared_mem_per_block) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    int max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    int max_blocks_per_sm = props.maxBlocksPerMultiProcessor;
    int shared_mem_per_sm = props.sharedMemPerMultiprocessor;
    
    // Calculate limits
    int blocks_by_threads = max_threads_per_sm / block_size;
    int blocks_by_shared_mem = shared_mem_per_sm / shared_mem_per_block;
    int blocks_by_registers = max_blocks_per_sm;  // Simplified
    
    int actual_blocks = min(blocks_by_threads, 
                           min(blocks_by_shared_mem, blocks_by_registers));
    
    float occupancy = (float)(actual_blocks * block_size) / max_threads_per_sm;
    
    printf("Theoretical occupancy: %.2f%%\n", occupancy * 100);
    printf("Blocks per SM: %d\n", actual_blocks);
    printf("Limiting factor: ");
    
    if (actual_blocks == blocks_by_threads) {
        printf("Thread limit\n");
    } else if (actual_blocks == blocks_by_shared_mem) {
        printf("Shared memory limit\n");
    } else {
        printf("Register limit\n");
    }
}
```

---

## Common Pitfalls and Solutions

### 1. Race Conditions in Shared Memory
```c
// Problematic: Race condition
__global__ void race_condition_example() {
    __shared__ int shared_data[256];
    int tid = threadIdx.x;
    
    // Race condition: multiple threads may write simultaneously
    shared_data[tid] = tid;
    // Missing __syncthreads() here!
    
    int value = shared_data[(tid + 1) % 256];  // May read uninitialized data
}

// Solution: Proper synchronization
__global__ void synchronized_example() {
    __shared__ int shared_data[256];
    int tid = threadIdx.x;
    
    shared_data[tid] = tid;
    __syncthreads();  // Ensure all writes complete
    
    int value = shared_data[(tid + 1) % 256];  // Safe to read
}
```

### 2. Shared Memory Size Limitations
```c
// Check shared memory availability
__host__ void check_shared_memory_limits() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    printf("Shared memory per block: %zu bytes\n", props.sharedMemPerBlock);
    printf("Shared memory per SM: %zu bytes\n", props.sharedMemPerMultiprocessor);
    
    // Calculate maximum tiles
    int max_tile_size = sqrt(props.sharedMemPerBlock / sizeof(float));
    printf("Maximum square tile size: %d x %d\n", max_tile_size, max_tile_size);
}

// Adaptive shared memory allocation
__global__ void adaptive_shared_memory() {
    extern __shared__ float dynamic_shared[];
    
    // Query available shared memory at runtime
    int available_shared = 48 * 1024;  // Example: 48KB
    int elements_per_thread = available_shared / (blockDim.x * sizeof(float));
    
    int tid = threadIdx.x;
    
    // Use only available space
    if (tid * elements_per_thread < available_shared / sizeof(float)) {
        dynamic_shared[tid * elements_per_thread] = tid;
    }
}
```

### 3. Warp Divergence in Shuffle Operations
```c
// Problematic: Warp divergence with shuffle
__global__ void divergent_shuffle() {
    int tid = threadIdx.x;
    int value = tid;
    
    if (tid % 2 == 0) {
        // Only even threads participate - causes divergence
        value = __shfl_sync(0xAAAAAAAA, value, 0);  // Wrong mask
    }
}

// Solution: Use correct active mask
__global__ void correct_shuffle() {
    int tid = threadIdx.x;
    int value = tid;
    
    if (tid % 2 == 0) {
        // Correct mask for even threads only
        unsigned int mask = 0xAAAAAAAA;  // Binary: 10101010...
        value = __shfl_sync(mask, value, 0);
    }
}
```

---

## Testing and Validation

### 1. Unit Tests for Memory Operations
```c
// Test shared memory correctness
__global__ void test_shared_memory_correctness() {
    __shared__ int test_data[256];
    int tid = threadIdx.x;
    
    // Test 1: Basic read/write
    test_data[tid] = tid * 2;
    __syncthreads();
    
    assert(test_data[tid] == tid * 2);
    
    // Test 2: Cross-thread access
    int neighbor = (tid + 1) % blockDim.x;
    int neighbor_value = test_data[neighbor];
    assert(neighbor_value == neighbor * 2);
    
    // Test 3: Reduction
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += test_data[i];
        }
        int expected = blockDim.x * (blockDim.x - 1);  // Sum of 0,2,4,...
        assert(sum == expected);
    }
}

// Test warp shuffle operations
__global__ void test_warp_shuffle() {
    int tid = threadIdx.x;
    int value = tid;
    
    // Test broadcast
    int broadcasted = __shfl_sync(0xFFFFFFFF, value, 0);
    assert(broadcasted == 0);
    
    // Test reduction
    int sum = warp_reduce_sum(value);
    if (tid % 32 == 0) {
        int expected = (32 * 31) / 2;  // Sum of 0 to 31
        assert(sum == expected);
    }
}
```

### 2. Performance Validation
```c
// Performance regression test
__host__ void performance_regression_test() {
    float *d_data;
    int n = 1024 * 1024;
    
    cudaMalloc(&d_data, n * sizeof(float));
    
    // Benchmark different implementations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test 1: Without shared memory
    cudaEventRecord(start);
    naive_kernel<<<(n + 255) / 256, 256>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    
    // Test 2: With shared memory
    cudaEventRecord(start);
    optimized_kernel<<<(n + 255) / 256, 256>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_optimized;
    cudaEventElapsedTime(&time_optimized, start, stop);
    
    // Validate performance improvement
    float speedup = time_naive / time_optimized;
    printf("Speedup: %.2fx\n", speedup);
    
    assert(speedup > 1.5);  // Expect at least 1.5x speedup
    
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

---

## Summary and Key Takeaways

### Memory Hierarchy Optimization Strategy
1. **Analyze Access Patterns**: Understand your application's memory access patterns
2. **Choose Appropriate Memory Type**: Select the right memory type for each use case
3. **Optimize Layout**: Arrange data to minimize conflicts and maximize coalescing
4. **Profile and Iterate**: Use profiling tools to identify bottlenecks

### Performance Guidelines Summary
- **Shared Memory**: Use for data reuse within thread blocks
- **Constant Memory**: Use for read-only data accessed uniformly
- **Warp Shuffle**: Use for simple inter-thread communication
- **Memory Coalescing**: Ensure consecutive threads access consecutive memory
- **Bank Conflicts**: Avoid or minimize through padding and access pattern optimization

### Best Practices Checklist
- [ ] Minimize global memory accesses through caching
- [ ] Use shared memory for frequently accessed data
- [ ] Avoid bank conflicts in shared memory
- [ ] Ensure coalesced global memory access
- [ ] Use constant memory for uniform read-only data
- [ ] Leverage warp shuffle for simple reductions
- [ ] Profile memory bandwidth utilization
- [ ] Validate correctness with unit tests
- [ ] Benchmark performance improvements

### Further Reading and Resources
- CUDA Programming Guide: Memory Hierarchy
- CUDA Best Practices Guide: Memory Optimization
- GPU Architecture Documentation
- Profiling Tools: nvprof, Nsight Compute
- Community Resources: CUDA Zone, Stack Overflow

---

*This completes the comprehensive teaching material for CUDA Shared Memory and Constant Memory programming. The material covers all specified learning objectives with practical examples, optimization techniques, and best practices for high-performance GPU programming.*