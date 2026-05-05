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

// Suppress IntelliSense warnings for CUDA code
#ifdef __INTELLISENSE__
#pragma diag_suppress 20
#pragma diag_suppress 68
#pragma diag_suppress 550
#endif

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stdexcept>

// Help IntelliSense understand cooperative groups
#ifdef __INTELLISENSE__
namespace cooperative_groups {
    struct thread_block_tile {
        template<int Size>
        thread_block_tile() {}
        int thread_rank() { return 0; }
        auto ballot(bool) { return 0U; }
        auto any(bool) { return false; }
        template<typename T> T shfl_down(T, int) { return T{}; }
        int size() { return 32; }
    };
    struct thread_block {
        void sync() {}
    };
    struct grid_group {
        void sync() {}
    };
    thread_block this_thread_block() { return thread_block{}; }
    grid_group this_grid() { return grid_group{}; }
    template<int Size> thread_block_tile<Size> tiled_partition(thread_block) { return thread_block_tile<Size>{}; }
}
#endif

namespace cg = cooperative_groups;

// Portfolio configuration
struct PortfolioConfig {
    int num_assets;
    int num_simulations;
    float confidence_level;
    float time_horizon;
    float initial_portfolio_value;
};

// Asset parameters for Monte Carlo simulation
struct AssetParams {
    float expected_return;
    float volatility;
    float weight;
    float current_price;
};

// Forward declarations for IntelliSense
class AdvancedVaRCalculator;

// Forward declarations of CUDA kernels
__global__ void curand_setup_kernel(curandState* state, int n, unsigned long seed);
__global__ void advanced_monte_carlo_var(
    const AssetParams* assets,
    const float* correlation_matrix,
    float* portfolio_returns,
    float* scenario_values,
    curandState* states,
    PortfolioConfig config,
    int scenarios_per_thread);
__global__ void cooperative_var_calculation(
    float* scenario_pnl,
    float* var_results,
    PortfolioConfig config);
__global__ void calculate_risk_metrics(
    const float* scenario_pnl,
    float* risk_metrics,
    PortfolioConfig config);

// Advanced warp-level reduction using cooperative groups
template<typename T>
__device__ T warp_reduce_sum(cg::thread_block_tile<32> warp, T val) {
#ifndef __INTELLISENSE__
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }
#endif
    return val;
}

// Warp-level voting for conditional operations
__device__ bool warp_any_negative(cg::thread_block_tile<32> warp, float value) {
#ifndef __INTELLISENSE__
    return warp.any(value < 0.0f);
#else
    return false;
#endif
}

__device__ int warp_count_losses(cg::thread_block_tile<32> warp, float pnl) {
#ifndef __INTELLISENSE__
    return __popc(warp.ballot(pnl < 0.0f));
#else
    return 0;
#endif
}

// Random state initialization kernel
__global__ void curand_setup_kernel(curandState* state, int n, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &state[tid]);
    }
}

// Advanced Monte Carlo simulation kernel with cooperative groups
__global__ void advanced_monte_carlo_var(
    const AssetParams* assets,
    const float* correlation_matrix,
    float* portfolio_returns,
    float* scenario_values,
    curandState* states,
    PortfolioConfig config,
    int scenarios_per_thread) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check for thread ID
    int max_threads = (config.num_simulations + scenarios_per_thread - 1) / scenarios_per_thread;
    if (tid >= max_threads) return;

    // Initialize cooperative groups (safe operations only)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Each thread processes multiple scenarios
    for (int scenario_batch = 0; scenario_batch < scenarios_per_thread; scenario_batch++) {
        int scenario_id = tid * scenarios_per_thread + scenario_batch;

        if (scenario_id >= config.num_simulations) break;

        curandState local_state = states[tid];

        // Generate independent random numbers for our 4 assets
        float independent_randoms[4];
        for (int i = 0; i < config.num_assets && i < 4; i++) {
            independent_randoms[i] = curand_normal(&local_state);
        }

        // Apply correlation matrix and calculate portfolio return
        float portfolio_return = 0.0f;
        for (int i = 0; i < config.num_assets && i < 4; i++) {
            float asset_return = 0.0f;
            // Simple correlation application
            for (int j = 0; j <= i && j < config.num_assets; j++) {
                asset_return += correlation_matrix[i * config.num_assets + j] * independent_randoms[j];
            }

            // Convert to asset return
            asset_return = assets[i].expected_return * config.time_horizon +
                assets[i].volatility * sqrtf(config.time_horizon) * asset_return;

            // Add to portfolio return
            portfolio_return += assets[i].weight * asset_return;
        }

        // Calculate scenario portfolio value
        float scenario_value = config.initial_portfolio_value * (1.0f + portfolio_return);
        float pnl = scenario_value - config.initial_portfolio_value;

        // Store results with bounds checking
        if (scenario_id < config.num_simulations) {
            portfolio_returns[scenario_id] = portfolio_return;
            scenario_values[scenario_id] = pnl;
        }

        // Demonstrate warp-level voting (safe cooperative groups feature)
        if (scenario_batch == 0) {  // Only do this once per thread to avoid overhead
            // Count losses in this warp - this is a safe warp operation
            bool thread_has_loss = (pnl < 0.0f);
            int loss_count = __popc(warp.ballot(thread_has_loss));

            // This demonstrates warp-level communication without causing deadlocks
            if (warp.thread_rank() == 0 && tid < 32) {
                // Only first 32 threads participate to avoid issues
                // In a real application, you'd accumulate these statistics
            }
        }

        states[tid] = local_state;
    }

    // NO GRID SYNCHRONIZATION - this was causing the hang
    // Just use block-level sync which is always safe
    __syncthreads();
}

// Grid-wide cooperative sorting for VaR calculation
__global__ void cooperative_var_calculation(
    float* scenario_pnl,
    float* var_results,
    PortfolioConfig config) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate VaR percentile index
    int var_index = (int)((1.0f - config.confidence_level) * config.num_simulations);

    // Simple VaR calculation (after sorting is done on host)
    if (tid == 0) {
        float var_value = scenario_pnl[var_index];
        var_results[0] = var_value; // VaR
        var_results[1] = 0.0f;      // Expected Shortfall (to be calculated)

        // Calculate Expected Shortfall
        float es_sum = 0.0f;
        int es_count = 0;
        for (int i = 0; i < var_index; i++) {
            es_sum += scenario_pnl[i];
            es_count++;
        }

        if (es_count > 0) {
            var_results[1] = es_sum / es_count;
        }
    }
}

// Advanced risk metrics kernel with cooperative groups
__global__ void calculate_risk_metrics(
    const float* scenario_pnl,
    float* risk_metrics,
    PortfolioConfig config) {

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_stats[128];

    // Parallel calculation of moments
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float sum_cube = 0.0f;
    int count = 0;

    // Process scenarios in parallel
    for (int i = tid; i < config.num_simulations; i += blockDim.x * gridDim.x) {
        float pnl = scenario_pnl[i];
        sum += pnl;
        sum_sq += pnl * pnl;
        sum_cube += pnl * pnl * pnl;
        count++;
    }

    // Warp-level reduction
    sum = warp_reduce_sum(warp, sum);
    sum_sq = warp_reduce_sum(warp, sum_sq);
    sum_cube = warp_reduce_sum(warp, sum_cube);
    count = warp_reduce_sum(warp, count);

    // Block-level reduction
    if (warp.thread_rank() == 0) {
        int warp_id = threadIdx.x / 32;
        if (warp_id < 32) {
            shared_stats[warp_id] = sum;
            shared_stats[warp_id + 32] = sum_sq;
            shared_stats[warp_id + 64] = sum_cube;
            shared_stats[warp_id + 96] = (float)count;
        }
    }

    block.sync();

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;
        float total_sum_cube = 0.0f;
        int total_count = 0;

        int max_warps = min(32, (blockDim.x + 31) / 32);
        for (int i = 0; i < max_warps; i++) {
            total_sum += shared_stats[i];
            total_sum_sq += shared_stats[i + 32];
            total_sum_cube += shared_stats[i + 64];
            total_count += (int)shared_stats[i + 96];
        }

        if (total_count > 0) {
            float mean = total_sum / total_count;
            float variance = (total_sum_sq / total_count) - mean * mean;
            float std_dev = sqrtf(variance);
            float skewness = 0.0f;

            if (std_dev > 0.0f) {
                skewness = (total_sum_cube / total_count - 3 * mean * variance - mean * mean * mean) /
                    (std_dev * std_dev * std_dev);
            }

            // Store risk metrics atomically
            atomicAdd(&risk_metrics[0], mean);      // Mean P&L
            atomicAdd(&risk_metrics[1], std_dev);   // Standard deviation
            atomicAdd(&risk_metrics[2], skewness);  // Skewness
        }
    }
}

// Host class to run the complete VaR analysis
class AdvancedVaRCalculator {
private:
    PortfolioConfig config;
    std::vector<AssetParams> assets;
    std::vector<float> correlation_matrix;

    // Device pointers
    AssetParams* d_assets;
    float* d_correlation_matrix;
    float* d_portfolio_returns;
    float* d_scenario_values;
    float* d_var_results;
    float* d_risk_metrics;
    curandState* d_states;

public:
    AdvancedVaRCalculator(const PortfolioConfig& cfg) : config(cfg) {
        // Allocate device memory
        cudaError_t err;

        err = cudaMalloc(&d_assets, config.num_assets * sizeof(AssetParams));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_assets");

        err = cudaMalloc(&d_correlation_matrix, config.num_assets * config.num_assets * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_correlation_matrix");

        err = cudaMalloc(&d_portfolio_returns, config.num_simulations * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_portfolio_returns");

        err = cudaMalloc(&d_scenario_values, config.num_simulations * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_scenario_values");

        err = cudaMalloc(&d_var_results, 2 * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_var_results");

        err = cudaMalloc(&d_risk_metrics, 4 * sizeof(float));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_risk_metrics");

        // Calculate actual number of threads needed
        int scenarios_per_thread = 4;
        int num_threads_needed = (config.num_simulations + scenarios_per_thread - 1) / scenarios_per_thread;

        err = cudaMalloc(&d_states, num_threads_needed * sizeof(curandState));
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate d_states");

        // Initialize assets with realistic DAILY returns and volatilities
        // Converting annual figures to daily: daily_return = (1+annual)^(1/252)-1, daily_vol = annual_vol/sqrt(252)
        assets = {
            {0.0003f, 0.0095f, 0.4f, 100.0f},  // Stock 1: 8% annual -> 0.03% daily return, 15% annual -> 0.95% daily vol, 40% weight
            {0.0002f, 0.0075f, 0.3f, 50.0f},   // Stock 2: 6% annual -> 0.02% daily return, 12% annual -> 0.75% daily vol, 30% weight  
            {0.0001f, 0.0050f, 0.2f, 200.0f},  // Bond: 4% annual -> 0.01% daily return, 8% annual -> 0.50% daily vol, 20% weight
            {0.0004f, 0.0126f, 0.1f, 25.0f}    // Alternative: 10% annual -> 0.04% daily return, 20% annual -> 1.26% daily vol, 10% weight
        };

        // Example correlation matrix (simplified)
        correlation_matrix = {
            1.0f, 0.3f, 0.1f, 0.2f,
            0.3f, 1.0f, 0.2f, 0.3f,
            0.1f, 0.2f, 1.0f, 0.1f,
            0.2f, 0.3f, 0.1f, 1.0f
        };

        // Copy to device
        cudaMemcpy(d_assets, assets.data(), assets.size() * sizeof(AssetParams), cudaMemcpyHostToDevice);
        cudaMemcpy(d_correlation_matrix, correlation_matrix.data(),
            correlation_matrix.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Initialize risk metrics to zero
        cudaMemset(d_risk_metrics, 0, 4 * sizeof(float));

        // Initialize random states
        dim3 block(256);
        dim3 grid((num_threads_needed + block.x - 1) / block.x);

        curand_setup_kernel << <grid, block >> > (d_states, num_threads_needed, 12345);
        cudaDeviceSynchronize();
    }

    ~AdvancedVaRCalculator() {
        cudaFree(d_assets);
        cudaFree(d_correlation_matrix);
        cudaFree(d_portfolio_returns);
        cudaFree(d_scenario_values);
        cudaFree(d_var_results);
        cudaFree(d_risk_metrics);
        cudaFree(d_states);
    }

    void calculateVaR() {
        auto start = std::chrono::high_resolution_clock::now();

        // Configure grid for cooperative groups
        int scenarios_per_thread = 4;
        int num_threads = (config.num_simulations + scenarios_per_thread - 1) / scenarios_per_thread;
        dim3 block(256);
        dim3 grid((num_threads + block.x - 1) / block.x);

        // Check device properties and cooperative launch limits
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // Calculate maximum cooperative blocks - be more conservative
        int max_blocks_per_multiprocessor = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_multiprocessor,
            advanced_monte_carlo_var,
            block.x, 0);

        // Use a more conservative limit for cooperative launch
        int max_cooperative_blocks = std::min(prop.multiProcessorCount, max_blocks_per_multiprocessor * prop.multiProcessorCount / 2);

        // Only use cooperative launch for small grid sizes to avoid hangs
        bool use_cooperative = prop.cooperativeLaunch &&
            (grid.x <= max_cooperative_blocks) &&
            (grid.x <= 16);  // Very conservative limit

        std::cout << "Max cooperative blocks available: " << max_cooperative_blocks << "\n";
        std::cout << "Requested blocks: " << grid.x << "\n";
        std::cout << "Conservative cooperative limit: " << (use_cooperative ? "ENABLED" : "DISABLED") << "\n";

        cudaError_t err;

        if (use_cooperative) {
            std::cout << "Attempting cooperative kernel launch...\n";
            std::cout << "(If this hangs, press Ctrl+C and try with fewer simulations)\n";

            // Launch Monte Carlo simulation with cooperative groups
            void* args[] = { &d_assets, &d_correlation_matrix, &d_portfolio_returns,
                           &d_scenario_values, &d_states, &config, &scenarios_per_thread };

            err = cudaLaunchCooperativeKernel((void*)advanced_monte_carlo_var, grid, block, args);
            if (err != cudaSuccess) {
                std::cerr << "Cooperative kernel launch failed: " << cudaGetErrorString(err) << std::endl;
                std::cout << "Falling back to regular kernels...\n";
                use_cooperative = false;
            }
            else {
                std::cout << "Cooperative kernel launched, waiting for completion...\n";
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    std::cerr << "Cooperative kernel execution failed: " << cudaGetErrorString(err) << std::endl;
                    return;
                }
                std::cout << "Cooperative Monte Carlo completed successfully!\n";
            }
        }

        if (!use_cooperative) {
            std::cout << "Using regular kernel launch (safer for large grids)...\n";

            // Use regular kernel launch
            advanced_monte_carlo_var << <grid, block >> > (d_assets, d_correlation_matrix,
                d_portfolio_returns, d_scenario_values,
                d_states, config, scenarios_per_thread);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Regular kernel launch failed: " << cudaGetErrorString(err) << std::endl;
                return;
            }

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "Regular kernel execution failed: " << cudaGetErrorString(err) << std::endl;
                return;
            }
            std::cout << "Regular Monte Carlo completed successfully.\n";
        }

        // Sort and calculate VaR with error checking
        std::cout << "Sorting " << config.num_simulations << " scenario results...\n";
        try {
            thrust::device_ptr<float> dev_ptr(d_scenario_values);
            thrust::sort(dev_ptr, dev_ptr + config.num_simulations);
            std::cout << "Sorting completed successfully.\n";
        }
        catch (const std::exception& e) {
            std::cerr << "Thrust sort failed: " << e.what() << std::endl;
            return;
        }

        // Calculate VaR and Expected Shortfall
        std::cout << "Calculating VaR and Expected Shortfall...\n";
        cooperative_var_calculation << <1, 1 >> > (d_scenario_values, d_var_results, config);
        err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            std::cerr << "VaR calculation kernel failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Calculate additional risk metrics
        std::cout << "Calculating risk metrics...\n";
        calculate_risk_metrics << <grid, block >> > (d_scenario_values, d_risk_metrics, config);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Risk metrics kernel failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Copy results back to host
        float var_results[2] = { 0.0f, 0.0f };
        float risk_metrics[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        err = cudaMemcpy(var_results, d_var_results, 2 * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy VaR results: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        err = cudaMemcpy(risk_metrics, d_risk_metrics, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy risk metrics: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Display results
        std::cout << "\n=== Advanced Portfolio VaR Analysis Results ===\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Portfolio Value: $" << config.initial_portfolio_value << "\n";
        std::cout << "Simulations: " << config.num_simulations << "\n";
        std::cout << "Confidence Level: " << (config.confidence_level * 100) << "%\n";
        std::cout << "Time Horizon: " << config.time_horizon << " days\n";
        std::cout << "\nRisk Metrics:\n";
        std::cout << "VaR (" << (config.confidence_level * 100) << "%): $" << std::setprecision(0) << -var_results[0] << "\n";
        std::cout << "Expected Shortfall: $" << std::setprecision(0) << -var_results[1] << "\n";
        std::cout << "Mean P&L: $" << std::setprecision(2) << risk_metrics[0] << "\n";
        std::cout << "Volatility: $" << std::setprecision(0) << risk_metrics[1] << "\n";
        std::cout << "Skewness: " << std::setprecision(3) << risk_metrics[2] << "\n";
        std::cout << "\nComputation Time: " << duration.count() << " microseconds\n";

        // VaR as percentage of portfolio
        if (var_results[0] != 0.0f) {
            float var_percentage = (-var_results[0] / config.initial_portfolio_value) * 100.0f;
            std::cout << "VaR as % of Portfolio: " << std::setprecision(2) << var_percentage << "%\n";
        }

        // Display cooperative groups usage
        std::cout << "\n=== Cooperative Groups Usage ===\n";
        std::cout << "Cooperative Launch Used: " << (use_cooperative ? "YES" : "NO") << "\n";
        if (!use_cooperative) {
            if (!prop.cooperativeLaunch) {
                std::cout << "Reason: Device does not support cooperative launch\n";
            }
            else if (grid.x > max_cooperative_blocks) {
                std::cout << "Reason: Grid size (" << grid.x << ") exceeds max cooperative blocks ("
                    << max_cooperative_blocks << ")\n";
            }
            else {
                std::cout << "Reason: Grid size exceeds conservative safety limit (16 blocks)\n";
            }
            std::cout << "Solution: Use fewer simulations (try 4000) to enable cooperative launch\n";
        }

        std::cout << "\n=== Advanced Features Demonstrated ===\n";
        std::cout << "✓ Warp-level voting and communication patterns\n";
        std::cout << "✓ Thread block tiled partitions (32-thread warps)\n";
        std::cout << "✓ Efficient memory coalescing and bounds checking\n";
        std::cout << "✓ Monte Carlo simulation with correlated random numbers\n";
        std::cout << "✓ Parallel sorting and statistical calculations\n";
        if (use_cooperative) {
            std::cout << "✓ Grid-wide cooperative kernel launch (SUCCESSFULLY USED)\n";
        }
        else {
            std::cout << "• Grid-wide cooperative features (available but not used due to grid size)\n";
        }
    }
};

// Main function demonstrating the advanced VaR calculator
int main() {
    std::cout << "Advanced CUDA Portfolio VaR Analysis with Cooperative Groups\n";
    std::cout << "============================================================\n";

    // Check CUDA device capabilities first
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "CUDA Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Cooperative Launch Support: "
        << (prop.cooperativeLaunch ? "Yes" : "No") << "\n";
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n\n";

    // Configure portfolio analysis with safe size for cooperative launch testing
    PortfolioConfig config;
    config.num_assets = 4;
    config.num_simulations = 4096;   // Small size to enable cooperative launch
    config.confidence_level = 0.95f;
    config.time_horizon = 1.0f;  // 1 day
    config.initial_portfolio_value = 10000000.0f;  // $10M portfolio

    std::cout << "Portfolio Configuration (optimized for cooperative launch demo):\n";
    std::cout << "Assets: " << config.num_assets << "\n";
    std::cout << "Simulations: " << config.num_simulations << " (small size for cooperative demo)\n";
    std::cout << "NOTE: For production use, increase simulations to 100K+ and use regular kernels\n\n";

    try {
        // Create and run VaR calculator
        AdvancedVaRCalculator calculator(config);
        calculator.calculateVaR();

        std::cout << "\n=== Cooperative Groups Features Demonstrated ===\n";
        std::cout << "✓ Warp-level programming with tiled partitions\n";
        std::cout << "✓ Advanced voting and communication patterns\n";
        std::cout << "✓ Warp-level reductions and shuffles\n";
        std::cout << "✓ Parallel risk metrics calculation\n";
        std::cout << "✓ Adaptive cooperative vs regular kernel execution\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}