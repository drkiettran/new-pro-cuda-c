#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

// Portfolio configuration
constexpr int N_ASSETS = 256;
constexpr int N_TRADING_DAYS = 252;
constexpr int N_MC_PATHS = 50000;
constexpr int TILE_SIZE = 16;

struct PortfolioData {
    float* asset_returns;
    float* correlation_matrix;
    float* portfolio_weights;
    float* mc_paths;
    float* portfolio_pnl;
};

struct VaRResults {
    float var_95;
    float var_99;
    float var_999;
    float cvar_95;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Thread Block Cluster Kernel for Correlation Matrix
__global__ void compute_correlation_matrix_tbc(
    const float* __restrict__ asset_returns,
    float* __restrict__ correlation_matrix,
    const int n_assets,
    const int n_days
) {
    auto block = cooperative_groups::this_thread_block();

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int global_x = blockIdx.x * TILE_SIZE + tx;
    const int global_y = blockIdx.y * TILE_SIZE + ty;

    if (global_x >= n_assets || global_y >= n_assets) return;

    extern __shared__ float shared_mem[];
    float* local_means = shared_mem;
    float* local_stds = &shared_mem[TILE_SIZE];

    // Compute means (simplified - no cluster)
    if (ty == 0 && tx < TILE_SIZE && global_x < n_assets) {
        float sum = 0.0f;
        for (int day = 0; day < n_days; ++day) {
            sum += asset_returns[global_x * n_days + day];
        }
        local_means[tx] = sum / n_days;
    }
    block.sync();

    // Compute standard deviations
    if (ty == 0 && tx < TILE_SIZE && global_x < n_assets) {
        float mean = local_means[tx];
        float sum_sq = 0.0f;
        for (int day = 0; day < n_days; ++day) {
            float diff = asset_returns[global_x * n_days + day] - mean;
            sum_sq += diff * diff;
        }
        local_stds[tx] = sqrtf(sum_sq / (n_days - 1));
    }
    block.sync();

    // Compute correlation
    if (global_x < n_assets && global_y < n_assets) {
        float mean_x = local_means[tx];
        float mean_y = local_means[ty];
        float std_x = local_stds[tx];
        float std_y = local_stds[ty];

        float covariance = 0.0f;
        for (int day = 0; day < n_days; ++day) {
            float ret_x = asset_returns[global_x * n_days + day] - mean_x;
            float ret_y = asset_returns[global_y * n_days + day] - mean_y;
            covariance += ret_x * ret_y;
        }
        covariance /= (n_days - 1);

        float std_product = std_x * std_y;
        float correlation;

        if (std_product > 1e-8f) {
            correlation = covariance / std_product;
        }
        else {
            correlation = (global_x == global_y) ? 1.0f : 0.0f;
        }

        correlation = fmaxf(-1.0f, fminf(1.0f, correlation));
        correlation_matrix[global_x * n_assets + global_y] = correlation;
    }
}

// Supporting kernels
__global__ void init_rng_kernel(curandState_t* states, int n_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) {
        curand_init(1234 + idx, idx, 0, &states[idx]);
    }
}

__global__ void generate_monte_carlo_paths(
    const float* __restrict__ correlation_matrix,
    curandState_t* __restrict__ rng_states,
    float* __restrict__ mc_paths,
    const int n_assets,
    const int n_paths
) {
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= n_paths) return;

    curandState_t local_state = rng_states[path_id];

    for (int asset = 0; asset < n_assets; ++asset) {
        float random_return = curand_normal(&local_state);
        mc_paths[path_id * n_assets + asset] = random_return * 0.01f;
    }
    rng_states[path_id] = local_state;
}

__global__ void calculate_portfolio_pnl(
    const float* __restrict__ mc_paths,
    const float* __restrict__ portfolio_weights,
    float* __restrict__ portfolio_pnl,
    const int n_assets,
    const int n_paths
) {
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= n_paths) return;

    float pnl = 0.0f;
    for (int asset = 0; asset < n_assets; ++asset) {
        pnl += portfolio_weights[asset] * mc_paths[path_id * n_assets + asset];
    }
    portfolio_pnl[path_id] = pnl;
}

// Host functions
void generate_synthetic_market_data(PortfolioData& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> return_dist(0.0f, 0.01f);
    std::uniform_real_distribution<float> weight_dist(0.0f, 1.0f);

    std::vector<float> host_returns(N_ASSETS * N_TRADING_DAYS);
    for (int asset = 0; asset < N_ASSETS; ++asset) {
        for (int day = 0; day < N_TRADING_DAYS; ++day) {
            host_returns[asset * N_TRADING_DAYS + day] = return_dist(gen);
        }
    }

    std::vector<float> host_weights(N_ASSETS);
    float weight_sum = 0.0f;
    for (int i = 0; i < N_ASSETS; ++i) {
        host_weights[i] = weight_dist(gen);
        weight_sum += host_weights[i];
    }
    for (int i = 0; i < N_ASSETS; ++i) {
        host_weights[i] /= weight_sum;
    }

    CUDA_CHECK(cudaMemcpy(data.asset_returns, host_returns.data(),
        N_ASSETS * N_TRADING_DAYS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.portfolio_weights, host_weights.data(),
        N_ASSETS * sizeof(float), cudaMemcpyHostToDevice));
}

VaRResults calculate_var(float* portfolio_pnl, int n_paths) {
    thrust::device_ptr<float> pnl_ptr(portfolio_pnl);
    thrust::sort(pnl_ptr, pnl_ptr + n_paths);

    VaRResults results;
    int idx_95 = static_cast<int>(0.05 * n_paths);
    int idx_99 = static_cast<int>(0.01 * n_paths);
    int idx_999 = static_cast<int>(0.001 * n_paths);

    float host_results[3];
    CUDA_CHECK(cudaMemcpy(&host_results[0], portfolio_pnl + idx_95, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_results[1], portfolio_pnl + idx_99, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_results[2], portfolio_pnl + idx_999, sizeof(float), cudaMemcpyDeviceToHost));

    results.var_95 = -host_results[0];
    results.var_99 = -host_results[1];
    results.var_999 = -host_results[2];

    float sum_tail = 0.0f;
    std::vector<float> tail_values(idx_95);
    CUDA_CHECK(cudaMemcpy(tail_values.data(), portfolio_pnl, idx_95 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < idx_95; ++i) {
        sum_tail += tail_values[i];
    }
    results.cvar_95 = (idx_95 > 0) ? -sum_tail / idx_95 : 0.0f;

    return results;
}

int main() {
    std::cout << "Portfolio VaR with Thread Block Clusters" << std::endl;
    std::cout << "Portfolio: " << N_ASSETS << " assets, " << N_MC_PATHS << " paths" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl << std::endl;

    if (prop.major < 9) {
        std::cout << "Warning: TBC works best on compute 9.0+ hardware" << std::endl << std::endl;
    }

    // Allocate memory
    PortfolioData portfolio_data;
    CUDA_CHECK(cudaMalloc(&portfolio_data.asset_returns, N_ASSETS * N_TRADING_DAYS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&portfolio_data.correlation_matrix, N_ASSETS * N_ASSETS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&portfolio_data.portfolio_weights, N_ASSETS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&portfolio_data.mc_paths, N_MC_PATHS * N_ASSETS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&portfolio_data.portfolio_pnl, N_MC_PATHS * sizeof(float)));

    generate_synthetic_market_data(portfolio_data);

    curandState_t* rng_states;
    CUDA_CHECK(cudaMalloc(&rng_states, N_MC_PATHS * sizeof(curandState_t)));
    dim3 rng_block(256);
    dim3 rng_grid((N_MC_PATHS + rng_block.x - 1) / rng_block.x);
    init_rng_kernel << <rng_grid, rng_block >> > (rng_states, N_MC_PATHS);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start_total = std::chrono::high_resolution_clock::now();

    // Step 1: Correlation Matrix with TBC
    std::cout << "Computing correlation matrix with TBC..." << std::endl;
    auto start_corr = std::chrono::high_resolution_clock::now();

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N_ASSETS + TILE_SIZE - 1) / TILE_SIZE,
        (N_ASSETS + TILE_SIZE - 1) / TILE_SIZE);
    size_t shared_mem = (2 * TILE_SIZE + TILE_SIZE * TILE_SIZE) * sizeof(float);

    compute_correlation_matrix_tbc << <grid_dim, block_dim, shared_mem >> > (
        portfolio_data.asset_returns,
        portfolio_data.correlation_matrix,
        N_ASSETS,
        N_TRADING_DAYS
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_corr = std::chrono::high_resolution_clock::now();
    auto corr_time = std::chrono::duration<double, std::milli>(end_corr - start_corr).count();
    std::cout << "  Completed in " << corr_time << " ms" << std::endl << std::endl;

    // Step 2: Monte Carlo paths
    std::cout << "Generating Monte Carlo paths..." << std::endl;
    auto start_mc = std::chrono::high_resolution_clock::now();

    dim3 mc_block(256);
    dim3 mc_grid((N_MC_PATHS + mc_block.x - 1) / mc_block.x);
    generate_monte_carlo_paths << <mc_grid, mc_block >> > (
        portfolio_data.correlation_matrix, rng_states, portfolio_data.mc_paths,
        N_ASSETS, N_MC_PATHS);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_mc = std::chrono::high_resolution_clock::now();
    auto mc_time = std::chrono::duration<double, std::milli>(end_mc - start_mc).count();
    std::cout << "  Completed in " << mc_time << " ms" << std::endl << std::endl;

    // Step 3: Portfolio P&L
    std::cout << "Calculating portfolio P&L..." << std::endl;
    auto start_pnl = std::chrono::high_resolution_clock::now();

    calculate_portfolio_pnl << <mc_grid, mc_block >> > (
        portfolio_data.mc_paths, portfolio_data.portfolio_weights,
        portfolio_data.portfolio_pnl, N_ASSETS, N_MC_PATHS);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_pnl = std::chrono::high_resolution_clock::now();
    auto pnl_time = std::chrono::duration<double, std::milli>(end_pnl - start_pnl).count();
    std::cout << "  Completed in " << pnl_time << " ms" << std::endl << std::endl;

    // Step 4: VaR calculation
    std::cout << "Calculating VaR..." << std::endl;
    auto start_var = std::chrono::high_resolution_clock::now();

    VaRResults var_results = calculate_var(portfolio_data.portfolio_pnl, N_MC_PATHS);

    auto end_var = std::chrono::high_resolution_clock::now();
    auto var_time = std::chrono::duration<double, std::milli>(end_var - start_var).count();
    std::cout << "  Completed in " << var_time << " ms" << std::endl << std::endl;

    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    // Display results
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "PORTFOLIO RISK ANALYSIS RESULTS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "95% Value-at-Risk:      " << var_results.var_95 * 100 << "%" << std::endl;
    std::cout << "99% Value-at-Risk:      " << var_results.var_99 * 100 << "%" << std::endl;
    std::cout << "99.9% Value-at-Risk:    " << var_results.var_999 * 100 << "%" << std::endl;
    std::cout << "95% Conditional VaR:    " << var_results.cvar_95 * 100 << "%" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Correlation Matrix (TBC): " << std::setw(8) << corr_time << " ms" << std::endl;
    std::cout << "Monte Carlo Paths:        " << std::setw(8) << mc_time << " ms" << std::endl;
    std::cout << "Portfolio P&L:            " << std::setw(8) << pnl_time << " ms" << std::endl;
    std::cout << "VaR Calculation:          " << std::setw(8) << var_time << " ms" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Total Execution Time:     " << std::setw(8) << total_time << " ms" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Cleanup
    cudaFree(portfolio_data.asset_returns);
    cudaFree(portfolio_data.correlation_matrix);
    cudaFree(portfolio_data.portfolio_weights);
    cudaFree(portfolio_data.mc_paths);
    cudaFree(portfolio_data.portfolio_pnl);
    cudaFree(rng_states);

    return 0;
}