#include <chrono>
#include <iostream>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

std::chrono::steady_clock::time_point StartTimer();
long GetDurationInSeconds(std::chrono::steady_clock::time_point start);
long GetDurationInMilliSeconds(std::chrono::steady_clock::time_point start);
long GetDurationInMicroSeconds(std::chrono::steady_clock::time_point start);
long GetDurationInNanoSeconds(std::chrono::steady_clock::time_point start);


inline std::chrono::steady_clock::time_point StartTimer()
{
    return std::chrono::steady_clock::now();
}

inline std::chrono::steady_clock::time_point StopTimer()
{
    return std::chrono::steady_clock::now();
}

inline long GetDurationInSeconds(std::chrono::steady_clock::time_point start, 
                                std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
}

inline long GetDurationInMilliSeconds(std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

inline long GetDurationInMicroSeconds(std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

inline long GetDurationInNanoSeconds(std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}