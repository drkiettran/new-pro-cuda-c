#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_timer.h"

#include <iostream>

StopWatchInterface* timer = NULL;

float elapsedTimeInMs = 0.0f;
cudaEvent_t start, stop;

void StartCounter()
{
    sdkCreateTimer(&timer);
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    sdkStartTimer(&timer);
    checkCudaErrors(cudaEventRecord(start, 0));
}

double GetCounter()
{
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
    
    return elapsedTimeInMs;
}