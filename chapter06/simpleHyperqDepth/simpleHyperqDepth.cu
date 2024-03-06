
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
 * This example demonstrates submitting work to a CUDA stream in depth-first
 * order. Work submission in depth-first order may introduce false-dependencies
 * between unrelated tasks in different CUDA streams, limiting the parallelism
 * of a CUDA application. kernel_1, kernel_2, kernel_3, and kernel_4 simply
 * implement identical, dummy computation. Separate kernels are used to make the
 * scheduling of these kernels simpler to visualize in the Visual Profiler.
 */

#define N 300000
#define NSTREAM 4

__global__ void kernel_1()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char** argv)
{
    setbuf(stdout, NULL); // disable buffering.
    printf("Run program as follows:\n");
    printf("%s [num streams] [bigcase (0/1)]\n\n", argv[0]);

    int n_streams = NSTREAM;
    int isize = 1;
    int iblock = 1;
    int bigcase = 0;

    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    std::cout << "n_streams: " << n_streams << std::endl;
    std::cout << "bigcase: " << bigcase << std::endl;
    float elapsed_time;

    // set up max connection
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    // setenv(iname, "32", 1); UNIX ONLY. In the Debugging settings, set the environment var there
    //_putenv(strcat(iname,"=32"));

    char* ivalue = getenv(iname);
    printf("%s = %s\n", iname, ivalue);

    std::cout << iname << "=" << ivalue << std::endl;
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name, n_streams);
    CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
        deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(
        cudaStream_t));

    for (int i = 0; i < n_streams; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration
    dim3 block(iblock);
    dim3 grid(isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // record start event
    CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        kernel_1 << <grid, block, 0, streams[i] >> > ();
        kernel_2 << <grid, block, 0, streams[i] >> > ();
        kernel_3 << <grid, block, 0, streams[i] >> > ();
        kernel_4 << <grid, block, 0, streams[i] >> > ();
    }

    // record stop event
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n",
        elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0; i < n_streams; i++)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    // destroy events
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}
