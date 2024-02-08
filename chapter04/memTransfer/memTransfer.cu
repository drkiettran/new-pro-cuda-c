#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include <tclap/CmdLine.h>

/*
 * An example of using CUDA's memory copy API to transfer data to and from the
 * device. In this case, cudaMalloc is used to allocate memory on the GPU and
 * cudaMemcpy is used to transfer the contents of host memory to an array
 * allocated using cudaMalloc.
 */

int get_power(int argc, char** argv) {
    TCLAP::CmdLine cmd("Getting power", ' ', "1.0");

    TCLAP::ValueArg<int> powerArg("p", "power", "Power value", true, 32, "int");
    cmd.add(powerArg);

    cmd.parse(argc, argv);

    return powerArg.getValue();
}

int main(int argc, char** argv)
{
    std::cout << "Starting ..." << argv[0] << std::endl << std::endl;
    int power = get_power(argc, argv);
    std::chrono::steady_clock::time_point begin;

    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int isize = 1 << power; // 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev,
        deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));

    // allocate the host memory
    float* h_a = (float*)malloc(nbytes);

    // allocate the device memory
    float* d_a;
    begin = StartTimer();
    CHECK(cudaMalloc((float**)&d_a, nbytes));
    std::cout << "cudaMalloc: " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;

    // initialize the host memory
    for (unsigned int i = 0; i < isize; i++) h_a[i] = 0.5f;

    // transfer data from the host to the device
    begin = StartTimer();
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    std::cout << "cudaMemcpy(H2D): " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;

    // transfer data from the device to the host
    begin = StartTimer();
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    std::cout << "cudaMemcpy(D2H): " << GetDurationInMicroSeconds(begin, StopTimer()) << " mcs" << std::endl;

    // free memory
    CHECK(cudaFree(d_a));
    free(h_a);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
