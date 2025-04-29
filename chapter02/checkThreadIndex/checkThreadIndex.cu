#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include <tclap/CmdLine.h>

//#define CHECK(call)                                                            \
//{                                                                              \
//    const cudaError_t error = call;                                            \
//    if (error != cudaSuccess)                                                  \
//    {                                                                          \
//        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
//        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
//                cudaGetErrorString(error));                                    \
//    }                                                                          \
//}

/*
 * This example helps to visualize the relationship between thread/block IDs and
 * offsets into data. For each CUDA thread, this example displays the
 * intra-block thread ID, the inter-block block ID, the global coordinate of a
 * thread, the calculated offset into input data, and the input data at that
 * offset.
 */

void printMatrix(int* C, const int nx, const int ny)
{
    int* ic = C;
    printf("\nMatrix: (%d,%d)\n", nx, ny);

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%3d", ic[ix]);

        }

        ic += nx;
        printf("\n");
    }

    printf("\n");
    return;
}

__global__ void printThreadIndex(int* A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"
        " %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
        ix, iy, idx, A[idx]);
}

void getArgs(int argc, char** argv, int& x, int& y) {
    try {
        TCLAP::CmdLine cmd("MyProgram - A sample C++ program", ' ', "1.0");

        TCLAP::ValueArg<int> xArg("x", "cols", "Number of columns", false, 8, "int");
        TCLAP::ValueArg<int> yArg("y", "rows", "Number of rows", false, 6, "int");
        cmd.add(xArg);
        cmd.add(yArg);
        cmd.parse(argc, argv);
        x = xArg.getValue();
        y = yArg.getValue();
    }
    catch (TCLAP::ArgException& e) {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    }
}

int main(int argc, char** argv)
{
    int nx, ny;
    printf("%s Starting...\n", argv[0]);
    getArgs(argc, argv, nx, ny);
    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    int* h_A;
    h_A = (int*)malloc(nBytes);

    // iniitialize host matrix with integer
    for (int i = 0; i < nxy; i++)
    {
        h_A[i] = i;
    }
    printMatrix(h_A, nx, ny);

    // malloc device memory
    int* d_MatA;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up execution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printf("block: %d, %d, %d\n", block.x, block.y, block.z);
    printf("grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
    // invoke the kernel
    printThreadIndex << <grid, block >> > (d_MatA, nx, ny);
    CHECK(cudaGetLastError());

    // free host and devide memory
    CHECK(cudaFree(d_MatA));
    free(h_A);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
