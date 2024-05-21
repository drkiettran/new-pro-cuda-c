
#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include <tclap/CmdLine.h>

void getArgs(int argc, char** argv, int& n, int& b) {
    try {
        TCLAP::CmdLine cmd("MyProgram - A sample C++ program", ' ', "1.0");

        TCLAP::ValueArg<int> nArg("n", "num-elements", "Number of data elements", false, 1024, "int");
        TCLAP::ValueArg<int> bArg("b", "num-blocks", "Number of threads per block", false, 1024, "int");
        cmd.add(nArg);
        cmd.add(bArg);
        cmd.parse(argc, argv);
        n = nArg.getValue();
        b = bArg.getValue();
    }
    catch (TCLAP::ArgException& e) {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    }
}

/*
 * Demonstrate defining the dimensions of a block of threads and a grid of
 * blocks from the host.
 */

int main(int argc, char** argv)
{
    // define total data element
    int n = 1024, b = 1024;
    getArgs(argc, argv, n, b);

    std::cout << "define grid block: \n\n";

    // define grid and block structure
    dim3 block(b);
    dim3 grid((n + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);
    std::cout << "if block size is: " << block.x << ", number of blocks is: " << grid.x << std::endl;

    // reset block
    block.x = b/2;
    grid.x = (n + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);
    std::cout << "if block size is: " << block.x << ", number of blocks is: " << grid.x << std::endl;

    // reset block
    block.x = b/4;
    grid.x = (n + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);
    std::cout << "if block size is: " << block.x << ", number of blocks is: " << grid.x << std::endl;

    // reset block
    block.x = b/8;
    grid.x = (n + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);
    std::cout << "if block size is: " << block.x << ", number of blocks is: " << grid.x << std::endl;

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return(0);
}


