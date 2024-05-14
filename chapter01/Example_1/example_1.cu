
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <tclap/CmdLine.h>

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

__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

int proc_args(int argc, char* argv[], int* a, int* b) {

    try {
        std::cout << "Processing command line arguments" << std::endl;
        TCLAP::CmdLine cmd(argv[0], ' ', "1.0");
        TCLAP::ValueArg<int> aArg("a", "val_a", "value a", false, 1, "int");
        TCLAP::ValueArg<int> bArg("b", "val_b", "value b", true, 10, "int");
        cmd.add(aArg);
        cmd.add(bArg);
        cmd.parse(argc, argv);
        *a = aArg.getValue();
        *b = bArg.getValue();
    }
    catch (TCLAP::ArgException& e) {
        std::cerr << "error: " << e.error() << "for arg " << e.argId() << std::endl;
        exit(-1);
    }
}


int main(int argc, char* argv[]) {
    int a, b;

    proc_args(argc, argv, &a, &b);

	int c;
	int* dev_c;
	CHECK(cudaMalloc((void**)&dev_c, sizeof(int)));
	add << <1, 1 >> > (a, b, dev_c);
	CHECK(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
	printf("%d + %d = %d\n", a, b, c);
	cudaFree(dev_c);
	return 0;
}