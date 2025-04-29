# Chapter 1 - Heterogeneous Parallel Computing with CUDA
Based on the book by Cheng, et al. (2014)

Prerequisites:
- Some experience in programming in C/C++ on a Linux system.

Objectives:
- Develop programming examples for NVIDIA GPU using CUDA (**C**ompute **U**nified **D**evice **A**rchitecture)) Toolkit

## Example 1:
- Source code: `Example1/example_1.cu`

```c++

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

void proc_args(int argc, char* argv[], int* a, int* b) {

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
```
- Compile source code to executable:

```bash
# you need to be in "chapter01" directory.
nvcc -g -G -I../tclap-1.4.0-rc1/include -I../Common -diag-suppress 940 -diag-suppress 611 -diag-suppress 191 -o Example_1/example_1 Example_1/example_1.cu
```

- Run the executable:
```bash
./Example_1/example_1 -a 102 -b 99
```

- Output:
```
Processing command line arguments
102 + 99 = 201
```

- To profile and program:

```bash
pushd Example_1
nsys profile --stats=true Example_1/example_1
popd
```

- output:
```
Generated:
    /home/user/dev/new-pro-cuda-c/chapter01/Example_1/report1.nsys-rep
    /home/user/dev/new-pro-cuda-c/chapter01/Example_1/report1.sqlite
```