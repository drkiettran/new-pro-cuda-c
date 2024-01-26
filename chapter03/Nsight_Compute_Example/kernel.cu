#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define USE_DOUBLE

#ifdef USE_DOUBLE
#define MYNORM(x) normcdf(x)
typedef double2 mt;
#else
#define MYNORM(x) normcdff(x)
typedef float2 mt;
#endif

__global__ void cdf_kernel(const mt* __restrict__ in,
	mt* __restrict__ out, size_t n) {
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {

#ifndef USE_VECTOR
		out[i].x = MYNORM(in[i].x);
		out[i].y = MYNORM(in[i].y);
	}
#else
		mt a = in[i];
		a.x = MYNORM(a.x);
		a.y = MYNORM(a.y);
	}
#endif
}

const size_t ds = 1048576;
const int nTPB = 1024;
const int nB = 22;

int main() {
	mt* d_in, * d_out, * h_in = new mt[ds];
	cudaMalloc(&d_in, sizeof(mt) * ds);
	cudaMalloc(&d_out, sizeof(double2) * ds);
	
	for (size_t i = 0; i < ds; i++) {
		h_in[i].x = 2 * i;
		h_in[i].y = 2 * i + 1;
	}

	cudaMemcpy(d_in, h_in, sizeof(mt) * ds, cudaMemcpyHostToDevice);
	cdf_kernel << <nB, nTPB >> > (d_in, d_out, ds);
	cudaDeviceSynchronize();
}
