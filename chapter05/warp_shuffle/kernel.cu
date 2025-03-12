
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdio.h>
__global__ void bcast(int arg) {
	int laneId = threadIdx.x & 0x1f;
	int value;
	
	if (laneId == 0) // Note unused variable for
		value = arg; // all threads except lane 0
	printf("%d. laneId: %d; value: %d\n", threadIdx.x, laneId, value);
	value = __shfl_sync(0xffffffff, value, 0); // Synchronize all threads in warp, and get "value" from lane 0
	printf("+%d. laneId: %d; value: %d\n", threadIdx.x, laneId, value);
	if (value != arg)
		printf("Thread %d failed.\n", threadIdx.x);
}
int main() {
	bcast <<<1, 32>>>(1234);
	cudaDeviceSynchronize();
	return 0;
}