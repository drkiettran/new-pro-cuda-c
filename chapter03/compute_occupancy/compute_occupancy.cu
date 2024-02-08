
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <tclap/CmdLine.h>

void proc_args(int argc, char** argv) {
	try {
		TCLAP::CmdLine cmd("MyProgram - A sample C++ program", ' ', "1.0");

		TCLAP::ValueArg<std::string> inputArg("i", "input", "Input file", true, "", "string");
		cmd.add(inputArg);

		TCLAP::ValueArg<int> countArg("c", "count", "Number of items", false, 100, "int");
		cmd.add(countArg);

		cmd.parse(argc, argv);

		std::string input = inputArg.getValue();
		int count = countArg.getValue();

		// Process input and count arguments
		std::cout << "Input file: " << input << std::endl;
		std::cout << "Count: " << count << std::endl;
	}
	catch (TCLAP::ArgException& e) {
		std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
	}
}

int get_num_blocks(int argc, char** argv) {
	TCLAP::CmdLine cmd("Getting number of blocks", ' ', "1.0");

	TCLAP::ValueArg<int> nbArg("n", "numBlocks", "Number of blocks", false, 32, "int");
	cmd.add(nbArg);

	cmd.parse(argc, argv);

	return nbArg.getValue();
}

// Device code
__global__ void MyKernel(int* d, int* a, int* b)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d[idx] = a[idx] * b[idx];
}

// Host code
int main(int argc, char** argv)
{
	// proc_args(argc, argv);

	int numBlocks; // Occupancy in terms of active blocks
	int blockSize = get_num_blocks(argc, argv);

	std::cout << "Computing occupancy ... for a block size of: " << blockSize << std::endl;

	// These variables are used to convert occupancy to warps
	int device;
	cudaDeviceProp prop;
	int activeWarps;
	int maxWarps;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocks,
		MyKernel,
		blockSize,
		0);
	activeWarps = numBlocks * blockSize / prop.warpSize;
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
	std::cout << std::endl;
	std::cout << "Max Threads Per SM (fixed): " << prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Warp size (fixed): " << prop.warpSize << std::endl;
	std::cout << "Max Warps Per SM (fixed): " << maxWarps << std::endl;
	std::cout << "Block size: " << blockSize << std::endl;
	std::cout << "Max Active Blocks Per SM: " << numBlocks << std::endl;
	std::cout << "Active warps: " << activeWarps << std::endl;
	std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

	return 0;
}