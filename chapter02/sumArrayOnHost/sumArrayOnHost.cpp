#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "common.h"
#include <tclap/CmdLine.h>


#define BILLION  1000.0;

/*
 * This example demonstrates a simple vector sum on the host. sumArraysOnHost
 * sequentially iterates through vector elements on the host.
 */

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }

}

/*
* Initialize an array of floats with randomly generated float values.
*/
void initialData(float* ip, int size)
{
    // generate different seed for random number
    time_t t;
    
    srand((unsigned)time(&t)+(unsigned)ip);

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
        //ip[i] = (float)i;
    }

    return;
}

/*
 * Print out the three arrays A, B, and C.
 */
void print_arrays(float* a, float* b, float* c, int size) {
    printf("\n>>>>\n");
    for (int i = 0; i < size; i++) {
        printf("%4d. %6.3f + %6.3f = %6.3f\n", i, a[i], b[i], c[i]);
    }
    printf("\n<<<<\n");
}

void getArgs(int argc, char** argv, int& n) {
    try {
        TCLAP::CmdLine cmd("MyProgram - A sample C++ program", ' ', "1.0");

        TCLAP::ValueArg<int> nArg("n", "num-elements", "Number of data elements", false, 1024, "int");
        cmd.add(nArg);
        cmd.parse(argc, argv);
        n = nArg.getValue();
    }
    catch (TCLAP::ArgException& e) {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::chrono::steady_clock::time_point begin;
    int n;
    getArgs(argc, argv, n);
    printf("Number of elements: %d\n", n);
    size_t nBytes = n * sizeof(float);

    float* h_A, * h_B, * h_C;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);

    initialData(h_A, n);
    initialData(h_B, n);
    
    begin = StartTimer();
    sumArraysOnHost(h_A, h_B, h_C, n);
    std::cout << "Sum Arrays on Host: " << GetDurationInMicroSeconds(begin, StopTimer()) << " microsecs" << std::endl;

    print_arrays(h_A, h_B, h_C, n);

    free(h_A);
    free(h_B);
    free(h_C);

    return(0);
}
