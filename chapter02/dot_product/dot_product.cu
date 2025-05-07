/* File:     dot0.cu
 * Purpose:  Implement dot product on a gpu using cuda.  This version
 *           uses an implementation of atomicAddf taken from the CUDA
 *           C Programming Guide (Appendix B.11)
 *
 * Compile:  nvcc  -arch=sm_21 -o dot0 dot0.cu 
 * Run:      ./dot0 <n> <blocks> <threads_per_block>
 *              n is the vector length
 *
 * Input:    None
 * Output:   Result of dot product of a collection of random floats
 *
 * Note:
 * 1.  n should be less than or equal to block*threads_per_block.
 * 2.  This requires compute capability >= 2.0 for atomicAdd
 *     with floats.
 */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"


/*-------------------------------------------------------------------
 * Function:    Dev_dot  (kernel)
 * Purpose:     Implement a dot product of floating point vectors
 *              using atomic operations for the global sum
 * In args:     x, y, n
 * In/out arg:  dot_p 
 *
 * Note:        *dot_p should be initialized to 0 by the calling
 *              function
 */
__global__ void Dev_dot(float x[], float y[], int n, float* dot_p) {
   float tmp;
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   if (i < n) {
      tmp = x[i]*y[i];
      atomicAdd(dot_p, tmp);
   }
}  /* Dev_dot */    


/*-------------------------------------------------------------------
 * Host code 
 */
void Get_args(int argc, char* argv[], int* n_p, int* threads_per_block_p,
      int* blocks_per_grid_p);
void Setup(int n, float** x_h_p, float** y_h_p, float** x_d_p,
      float** y_d_p, float** dot_d_p);
float Serial_dot(float x[], float y[], int n);
void Free_mem(float* x_h, float* y_h, float* x_d, float* y_d,
      float* dot_d);
float Dot_wrapper(float x_d[], float y_d[], float* dot_d, int n,
      int blocks, int threads);

/*-------------------------------------------------------------------
 * main
 */
int main(int argc, char* argv[]) {
   int n, threads_per_block, blocks_per_grid;
   float *x_h, *y_h, dot = 0;
   float *x_d, *y_d, *dot_d;
   double start, finish;  /* Only used on host */

   Get_args(argc, argv, &n, &threads_per_block, &blocks_per_grid);
   Setup(n, &x_h, &y_h, &x_d, &y_d, &dot_d);

   GET_TIME(start);
   dot = Dot_wrapper(x_d, y_d, dot_d, n, blocks_per_grid, 
         threads_per_block);
   GET_TIME(finish);

   printf("The dot product as computed by cuda is: %e\n", dot);
   printf("Elapsed time for cuda = %e seconds\n", finish-start);

   GET_TIME(start)
   dot = Serial_dot(x_h, y_h, n);
   GET_TIME(finish);
   printf("The dot product as computed by cpu is: %e\n", dot);
   printf("Elapsed time for cpu = %e seconds\n", finish-start);

   Free_mem(x_h, y_h, x_d, y_d, dot_d);

   return 0;
}  /* main */


/*-------------------------------------------------------------------
 * Function:  Get_args
 * Purpose:   Get and check command line args.  If there's an error
 *            quit.
 */
void Get_args(int argc, char* argv[], int* n_p, int* threads_per_block_p,
      int* blocks_per_grid_p) {

   if (argc != 4) {
      fprintf(stderr, "usage: %s <vector order> <blocks> <threads>\n", 
            argv[0]);
      exit(0);
   }
   *n_p = strtol(argv[1], NULL, 10);
   *blocks_per_grid_p = strtol(argv[2], NULL, 10);
   *threads_per_block_p = strtol(argv[3], NULL, 10);
}  /* Get_args */


/*-------------------------------------------------------------------
 * Function:  Setup
 * Purpose:   Allocate and initialize host and device memory
 */
void Setup(int n, float** x_h_p, float** y_h_p, float** x_d_p, 
      float** y_d_p, float** dot_d_p) {
   int i;
   size_t size = n*sizeof(float);

   /* Allocate input vectors in host memory */
   *x_h_p = (float*) malloc(size);
   *y_h_p = (float*) malloc(size);
   
   /* Initialize input vectors */
   srandom(1);
   for (i = 0; i < n; i++) {
      (*x_h_p)[i] = random()/((double) RAND_MAX);
      (*y_h_p)[i] = random()/((double) RAND_MAX);
   }

   /* Allocate vectors in device memory */
   cudaMalloc(x_d_p, size);
   cudaMalloc(y_d_p, size);
   cudaMalloc(dot_d_p, sizeof(float));

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(*x_d_p, *x_h_p, size, cudaMemcpyHostToDevice);
   cudaMemcpy(*y_d_p, *y_h_p, size, cudaMemcpyHostToDevice);
}  /* Setup */

/*-------------------------------------------------------------------
 * Function:  Dot_wrapper
 * Purpose:   CPU wrapper function for GPU dot product
 * Note:      Assumes x_d, y_d, dot_d have already been 
 *            allocated on device.  Also assumes x_d and y_d
 *            have been initialized.
 */
float Dot_wrapper(float x_d[], float y_d[], float* dot_d, int n,
      int blocks, int threads) {
   float dot;

   cudaMemset(dot_d, 0, sizeof(float));

   /* Invoke kernel */
   Dev_dot<<<blocks, threads>>>(x_d, y_d, n, dot_d);
   cudaThreadSynchronize();

   cudaMemcpy(&dot, dot_d, sizeof(float), cudaMemcpyDeviceToHost);

   return dot;

}  /* Dot_wrapper */


/*-------------------------------------------------------------------
 * Function:  Serial_dot
 * Purpose:   Compute a dot product on the cpu
 */
float Serial_dot(float x[], float y[], int n) {
   int i;
   float dot = 0;

   for (i = 0; i < n; i++)
      dot += x[i]*y[i];

   return dot;
}  /* Serial_dot */

/*-------------------------------------------------------------------
 * Function:  Free_mem
 * Purpose:   Free host and device memory
 */
void Free_mem(float* x_h, float* y_h, float* x_d, float* y_d,
      float* dot_d) {

   /* Free device memory */
   cudaFree(x_d);
   cudaFree(y_d);
   cudaFree(dot_d);

   /* Free host memory */
   free(x_h);
   free(y_h);

}  /* Free_mem */