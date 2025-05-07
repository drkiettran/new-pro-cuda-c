/* File:     dot1.cu
 * Purpose:  Implement dot product on a gpu using cuda.  This version
 *           uses a basic binary tree reduction.  Assumes both 
 *           threads_per_block and blocks_per_grid are powers of 2.
 *
 * Compile:  
 *    nvcc  -arch=sm_12 -o dot1 dot1.cu 
 * Run:      ./dot1 <n> <blocks> <threads_per_block>
 *              n is the vector length
 *
 * Input:    None
 * Output:   Result of dot product of a collection of random floats
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"


/*-------------------------------------------------------------------
 * Function:    Dev_dot  (kernel)
 * Purpose:     Implement a dot product of floating point vectors
 *              using atomic operations for the global sum
 * In args:     x, y, n
 * Out arg:     z
 * Scratch:     tmp
 *
 */
__global__ void Dev_dot(float x[], float y[], float z[], float tmp[], int n) {

   /* blockDim gives the number of threads in a block */
   int t = blockDim.x * blockIdx.x + threadIdx.x;

   if (t < n) tmp[t] = x[t]*y[t];
   __syncthreads();

   
   /* This uses a tree structure to do the addtions */
   for (int stride = 1; stride < blockDim.x; stride *= 2) {
      if (t % (2*stride) == 0)
         tmp[t] += tmp[t + stride];
      __syncthreads();
   }

   /* Store the result from this cache block in z[blockIdx.x] */
   if (threadIdx.x == 0) z[blockIdx.x] = tmp[t];
}  /* Dev_dot */    


/*-------------------------------------------------------------------
 * Host code 
 */
void Get_args(int argc, char* argv[], int* n_p, int* threads_per_block_p,
      int* blocks_per_grid_p);
void Setup(int n, int blocks, float** x_h_p, float** y_h_p, float** x_d_p,
      float** y_d_p, float** z_d_p, float** tmp);
float Serial_dot(float x[], float y[], int n);
void Free_mem(float* x_h, float* y_h, float* x_d, float* y_d,
      float* z_d, float* tmp_d);
float Dot_wrapper(float x_d[], float y_d[], float z_d[],  float tmp_d[],
      int n, int blocks, int threads);

/*-------------------------------------------------------------------
 * main
 */
int main(int argc, char* argv[]) {
   int n, threads_per_block, blocks_per_grid;
   float *x_h, *y_h, dot = 0;
   float *x_d, *y_d, *z_d, *tmp_d;
   double start, finish;  /* Only used on host */

   Get_args(argc, argv, &n, &threads_per_block, &blocks_per_grid);
   Setup(n, blocks_per_grid, &x_h, &y_h, &x_d, &y_d, &z_d, &tmp_d);

   GET_TIME(start);
   dot = Dot_wrapper(x_d, y_d, z_d, tmp_d, n, blocks_per_grid, 
         threads_per_block);
   GET_TIME(finish);

   printf("The dot product as computed by cuda is: %e\n", dot);
   printf("Elapsed time for cuda = %e seconds\n", finish-start);

   GET_TIME(start)
   dot = Serial_dot(x_h, y_h, n);
   GET_TIME(finish);
   printf("The dot product as computed by cpu is: %e\n", dot);
   printf("Elapsed time for cpu = %e seconds\n", finish-start);

   Free_mem(x_h, y_h, x_d, y_d, z_d, tmp_d);

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
void Setup(int n, int blocks, float** x_h_p, float** y_h_p, float** x_d_p, 
      float** y_d_p, float** z_d_p, float** tmp_d_p) {
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
   cudaMalloc(z_d_p, blocks*sizeof(float));
   cudaMalloc(tmp_d_p, size);

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(*x_d_p, *x_h_p, size, cudaMemcpyHostToDevice);
   cudaMemcpy(*y_d_p, *y_h_p, size, cudaMemcpyHostToDevice);
}  /* Setup */

/*-------------------------------------------------------------------
 * Function:  Dot_wrapper
 * Purpose:   CPU wrapper function for GPU dot product
 * Note:      Assumes x_d, y_d have already been 
 *            allocated and initialized on device.  Also
 *            assumes z_d and tmp_d have been allocated.
 */
float Dot_wrapper(float x_d[], float y_d[], float z_d[], float tmp_d[],
      int n, int blocks, int threads) {
   int i;
   float dot = 0.0;
   float z_h[blocks];

   /* Invoke kernel */
   Dev_dot<<<blocks, threads>>>(x_d, y_d, z_d, tmp_d, n);
   cudaThreadSynchronize();

   cudaMemcpy(z_h, z_d, blocks*sizeof(float), cudaMemcpyDeviceToHost);

   for (i = 0; i < blocks; i++)
      dot += z_h[i];
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
      float* z_d, float* tmp_d) {

   /* Free device memory */
   cudaFree(x_d);
   cudaFree(y_d);
   cudaFree(z_d);
   cudaFree(tmp_d);

   /* Free host memory */
   free(x_h);
   free(y_h);

}  /* Free_mem */