/* File:     vec_add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 * Compile:  nvcc [-g] [-G] -arch=sm_21 -o vec_add vec_add.cu 
 * Run:      ./vec_add <n>
 *              n is the vector length
 *
 * Input:    None
 * Output:   Result of vector addition.  If all goes well it should
 *           be a vector consisting of n copies of n+1.
 *
 * Notes:
 * 1.  CUDA is installed on all of the machines in HR 530, HR 235, and
 *     and LS G12
 * 2.  If you get something like "nvcc: command not found" when you try
 *     to compile your program.  Type the following command
 *
 *           $ export PATH=/usr/local/cuda/bin:$PATH
 *
 *     (As usual the "$" is the shell prompt:  just type the rest 
 *     of the line.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Kernel for vector addition */
__global__ void Vec_add(float x[], float y[], float z[], int n) {
   /* blockDim.x = threads_per_block                            */
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   /* block_count*threads_per_block may be >= n */
   if (i < n) z[i] = x[i] + y[i];
}  /* Vec_add */


/* Host code */
int main(int argc, char* argv[]) {
   int n, i;
   float *h_x, *h_y, *h_z;
   float *d_x, *d_y, *d_z;
   int threads_per_block;
   int block_count;
   size_t size;

   /* Get number of components in vector */
   if (argc != 2) {
      fprintf(stderr, "usage: %s <vector order>\n", argv[0]);
      exit(0);
   }
   n = strtol(argv[1], NULL, 10);
   size = n*sizeof(float);

   /* Allocate input vectors in host memory */
   h_x = (float*) malloc(size);
   h_y = (float*) malloc(size);
   h_z = (float*) malloc(size);
   
   /* Initialize input vectors */
   for (i = 0; i < n; i++) {
      h_x[i] = i+1;
      h_y[i] = n-i;
   }

   printf("h_x = ");
   for (i = 0; i < n; i++)
      printf("%.1f ", h_x[i]);
   printf("\n");

   printf("h_y = ");
   for (i = 0; i < n; i++)
      printf("%.1f ", h_y[i]);
   printf("\n");

   /* Allocate vectors in device memory */
   cudaMalloc(&d_x, size);
   cudaMalloc(&d_y, size);
   cudaMalloc(&d_z, size);

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

   /* Define block size */
   threads_per_block = 256;

   /* Define grid size.  If we just computed n/threads_per_block */
   /* we might get fewer threads than vector components.  Using  */
   /* ceil(n/threads_per_block) guarantees at least one thread   */
   /* per vector component.  The following formula is a kludge   */
   /* since it appears that the CUDA ceil function doesn't work  */
   /* correctly.                                                 */
   block_count = (n + threads_per_block - 1)/threads_per_block;

   /* Invoke kernel using block_count blocks, each of which  */
   /* contains threads_per_block threads                     */
   Vec_add<<<block_count, threads_per_block>>>(d_x, d_y, d_z, n);

   /* Wait for the kernel to complete */
   cudaThreadSynchronize();

   /* Copy result from device memory to host memory */
   /* h_z contains the result in host memory        */
   cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

   printf("The sum is: \n");
   for (i = 0; i < n; i++)
      printf("%.1f ", h_z[i]);
   printf("\n");

   /* Free device memory */
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_z);

   /* Free host memory */
   free(h_x);
   free(h_y);
   free(h_z);

   return 0;
}  /* main */