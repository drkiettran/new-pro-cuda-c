/*
* From Chapter 16 - Appendix B: Examples of prune
* Example 16.1 - Prune dense to sparse
* 
* How to compile (assume cuda is installed at /usr/local/cuda/)
* nvcc -c -I/usr/local/cuda/include prunedense_example.cpp
* g++ -o prunedense_example.cpp prunedense_example.o -L/usr/local/cuda/lib64 -lcusparse -lcudart
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>

void printMatrix(int m, int n, const float* A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			float Areg = A[row + col * lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}
void printCsr(
	int m,
	int n,
	int nnz,
	const cusparseMatDescr_t descrA,
	const float* csrValA,
	const int* csrRowPtrA,
	const int* csrColIndA,
	const char* name)
{
	const int base = (cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ONE) ?
		0 : 1;
	printf("matrix %s is %d-by-%d, nnz=%d, base=%d\n", name, m, n, nnz, base);
	for (int row = 0; row < m; row++) {
		const int start = csrRowPtrA[row] - base;
		const int end = csrRowPtrA[row + 1] - base;
		for (int colidx = start; colidx < end; colidx++) {
			const int col = csrColIndA[colidx] - base;
			const float Areg = csrValA[colidx];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}
int main(int argc, char* argv[])
{
	cusparseHandle_t handle = NULL;
	cudaStream_t stream = NULL;
	cusparseMatDescr_t descrC = NULL;
	cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	const int m = 4;
	const int n = 4;
	const int lda = m;
	/*
	* | 1 0 2 -3 |
	* | 0 4 0 0 |
	* A = | 5 0 6 7 |
	* | 0 8 0 9 |
	*
	*/
	const float A[lda * n] = { 1, 0, 5, 0, 0, 4, 0, 8, 2, 0, 6, 0, -3, 0, 7, 9 };
	int* csrRowPtrC = NULL;
	int* csrColIndC = NULL;
	float* csrValC = NULL;
	float* d_A = NULL;
	int* d_csrRowPtrC = NULL;
	int* d_csrColIndC = NULL;
	float* d_csrValC = NULL;
	size_t lworkInBytes = 0;
	char* d_work = NULL;
	int nnzC = 0;
	float threshold = 4.1; /* remove Aij <= 4.1 */
	// float threshold = 0; /* remove zeros */
	printf("example of pruneDense2csr \n");
	printf("prune |A(i,j)| <= threshold \n");
	printf("threshold = %E \n", threshold);
	printMatrix(m, n, A, lda, "A");
	/* step 1: create cusparse handle, bind a stream */
	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);
	status = cusparseCreate(&handle);
	assert(CUSPARSE_STATUS_SUCCESS == status);
	status = cusparseSetStream(handle, stream);
	assert(CUSPARSE_STATUS_SUCCESS == status);
	/* step 2: configuration of matrix C */
	status = cusparseCreateMatDescr(&descrC);
	assert(CUSPARSE_STATUS_SUCCESS == status);
	cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) * lda * n);
	cudaStat2 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int) * (m + 1));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	/* step 3: query workspace */
	cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	status = cusparseSpruneDense2csr_bufferSizeExt(
		handle,
		m,
		n,
		d_A,
		lda,
		&threshold,
		descrC,
		d_csrValC,
		d_csrRowPtrC,
		d_csrColIndC,
		&lworkInBytes);
	assert(CUSPARSE_STATUS_SUCCESS == status);
	printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);
	if (NULL != d_work) { cudaFree(d_work); }
	cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
	assert(cudaSuccess == cudaStat1);
	/* step 4: compute csrRowPtrC and nnzC */
	status = cusparseSpruneDense2csrNnz(
		handle,
		m,
		n,
		d_A,
		lda,
		&threshold,
		descrC,
		d_csrRowPtrC,
		&nnzC, /* host */
		d_work);
	assert(CUSPARSE_STATUS_SUCCESS == status);
	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat1);
	printf("nnzC = %d\n", nnzC);
	if (0 == nnzC) {
		printf("C is empty \n");
		return 0;
	}
	/* step 5: compute csrColIndC and csrValC */
	cudaStat1 = cudaMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC);
	cudaStat2 = cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	status = cusparseSpruneDense2csr(
		handle,
		m,
		n,
		d_A,
		lda,
		&threshold,
		descrC,
		d_csrValC,
		d_csrRowPtrC,
		d_csrColIndC,
		d_work);
	assert(CUSPARSE_STATUS_SUCCESS == status);
	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat1);
	/* step 6: output C */
	csrRowPtrC = (int*)malloc(sizeof(int) * (m + 1));
	csrColIndC = (int*)malloc(sizeof(int) * nnzC);
	csrValC = (float*)malloc(sizeof(float) * nnzC);
	assert(NULL != csrRowPtrC);
	assert(NULL != csrColIndC);
	assert(NULL != csrValC);
	cudaStat1 = cudaMemcpy(csrRowPtrC, d_csrRowPtrC, sizeof(int) * (m + 1),
		cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(csrColIndC, d_csrColIndC, sizeof(int) * nnzC,
		cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(csrValC, d_csrValC, sizeof(float) * nnzC,
		cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");
	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
	if (d_csrColIndC) cudaFree(d_csrColIndC);
	if (d_csrValC) cudaFree(d_csrValC);
	if (csrRowPtrC) free(csrRowPtrC);
	if (csrColIndC) free(csrColIndC);
	if (csrValC) free(csrValC);
	if (handle) cusparseDestroy(handle);
	if (stream) cudaStreamDestroy(stream);
	if (descrC) cusparseDestroyMatDescr(descrC);
	cudaDeviceReset();
	return 0;
}