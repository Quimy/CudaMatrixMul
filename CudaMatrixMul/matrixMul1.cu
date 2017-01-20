/**
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
* Matrix multiplication: C = A * B.
* Host code.
*
* This sample implements matrix multiplication as described in Chapter 3
* of the programming guide.
* It has been written for clarity of exposition to illustrate various CUDA
* programming principles, not with the goal of providing the most
* performant generic kernel for matrix multiplication.
*
* See also:
* V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
* in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
* Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
*/

// System includes
#define WIN32
#include <stdio.h>
#include <assert.h>
#include <cmath>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* wA is A's width and wB is B's width
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA)
{
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	float C_local = 0;
	
	for (int k = 0; k < wA; ++k) 
		C_local += A[by*wA*BLOCK_SIZE+ty*wA+k] * B[k*wA+bx*BLOCK_SIZE+tx];
	C[by*wA*BLOCK_SIZE + ty*wA+ bx*BLOCK_SIZE + tx] =C_local;

}

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = dimsA.x * dimsA.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	// Allocate host matrix C
	dim3 dimsC(dimsA.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = (float *)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsA.x/block_size, dimsA.x / block_size, 1);


	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}


	matrixMulCUDA<32> <<< grid, threads >> >(d_C, d_A, d_B, dimsA.x);

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal;
	double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsA.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
		gigaFlops,
		msecPerMatrixMul,
		flopsPerMatrixMul,
		threads.x * threads.y);

	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("Checking computed result for correctness: ");
	bool correct = true;
	for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
	{
		if (fabs(h_C[i] - (dimsA.x * valB)) > 1e-5)
		{
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-5\n", i, h_C[i], dimsA.x*valB);
			correct = false;
			break;
		}
	}

	printf("%s\n", correct ? "OK" : "FAIL");

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("\nNote: For peak performance, please refer to the matrixMulCUBLAS example.\n");

	cudaDeviceReset();

	if (correct)
	{
		return EXIT_SUCCESS;
	}
	else
	{
		return EXIT_FAILURE;
	}
}


/**
* Program main
*/
int main(int argc, char **argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	int n = 5; // wczytywane z konsoli
			   // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess) {
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited) {
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess) {
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	int block_size = 32;
	int matrixSize = n * block_size;
	dim3 dimsA(matrixSize, matrixSize, 1);

	int matrix_result = matrixMultiply(argc, argv, block_size, dimsA);

	exit(matrix_result);
}
