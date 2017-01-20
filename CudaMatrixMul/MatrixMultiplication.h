#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
class MatrixMultiplication
{
	float *h_A;
	float *h_B;
	float *h_C;
	int vector_size;

	cudaError_t error;
	float *d_A;
	float *d_B;
	float *d_C;

	void constantInit(float* data, float val);
public:
	MatrixMultiplication(int matrixSize);
	~MatrixMultiplication();
	void initializeMatrixes();
	void allocateHostMatrixes();
	void allocateDeviceMatrixes();
	
};

