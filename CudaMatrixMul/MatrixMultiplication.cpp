#include "MatrixMultiplication.h"



MatrixMultiplication::MatrixMultiplication(int matrixSize)
{
	vector_size = matrixSize*matrixSize;
}


MatrixMultiplication::~MatrixMultiplication()
{
	delete h_A;
	delete h_B;
	delete h_C;
}

void MatrixMultiplication::initializeMatrixes()
{
	constantInit(h_A, 1.35);
	constantInit(h_B, 2.35);
}

void MatrixMultiplication::allocateHostMatrixes()
{
	int mem_size = sizeof(float) * vector_size;
	h_A = new float[mem_size];
	h_B = new float[mem_size];
	h_C = new float[mem_size];
}

void MatrixMultiplication::allocateDeviceMatrixes()
{
	int mem_size = vector_size * sizeof(float);
	error = cudaMalloc((void **)&d_A, mem_size);
	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
}

void MatrixMultiplication::constantInit(float* data, float val)
{
	for (int i = 0; i < vector_size; ++i)
	{
		data[i] = val;
	}
}
