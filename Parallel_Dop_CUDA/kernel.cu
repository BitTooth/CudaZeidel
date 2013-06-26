
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <stdio.h>
#include "stdlib.h"

#include "windows.h"
#include "GPUMatrix.h"
#include <math.h>

#define V2D(i, j) (i) * size + (j)
#define BLOCK_SIZE 256

cudaError_t launchCuda(int *c, const int *a, const int *b, size_t size);
void algorithm(int *c, const int *a, const int *b, size_t size);


// Globals

int initTime = 0;
bool g_Bl1_GPU = true;
bool g_Bl2_GPU = true;
bool g_Bl3_GPU = true;


int (*pInverse)(float*, float*, int, int);

/////////////////////////////////////////////////////////////////////////////////////////////
//									 CPU KERNELS										   //
/////////////////////////////////////////////////////////////////////////////////////////////

__host__ void Bl1_CPU(float *X_new, float *X_old, float *A, int j, int size)
{
    for (int i = j - 1; i < size; ++i)
	{
		X_new[i] = X_new[i] - A[i * size + j]*X_old[j];
	}
}

__host__ void Bl2_CPU(float *X_new, float *A, int t, int size)
{
	for (int j = max(1, t - size); j < (t - 1)/2; ++j)
	{
		int i = t - j - 1;
		X_new[i] = X_new[i] - A[i * size + j]*X_new[j];
	}
}

void helpBl_CPU(float *X, float *B)
{
	int i = 0;// threadIdx.x;
	X[i] = B[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA KERNELS										   //
/////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Bl1(float *X_new, float *X_old, float *A, int _j, int size, int stride)
{
	int j = blockIdx.y * blockDim.y + _j;
    int i = blockIdx.x * blockDim.x + (_j - 1) + threadIdx.x;
    X_new[i] = X_new[i] - A[i * stride + j]*X_old[j];
}

__global__ void Bl2(float *X_new, float *A, int t, int size, int stride)
{
	int j = ((t - size) < 1)? 1: (t - size) + threadIdx.x;

	int i = blockIdx.x * blockDim.x  + t - j;
	j += blockIdx.y * blockDim.y;

	X_new[i] = X_new[i] - A[i * stride + j]*X_new[j];
}

__global__ void helpBl(float *X, float *B)
{
	int i = threadIdx.x;
	X[i] = B[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////
//							CUDA MATRIX-VECTOR MULTIPLICATION							   //
/////////////////////////////////////////////////////////////////////////////////////////////
__global__ void MVKernel_gm(float* A, float* X, float* Y, int size)
{
	int bx = blockIdx.x; 
	int tx = threadIdx.x; 
	int Row = bx * blockDim.x + tx;
	float Pvalue = 0;
   
	for (unsigned int k = 0; k < size; k++) 
		{
			if(Row < size)         
			Pvalue += A[Row * size + k] * X[k];		
		}
	
	__syncthreads();
	
	if(Row < size)  		
	  Y[Row] = Pvalue;
	__syncthreads();
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA ZEIDEL ALGO									   //
/////////////////////////////////////////////////////////////////////////////////////////////
/// r - number of blocks
/// K - number of iterations
/// A, B, X - parts of linear system
/// size - size of system
cudaError_t algorithm(const int &r, const int &K, float *A, float *A_inv, float *B, const size_t &size, float *X)
{
	cudaError_t cudaStatus;
	float *X_new;
	float *X_old;

	float *test = new float[size];
	float *test1 = new float[size];

	cudaMalloc((void**)&X_new, size * sizeof(float));
	cudaMalloc((void**)&X_old, size * sizeof(float));

	cudaMemcpy(X_old, B, size * sizeof(float), cudaMemcpyDeviceToDevice);


	// Additional memory for CPU
	float *h_X_new = (float*)malloc(size * sizeof(float));
	float *h_X_old = (float*)malloc(size * sizeof(float));
	float *h_A = (float*)malloc(size * size * sizeof(float));
	cudaMemcpy(h_X_new, X_new, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_X_old, X_old, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_A, A, size * size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < K; ++i)
	{
		printf("%i", i);

		cudaMemcpy(X_new, B, size * sizeof(float), cudaMemcpyDeviceToDevice);

		cudaMemcpy(h_X_new, X_new, size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_X_old, X_old, size * sizeof(float), cudaMemcpyDeviceToHost);

		int n = size / ((g_Bl1_GPU)?r:1);

		for (int j = 1; j < n; ++j)
		{
			if (g_Bl1_GPU)
			{
				int num = n - j;
				if (num > 1024)
					fprintf(stderr, "\n\t !Number of thread in block is greater than 1024 (Block1)!\n");
				dim3 numBlocks(r, r);
				Bl1<<<numBlocks, num>>>(X_new, X_old, A, j, n, size);
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "\n !cudaDeviceSynchronize returned error code %d after launching Block1!\n");
					return cudaStatus;
				}
			}
			else
			{
				Bl1_CPU(h_X_new, h_X_old, h_A, j, size);
			}
		}

		if (!g_Bl1_GPU)
			cudaMemcpy(X_new, h_X_new, size * sizeof(float), cudaMemcpyHostToDevice);

		n = size / ((g_Bl2_GPU)?r:1);

		for (int t = 2; t < 2*n; ++t)
		{
			if (g_Bl2_GPU)
			{
				int num = ((t - 1) / 2) - max(1, t - n) + 1;
				if (t < 3 || t == 2*n - 1)
					continue;
				if (num > 1024)
					fprintf(stderr, "\n\t !Number of thread in block is greater than 1024(Block2)!\n");
				if (num < 0)
					fprintf(stderr, "\n\t !Number of thread in block is less than 1(Block2)! t = %d\n", t);

				dim3 numBlocks(r, r);
				Bl2<<<numBlocks, num>>>(X_new, A, t, n, size);
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "\n\t !cudaDeviceSynchronize returned error code %d after launching Block2!\n", cudaStatus);
					return cudaStatus;
				}
			}
			else
			{
				Bl2_CPU(h_X_new, h_A, t, size);
			}
			
			if ((t / 2) % 2 == 0)
			{
				// Multiply matrix by vector
				if (!g_Bl2_GPU)
					cudaMemcpy(X_new, h_X_new, size * sizeof(float), cudaMemcpyHostToDevice);

				dim3 dimGrid((size - 1) / BLOCK_SIZE + 1);
				dim3 dimBlock(BLOCK_SIZE);
				MVKernel_gm<<<dimGrid,dimBlock>>>(A, X_new, X_new, size);

				if (!g_Bl2_GPU)
					cudaMemcpy(h_X_new, X_new, size * sizeof(float), cudaMemcpyDeviceToHost);
			}
		}

		if (!g_Bl2_GPU)
			cudaMemcpy(X_new, h_X_new, size * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(X_old, X_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	
	cudaMemcpy(X, X_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
}

inline float randf(float min, float max)
{
	return min + (float)rand()/((float)RAND_MAX/(max-min));
}

void GenerateEquation(const int &size, float *A, float *B)
{
	float min = -1000.f;
	float max = 1000.f;
	
	srand((unsigned)time(0));
	for(int i = 0; i < size; ++i)
	{
		float absSum = 0.f;
		
		for(int j = 0; j < size; ++j)
		{
			A[i*size + j] = randf(min, max);
			absSum += fabsf(A[i*size + j]);
		}

		absSum -= fabsf(A[i*size + i]);
		A[i*size + i] = absSum + randf(5.f, max);
		if(randf(0.f, 1.f) < 0.5f)
		{
			A[i*size + i] *= -1.f;
		}

		B[i] = randf(min, max);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA TESTS  										   //
/////////////////////////////////////////////////////////////////////////////////////////////

void TestFunc()
{
	printf("blah-blah\n");
}
int cudaTest(const int &size, float &time, float *answer)
{
	srand((unsigned)std::time(NULL));
	time = initTime;
	for (int i = 0; i < size; ++i)
		answer[i] = rand();
	return 0;
}
void cudaSetInitTime(int t)
{
	initTime = t;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA SETTINGS										   //
/////////////////////////////////////////////////////////////////////////////////////////////

void setProcessingUnit(bool Bl1_GPU, bool Bl2_GPU, bool Bl3_GPU)
{
	g_Bl1_GPU = Bl1_GPU;
	g_Bl2_GPU = Bl2_GPU;
	g_Bl3_GPU = Bl3_GPU;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA ROUTIN  										   //
/////////////////////////////////////////////////////////////////////////////////////////////

// Helper function for using CUDA to add vectors in parallel.
cudaError_t launchCuda(const int &size, const int &r, float &time, float *answer)
{
    float *dev_a = 0;
	float *dev_a_inv = 0;
    float *dev_b = 0;
    float *dev_x = 0;

	float *A = (float*)(malloc(size * size * sizeof(float)));
	float *A_inv = (float*)(malloc(size * size * sizeof(float)));
	float *B = (float*)(malloc(size * sizeof(float)));

	GenerateEquation(size, A, B);
	GPUInverse(A, A_inv, size);


    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(float));
	cudaStatus = cudaMalloc((void**)&dev_a_inv, size * size * sizeof(float));
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(float));
	
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, A, size * size * sizeof(float),	cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_a_inv, A_inv, size * size * sizeof(float),	cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, B, size * sizeof(float),			cudaMemcpyHostToDevice);

    // Launch algorithm
	int K = 10;
	printf("start\n");

	__int64 startTime;
	__int64 endTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&startTime);

    cudaStatus = algorithm(r, K, dev_a, dev_a_inv, dev_b, size, dev_x);

	QueryPerformanceCounter((LARGE_INTEGER*)&endTime);
	printf("stop\n");

	__int64 countsPerSec;
	double secPerCount;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	secPerCount = 1.0 / (double)countsPerSec;

	time = (float)((endTime - startTime) * secPerCount);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(answer, dev_x, size * sizeof(float), cudaMemcpyDeviceToHost);

Error:
	free(A);
	free(A_inv);
	free(B);
    cudaFree(dev_x);
	cudaFree(dev_a_inv);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

int cudaMain(const int &size, const int& r, float &time, float *answer)
{
    cudaError_t cudaStatus = launchCuda(size, r, time, answer);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Seidel with cuda failed!\n");
        return 1;
    }
	
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

