﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <stdio.h>
#include "stdlib.h"

#include "windows.h"

#define V2D(i, j) (i) * size + (j)

cudaError_t launchCuda(int *c, const int *a, const int *b, size_t size);
void algorithm(int *c, const int *a, const int *b, size_t size);


// Globals

int initTime = 0;
bool g_Bl1_GPU = true;
bool g_Bl2_GPU = true;
bool g_Bl3_GPU = true;

/////////////////////////////////////////////////////////////////////////////////////////////
//									 CPU KERNELS										   //
/////////////////////////////////////////////////////////////////////////////////////////////

__host__ void Bl1_CPU(float *X_new, float *X_old, float *A, int j, int size)
{
	float *h_X_new = (float*)malloc(size * sizeof(float));
	float *h_X_old = (float*)malloc(size * sizeof(float));
	float *h_A = (float*)malloc(size * size * sizeof(float));
	cudaMemcpy(h_X_new, X_new, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_X_old, X_old, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_A, A, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = j - 1; i < size; ++i)
	{
		h_X_new[i] = h_X_new[i] - h_A[i * size + j]*h_X_old[j];
	}

	cudaMemcpy(X_new, h_X_new, size * sizeof(float), cudaMemcpyHostToDevice);
	free(h_X_new);
	free(h_X_old);
	free(h_A);
}

__host__ void Bl2_CPU(float *X_new, float *A, int t, int size)
{
	float *h_X_new = (float*)malloc(size * sizeof(float));
	float *h_A = (float*)malloc(size * size * sizeof(float));
	cudaMemcpy(h_X_new, X_new, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_A, A, size * size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int j = max(1, t - size); j < (t - 1)/2; ++j)
	{
		int i = t - j - 1;
		h_X_new[i] = h_X_new[i] - h_A[i * size + j]*h_X_new[j];
	}

	cudaMemcpy(X_new, h_X_new, size * sizeof(float), cudaMemcpyHostToDevice);
	free(h_X_new);
	free(h_A);
}

void helpBl_CPU(float *X, float *B)
{
	int i = 0;// threadIdx.x;
	X[i] = B[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA KERNELS										   //
/////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Bl1(float *X_new, float *X_old, float *A, int j, int size)
{
    int i = (j - 1) + threadIdx.x;
    X_new[i] = X_new[i] - A[i * size + j]*X_old[j];
}

__global__ void Bl2(float *X_new, float *A, int t, int size)
{
	int j = ((t - size) < 1)? 1: (t - size);
	j += threadIdx.x;

	int i = t - j;

	X_new[i] = X_new[i] - A[i * size + j]*X_new[j];
}

__global__ void helpBl(float *X, float *B)
{
	int i = threadIdx.x;
	X[i] = B[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA ZEIDEL ALGO									   //
/////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t algorithm(const int &K, float *A, float *B, const size_t &size, float *X)
{
	cudaError_t cudaStatus;
	float *X_new;
	float *X_old;

	float *test = new float[size];
	float *test1 = new float[size];

	cudaMalloc((void**)&X_new, size * sizeof(float));
	cudaMalloc((void**)&X_old, size * sizeof(float));

	cudaMemcpy(X_old, B, size * sizeof(float), cudaMemcpyDeviceToDevice);

	for (int i = 0; i < K; ++i)
	{
		cudaMemcpy(X_new, B, size * sizeof(float), cudaMemcpyDeviceToDevice);

		for (int j = 1; j < size; ++j)
		{
			int num = size - j;

			// cudaMemcpy(test, X_new, size * sizeof(float), cudaMemcpyDeviceToHost);
			// cudaMemcpy(test1, X_old, size * sizeof(float), cudaMemcpyDeviceToHost);

			if (g_Bl1_GPU)
			{
				Bl1<<<1, num>>>(X_new, X_old, A, j, size);
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					return cudaStatus;
				}
			}
			else
			{
				Bl1_CPU(X_new, X_old, A, j, size);
			}

			// cudaMemcpy(test, X_new, size * sizeof(float), cudaMemcpyDeviceToHost);
			// cudaMemcpy(test1, X_old, size * sizeof(float), cudaMemcpyDeviceToHost);
		}

		for (int t = 2; t < 2*size; ++t)
		{
			int num = 2 * size - 2;

			if (g_Bl2_GPU)
			{
				Bl2<<<1, num>>>(X_new, A, t, size);
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					return cudaStatus;
				}
			}
			else
			{
				Bl2_CPU(X_new, A, t, size);
			}
			
			if ((t / 2) % 2 == 0)
			{
				// Multiply matrix by vector
			}
		}

		cudaMemcpy(X_old, X_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	
	cudaMemcpy(X, X_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
}

void GenerateEquation(const int &size, float *A, float *B)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			float random = ((i != j)? RAND_MAX * 100 : 100);
			A[i * size + j] = (float)rand() / random;
		}

		B[i] = (float)rand() / (float)(RAND_MAX / size);
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
cudaError_t launchCuda(const int &size, float &time, float *answer)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_x = 0;

	float *A = (float*)(malloc(size * size * sizeof(float)));
	float *B = (float*)(malloc(size * sizeof(float)));

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(float));
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_x, size * sizeof(float));


	// Generate matrix for algorithm
	GenerateEquation(size, A, B);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, A, size * size * sizeof(float),	cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, B, size * sizeof(float),			cudaMemcpyHostToDevice);

    // Launch algorithm
	int K = 10;
	printf("start\n");

	__int64 startTime;
	__int64 endTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&startTime);

    cudaStatus = algorithm(K, dev_a, dev_b, size, dev_x);

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
    cudaFree(dev_x);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

int cudaMain(const int &size, float &time, float *answer)
{
    cudaError_t cudaStatus = launchCuda(size, time, answer);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

