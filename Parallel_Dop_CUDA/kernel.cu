
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <stdio.h>
#include "stdlib.h"

cudaError_t launchCuda(int *c, const int *a, const int *b, size_t size);
void algorithm(int *c, const int *a, const int *b, size_t size);

int initTime = 0;

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA KERNELS										   //
/////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Bl1(float *X_new, float *X_old, float **A, int j)
{
    int i = (j - 1) + threadIdx.x;
    X_new[i] = X_new[i] - A[i][j]*X_old[j];
}

__global__ void Bl2(float *X_new, float **A, int t, int n)
{
	int j = fmax((int)1, (int)(t - n)) + threadIdx.x;
	int i = t - j;
	X_new[i] = X_new[i] - A[i][j]*X_new[j];
}

__global__ void helpBl(float *X, float *B)
{
	int i = threadIdx.x;
	X[i] = B[i];
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA ZEIDEL ALGO									   //
/////////////////////////////////////////////////////////////////////////////////////////////

void algorithm(const int &K, float **A, float *B, const size_t &size, float *X)
{
	float *X_new;
	float *X_old;

	cudaMalloc((void**)&X_new, size * sizeof(float));
	cudaMalloc((void**)&X_old, size * sizeof(float));

	cudaMemcpy(X_old, B, size * sizeof(float), cudaMemcpyDeviceToDevice);

	for (int i = 0; i < K; ++i)
	{
		cudaMemcpy(X_new, B, size * sizeof(float), cudaMemcpyDeviceToDevice);

		for (int j = 2; j < size; ++j)
		{
			int num = n - j;
			Bl1<<<1, num>>>(X_new, X_old, A, j);
		}

		for (int t = 2; t < 2*size; ++t)
		{
			int num = 2 * size - 2;
			Bl2<<<1, num>>>(X_new, A, t, size);
		}

		if ((t / 2) % 2 == 0)
		{
			// Multiply matrix by vector
		}
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
//	answer = new float[size];
	for (int i = 0; i < size; ++i)
		answer[i] = rand();
	return 0;
}

void cudaSetInitTime(int t)
{
	initTime = t;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//									CUDA ROUTIN  										   //
/////////////////////////////////////////////////////////////////////////////////////////////


int cudaMain(const int &size, float &time, float *answer)
{
    const int arraySize = 5;
    const int *a = new int[size];
    const int *b = new int[size];
    int *c = new int[size];

    // Add vectors in parallel.
    cudaError_t cudaStatus = launchCuda(c, a, b, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	for (int i = 0; i < size; ++i)
	{
		answer[i] = c[i];
	}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

bool loadInput(char *path, float **A, float *B, int &size)
{
	return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t launchCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    algorithm(dev_c, dev_a, dev_b, size);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
