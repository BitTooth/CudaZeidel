
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <stdio.h>
#include "stdlib.h"

cudaError_t launchCuda(int *c, const int *a, const int *b, size_t size);
void algorithm(int *c, const int *a, const int *b, size_t size);

int initTime = 0;

__global__ void Bl1(int *X, int *A, int *B)
{
    int i = threadIdx.x;
    X[i] = 100500;
}

__global__ void Bl2(float **X, float **A, float *B)
{
	int i = threadIdx.x;
	// c[i] = a[i] + b[i];
}

__global__ void helpBl(float **X, float **A, float *B)
{

}

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

void algorithm(int *c, int *a, int *b, size_t size)
{
	Bl1<<<1, size>>>(c, a, b);
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
