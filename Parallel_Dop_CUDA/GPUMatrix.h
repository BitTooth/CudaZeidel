////////////////////////////////////////////////////////////////////////////////////////////////////
// GPUMatrix.h
#ifndef GPUMATRIX_H
#define GPUMATRIX_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_DIM 16
#define EPS 0.0000001

__global__ void validateRows (float* matrix, int size, int width, int row, int* newRow);
__global__ void swapRows (float* matrix, int width, int row1, int row2);
__global__ void setIdentity (float* matrix, int size);
__global__ void pass (float* matrix, int size, int width, int row);
__global__ void normalizeRow (float* matrix, int width, int row);

#ifdef __cplusplus
extern "C" {
#endif
// Вычисление обратной матрицы с использованием GPU
int GPUInverse(float* inMatrix, float* outMatrix, int size);
//// Решение системы линейных уравнений с использованием GPU
//int GPUSloveSystemOfLinearEquations(float* inMatrix, float* outResult, int size);
//// Вычисление детерминанта квадратной матрицы с использованием  GPU
//int GPUDeterminant(float* inMatrix, int size, float* determinant);
#ifdef __cplusplus
}
#endif
#endif