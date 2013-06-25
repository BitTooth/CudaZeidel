// Kernels/setIdentity_kernel.cu
// Добавление единичной матрицы
#include "../GPUMatrix.h"

__global__ void setIdentity (float* matrix, int size)
{ 
	// Получение текущей позиции
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex >= size) || (yIndex != 0))
		return;

	// Вычисление позиции в матрице
	unsigned int index = xIndex * 2 * size + size + xIndex;
	matrix[index] = 1; 
}
