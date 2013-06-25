// Kernels/validateRows_kernel.cu
// Проверка строк (что на диагонали нет нулей).
#include "../GPUMatrix.h"

__global__ void validateRows (float* matrix, int size, int width, int row, int* newRow)
{
	// Получение текущей позиции
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Проверка границ
	if( (xIndex != 0) || (yIndex != 0))
		return;

	newRow[0] = row;
	int index = row * width + row;

	if ((matrix[index] > - EPS) && (matrix[index] < EPS))
	{
		newRow[0] = -1;
		for (int i = row + 1; i < size; i++) 
		{
			int alt_index = i * width + row;
			if ((matrix[alt_index] <= - EPS) || (matrix[alt_index] >= EPS)) 
			{
				newRow[0] = i;
				return;
			}
		}
	} 
}