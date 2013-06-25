// Kernels/swap_kernel.cu
// Поменять местами строки матрицы
#include "../GPUMatrix.h"

__global__ void swapRows (float* matrix, int width, int row1, int row2)
{
	// Выделение блоков в разделяемой памяти
	__shared__ float row1Block[1][BLOCK_DIM];
	__shared__ float row2Block[1][BLOCK_DIM];

	// Получение текущей позиции
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Проверка позиции
	if ((xIndex >= width) || (yIndex != 0))
		return;
	// Вычисление индексов в матрице
	unsigned int index_row1 = row1 * width + xIndex;
	unsigned int index_row2 = row2 * width + xIndex;

	// Сохранение данных в разделяемой памяти
	row1Block[0][threadIdx.x] = matrix[index_row1];
	row2Block[0][threadIdx.x] = matrix[index_row2];
	__syncthreads();

	// Обмен значений
	matrix[index_row1] = row2Block[0][threadIdx.x];
	matrix[index_row2] = row1Block[0][threadIdx.x]; 
}