// Kernels/pass_kernel.cu
// Проход. Обращает все элементы в столбце (кроме диагонального) в 0.
#include "../GPUMatrix.h"

__global__ void pass (float* matrix, int size, int width, int row)
{
	// Выделение блока-кэша в разделяемой памяти
	__shared__ float block[BLOCK_DIM][BLOCK_DIM ];
	__shared__ float blockRow[1][BLOCK_DIM ];

	// Получение текущей позиции
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Проверка на размерность
	if ((xIndex < row) || (xIndex >= width) || (yIndex == row) || 
		(yIndex >= size))
			return;

	// Вычисление базовых индексов 
	unsigned int index_base = yIndex * width;
	unsigned int index_row = row * width;

	// Вычисление идексов для текущей строки 
	unsigned int index_currentRow = index_base + row;
	unsigned int index_currentElement = index_base + xIndex;   
	
	// Вычисление индексов в базовой строке 
	unsigned int index_baseElement = index_row + xIndex; 

	// Вычисление модификатора для текущей строки 
	float modifier = matrix[index_currentRow];

	// Кэширование
	block[threadIdx.y][threadIdx.x] = matrix[index_currentElement];
	blockRow[0][threadIdx.x] = matrix[index_baseElement];
	__syncthreads();

	// Сохранение результатов   
	matrix[index_currentElement] = 	block[threadIdx.y][threadIdx.x] - (blockRow[0][threadIdx.x] * modifier);
}

// Проход. Обращает все элементы в столбце(кроме диагонального) в 0.
__global__ void pass_determinant (float* matrix, int size, int row)
{
	// Выделение блока-кэша в разделяемой памяти
	__shared__ float block[BLOCK_DIM][BLOCK_DIM];
	__shared__ float blockRow[1][BLOCK_DIM];

	// Получение текущей позиции
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Проверка на размерность
	if ((xIndex < row) || (xIndex >= size) || (yIndex == row) || (yIndex >= size))
		return;

	// Вычисление базовых индексов 
	unsigned int index_base = yIndex * size;
	unsigned int index_row = row * size;

	// Вычисление идексов для текущей строки 
	unsigned int index_currentRow = index_base + row;
	unsigned int index_currentElement = index_base + xIndex; 

	// Вычисление индексов в базовой строке
	unsigned int index_baseRow = index_row + row; 
	unsigned int index_baseElement = index_row + xIndex; 

	// Вычисление модификатора для текущей строки 
	float modifier = matrix[index_currentRow] / matrix[index_baseRow];

	// Кэширование
	block[threadIdx.y][threadIdx.x] = matrix[index_currentElement];
	blockRow[0][threadIdx.x] = matrix[index_baseElement];
	__syncthreads();
	
	// Сохранение результатов   
	matrix[index_currentElement] = 
	block[threadIdx.y][threadIdx.x] - (blockRow[0][threadIdx.x] * modifier);
}