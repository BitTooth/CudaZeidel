#include "../GPUMatrix.h"

// Kernels/normalizeRow_kernel.cu
// Нормализация строки. Обращает диагональный элемент в единицу.
__global__ void normalizeRow (float* matrix, int width, int row)
{
	// Выделение блока-кэша в разделяемой памяти
	__shared__ float block[1][16];

	// Получение текущей позиции
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Проверка на размерность
	if ((xIndex >= width) || (yIndex != 0)) 
		return;

	// Вычисление индекса соотвествующего началу нормализуемой строки 
	unsigned int baseIndex = row * width;

	// Вычисление ключевого элемента (диаганального)
	float keyElement = matrix[baseIndex + row];

	// Кэширование
	block[0][threadIdx.x] = matrix[baseIndex + xIndex] / keyElement; 
	__syncthreads(); 
	// Сохранение данных
	matrix[baseIndex + xIndex] = block[0][threadIdx.x]; 
}