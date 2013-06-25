/////////////////////////////////////////////////////////////////////////////////////////////////////
// GPUMatrix.cu
// Системные библиотеки
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>
// CUDA Утилиты
#include "GPUMatrix.h"
// Размер блока потоков для GPU
// Вычислительные ядра GPU
//#include "Kernels/setIdentity_kernel.cu"
//#include "Kernels/validateRows_kernel.cu"
//#include "Kernels/swapRows_kernel.cu"
//#include "Kernels/normalizeRow_kernel.cu"
//#include "Kernels/pass_kernel.cu"


// Вычисление обратной матрицы с использованием GPU
int GPUInverse(float* inMatrix, float* outMatrix, int size)
{
	// Код ошибки
	int error_code = 0;

	// Общая ширина матрицы (исходная + единичная).
	int width = 2 * size;

	// Объем памяти необходимый для хранения матрицы.
	unsigned int memory_size = sizeof(float) * width * size;
	float *matrix = (float *) malloc(memory_size);

	// Инициализация матрицы
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = i * width + j;
			if (j < size) 
			{ 
				int in_index = i * size + j; 
				matrix[index] = (float) inMatrix[in_index];
			} else {
				matrix[index] = (float) 0;
			}
		} 
	}

	//// Поиск устройства с максимальной производительностью на котором будут проведены все вычисления. 
	//cudaSetDevice( cutGetMaxGflopsDeviceId() );

	// Инициализация памяти на устройстве
	float* d_matrix;
	int* d_row;
	cudaMalloc( (void**) &d_matrix, memory_size);
	cudaMalloc( (void**) &d_row, sizeof(int));

	// Копирование данных на устройство
	cudaMemcpy( d_matrix, matrix, memory_size, cudaMemcpyHostToDevice);

	// Определение параметров исполнения.
	// Сетка покрывающая всю матрицу
	dim3 grid(width / BLOCK_DIM, size / BLOCK_DIM); 
	// Потоки покрывающие весь блок
	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1); 
	// Сетка покрывающая один столбец матрицы
	dim3 column_grid(size / BLOCK_DIM);
	// Сетка из одного элемента
	dim3 single_grid(1); 
	// Сетка покрывающая одну строку матрицы 
	dim3 row_grid(width / BLOCK_DIM); 
	// Единичный поток
	dim3 single_thread(1);

	// Приведение правой части матрицы к единичному виду
	setIdentity<<< column_grid, threads >>>(d_matrix, size);
	cudaThreadSynchronize();

	// Временная переменная для обмена данными между CPU и GPU
	int* h_row = (int*) malloc(sizeof(int)); 

	// Основной цикл. Приведение всех строк матрицы.
	for (int row = 0; row < size; row ++)
	{
		// Проверка, не стоит ли 0 элемент на диагонали.
		validateRows<<< single_grid, single_thread >>> (d_matrix, size, width, row, d_row);
		cudaThreadSynchronize();

		// Копирование результата с GPU на CPU
		cudaMemcpy( h_row, d_row, sizeof(int), cudaMemcpyDeviceToHost);

		// Матрицу невозможно привести к единичному виду - окончание цикла.
		if(h_row[0] == -1) 
		{
			error_code = -1; 
			break;
		// На диагонали нулевой элемент - нужно поменять местами.
		} 
		else if (h_row[0] != row) 
		{
			// Поменять местами две строки матрицы
			swapRows<<< row_grid, threads>>> (d_matrix, width, row, h_row[0]);
			cudaThreadSynchronize();
		}

		// Номализовать строку
		normalizeRow<<< row_grid, threads>>> (d_matrix, width, row); 
		cudaThreadSynchronize();

		// Выполнить проход, обнуляя элементы над и под диагональю 
		pass<<< grid, threads>>> (d_matrix, size, width, row); 
		cudaThreadSynchronize(); 
	}
	
	if (error_code == 0) 
	{
		// Выделение памяти для сохранения результатов
		float* h_matrix = (float*) malloc(memory_size);

		// Копирование результатов в оперативную память
		cudaMemcpy( h_matrix, d_matrix, memory_size, cudaMemcpyDeviceToHost);
		//outMatrix = (float *) malloc(sizeof(float) * size * size);

		// Сохранение результатов
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{ 
				int out_index = i * size + j; 
				int h_index = i * 2 * size + size + j;
				outMatrix[out_index] = h_matrix[h_index]; 
			} 
		} 
		free( h_matrix); 
	}

	free( matrix);
	free( h_row);
	cudaFree(d_matrix); 
	cudaFree(d_row);
	cudaThreadExit(); 
	return error_code;
}


//// Решение системы линейных уравнений с использованием GPU
//int GPUSloveSystemOfLinearEquations(float* matrix, float* outResult, int size)
//{
//	// Код ошибки
//	int error_code = 0;
//	// Общая ширина матрицы (исходная + единичная).
//	int width = size + 1;
//	// Объем памяти необходимый для хранения матрицы.
//	unsigned int memory_size = sizeof(float) * width * size;
//	// Поиск устройства с максимальной производительностью на котором будут проведены все вычисления.
//		cudaSetDevice( cutGetMaxGflopsDeviceId() );
//
//	// Инициализация таймера
//	unsigned int timer = 0;
//	cutilCheckError( cutCreateTimer( &timer));
//	cutilCheckError( cutStartTimer( timer));
//
//	// Инициализация памяти на устройстве
//	float* d_matrix;
//	int* d_row;
//	cutilSafeCall( cudaMalloc( (void**) &d_matrix, memory_size));
//	cutilSafeCall( cudaMalloc( (void**) &d_row, sizeof(int)));
//
//	// Копирование данных на устройство
//	cutilSafeCall( cudaMemcpy( d_matrix, matrix, memory_size, cudaMemcpyHostToDevice) );
//
//	// Определение параметров исполнения.
//	// Сетка покрывающая всю матрицу
//	dim3 grid(width / BLOCK_DIM, size / BLOCK_DIM);
//	// Потоки покрывающие весь блок
//	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1); 
//	// Сетка покрывающая один столбец матрицы
//	dim3 column_grid(size / BLOCK_DIM); 
//	// Сетка из одного элемента
//	dim3 single_grid(1); 
//	// Сетка покрывающая одну строку матрицы
//	dim3 row_grid(width / BLOCK_DIM); 
//	// Единичный поток
//	dim3 single_thread(1); 
//	// Временная переменная для обмена данными между CPU и GPU 
//	int* h_row = (int*) malloc(sizeof(int)); 
//
//	// Основной цикл. Приведение всех строк матрицы.
//	for (int row = 0; row < size; row ++)
//	{
//		// Проверка, не стоит ли 0 элемент на диагонали.
//		validateRows<<< single_grid, single_thread >>> (d_matrix, size, width, row, d_row);
//		cudaThreadSynchronize();
//
//		// Копирование результата с GPU на CPU
//		cutilSafeCall( cudaMemcpy( h_row, d_row, sizeof(int), cudaMemcpyDeviceToHost) );
//		// Матрицу невозможно привести к единичному виду - окончание цикла.
//		if(h_row[0] == -1) 
//		{
//			error_code = -1;
//			break;
//			// На диагонали нулевой элемент - нужно поменять местами.
//		} 
//		else if (h_row[0] != row) 
//		{
//			// Поменять местами две строки матрицы
//			swapRows<<< row_grid, threads>>> (d_matrix, width, row, h_row[0]);
//			cudaThreadSynchronize();
//		} 
//		// Номализовать строку
//		normalizeRow<<< row_grid, threads>>> (d_matrix, width, row); 
//		cudaThreadSynchronize();
//
//		// Выполнить проход, обнуляя элементы над и под диагональю 
//		pass<<< grid, threads>>> (d_matrix, size, width, row); 
//		cudaThreadSynchronize(); 
//	}
//
//	// Проверка, успешно ли выполнились ядра GPU
//	cutilCheckMsg("Ошибка исполнения ядра.");
//	cutilCheckError( cutStopTimer( timer));
//	printf( "Время решения системы линейных уравнений: %f (ms)\n", cutGetTimerValue( timer));
//	cutilCheckError(cutDeleteTimer( timer));
//
//	if (error_code == 0) 
//	{
//		// Выделение памяти для сохранения результатов
//		float* h_matrix = (float*) malloc(memory_size);
//		// Копирование результатов в оперативную память
//		cutilSafeCall( cudaMemcpy( h_matrix, d_matrix, memory_size, cudaMemcpyDeviceToHost) );
//		// Сохранение результатов 
//		for (int i = 0; i < size; i++)
//		{
//			int index = i * width + width - 1;
//			outResult[i] = h_matrix[index];
//		} 
//		free( h_matrix); 
//	}
//	free( matrix);
//	free( h_row);
//	cutilSafeCall(cudaFree(d_matrix));
//	cutilSafeCall(cudaFree(d_row));
//	cudaThreadExit(); 
//	return error_code;
//}
//// Вычисление детерминанта квадратной матрицы с использованием  GPU
//int GPUDeterminant(float* matrix, int size, float* determinant) 
//{
//	// Код ошибки
//	int error_code = 0; 
//	// Множитель, определяющий знак детерминанта.
//	float result = 1.0; 
//	// Объем памяти необходимый для хранения матрицы.
//	unsigned int memory_size = sizeof(float) * size * size;
//
//	// Поиск устройства с максимальной производительностью на котором будут проведены все вычисления.
//		cudaSetDevice( cutGetMaxGflopsDeviceId() );
//	// Инициализация таймера
//	unsigned int timer = 0;
//	cutilCheckError( cutCreateTimer( &timer));
//	cutilCheckError( cutStartTimer( timer));
//	// Инициализация памяти на устройстве
//	float* d_matrix;
//	int* d_row;
//	cutilSafeCall( cudaMalloc( (void**) &d_matrix, memory_size));
//	cutilSafeCall( cudaMalloc( (void**) &d_row, sizeof(int)));
//	// Копирование данных на устройство
//	cutilSafeCall( cudaMemcpy( d_matrix, matrix, memory_size, cudaMemcpyHostToDevice) );
//	// Определение параметров исполнения.
//	// Сетка покрывающая всю матрицу
//	dim3 grid(size / BLOCK_DIM, size / BLOCK_DIM); 
//	// Потоки покрывающие весь блок 
//	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1); 
//	// Сетка покрывающая один столбец матрицы
//	dim3 column_grid(size / BLOCK_DIM); 
//	// Сетка из одного элемента
//	dim3 single_grid(1); 
//	// Сетка покрывающая одну строку матрицы
//	dim3 row_grid(size / BLOCK_DIM); 
//	// Единичный поток
//	dim3 single_thread(1); 
//	// Временная переменная для обмена данными между CPU и GPU
//	int* h_row = (int*) malloc(sizeof(int)); 
//	// Основной цикл. Приведение всех строк матрицы.
//	for (int row = 0; row < size; row ++)
//	{
//		// Проверка, не стоит ли 0 элемент на диагонали.
//		validateRows<<< single_grid, single_thread >>> (d_matrix, size, size, 
//			row, d_row);
//		cudaThreadSynchronize();
//		// Копирование результата с GPU на CPU
//		cutilSafeCall( cudaMemcpy( h_row, d_row, sizeof(int), 
//			cudaMemcpyDeviceToHost) );
//		// Матрицу невозможно привести к единичному виду - окончание цикла.
//		if(h_row[0] == -1) {
//			error_code = -1;
//			break;
//			// На диагонали нулевой элемент - нужно поменять местами.
//		} else if (h_row[0] != row) {
//			result = result * -1.0;
//			// Поменять местами две строки матрицы
//			swapRows<<< row_grid, threads>>> (d_matrix, size, row, h_row[0]);
//			cudaThreadSynchronize();
//		} 
//		// Выполнить проход, обнуляя элементы над и под диагональю 
//		pass_determinant<<< grid, threads>>> (d_matrix, size, row); 
//		cudaThreadSynchronize(); 
//	}
//	// Проверка, успешно ли выполнились ядра GPU
//	cutilCheckMsg("Ошибка исполнения ядра.");
//	cutilCheckError( cutStopTimer( timer));
//	printf( "Время поиска детерминанта: %f (ms)\n", cutGetTimerValue( 
//		timer));
//	cutilCheckError(cutDeleteTimer( timer));
//	if (error_code == 0) 
//	{
//		// Выделение памяти для сохранения результатов 
//		float* h_matrix = (float*) malloc(memory_size);
//		// Копирование результатов в оперативную память
//		cutilSafeCall( cudaMemcpy( h_matrix, d_matrix, memory_size, 
//			cudaMemcpyDeviceToHost) );
//		// Сохранение результатов
//		for (int i = 0; i < size; i++)
//		{
//			int index = i * size + i; 
//			result *= h_matrix[index]; 
//		} 
//		free( h_matrix); 
//	}
//	free( matrix);
//	free( h_row);
//	cutilSafeCall(cudaFree(d_matrix));
//	cutilSafeCall(cudaFree(d_row));
//	cudaThreadExit(); 
//	*determinant = result;
//	return error_code;