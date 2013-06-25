/////////////////////////////////////////////////////////////////////////////////////////////////////
// GPUMatrix.cu
// ��������� ����������
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>
// CUDA �������
#include "GPUMatrix.h"
// ������ ����� ������� ��� GPU
// �������������� ���� GPU
//#include "Kernels/setIdentity_kernel.cu"
//#include "Kernels/validateRows_kernel.cu"
//#include "Kernels/swapRows_kernel.cu"
//#include "Kernels/normalizeRow_kernel.cu"
//#include "Kernels/pass_kernel.cu"


// ���������� �������� ������� � �������������� GPU
int GPUInverse(float* inMatrix, float* outMatrix, int size)
{
	// ��� ������
	int error_code = 0;

	// ����� ������ ������� (�������� + ���������).
	int width = 2 * size;

	// ����� ������ ����������� ��� �������� �������.
	unsigned int memory_size = sizeof(float) * width * size;
	float *matrix = (float *) malloc(memory_size);

	// ������������� �������
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

	//// ����� ���������� � ������������ ������������������� �� ������� ����� ��������� ��� ����������. 
	//cudaSetDevice( cutGetMaxGflopsDeviceId() );

	// ������������� ������ �� ����������
	float* d_matrix;
	int* d_row;
	cudaMalloc( (void**) &d_matrix, memory_size);
	cudaMalloc( (void**) &d_row, sizeof(int));

	// ����������� ������ �� ����������
	cudaMemcpy( d_matrix, matrix, memory_size, cudaMemcpyHostToDevice);

	// ����������� ���������� ����������.
	// ����� ����������� ��� �������
	dim3 grid(width / BLOCK_DIM, size / BLOCK_DIM); 
	// ������ ����������� ���� ����
	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1); 
	// ����� ����������� ���� ������� �������
	dim3 column_grid(size / BLOCK_DIM);
	// ����� �� ������ ��������
	dim3 single_grid(1); 
	// ����� ����������� ���� ������ ������� 
	dim3 row_grid(width / BLOCK_DIM); 
	// ��������� �����
	dim3 single_thread(1);

	// ���������� ������ ����� ������� � ���������� ����
	setIdentity<<< column_grid, threads >>>(d_matrix, size);
	cudaThreadSynchronize();

	// ��������� ���������� ��� ������ ������� ����� CPU � GPU
	int* h_row = (int*) malloc(sizeof(int)); 

	// �������� ����. ���������� ���� ����� �������.
	for (int row = 0; row < size; row ++)
	{
		// ��������, �� ����� �� 0 ������� �� ���������.
		validateRows<<< single_grid, single_thread >>> (d_matrix, size, width, row, d_row);
		cudaThreadSynchronize();

		// ����������� ���������� � GPU �� CPU
		cudaMemcpy( h_row, d_row, sizeof(int), cudaMemcpyDeviceToHost);

		// ������� ���������� �������� � ���������� ���� - ��������� �����.
		if(h_row[0] == -1) 
		{
			error_code = -1; 
			break;
		// �� ��������� ������� ������� - ����� �������� �������.
		} 
		else if (h_row[0] != row) 
		{
			// �������� ������� ��� ������ �������
			swapRows<<< row_grid, threads>>> (d_matrix, width, row, h_row[0]);
			cudaThreadSynchronize();
		}

		// ������������ ������
		normalizeRow<<< row_grid, threads>>> (d_matrix, width, row); 
		cudaThreadSynchronize();

		// ��������� ������, ������� �������� ��� � ��� ���������� 
		pass<<< grid, threads>>> (d_matrix, size, width, row); 
		cudaThreadSynchronize(); 
	}
	
	if (error_code == 0) 
	{
		// ��������� ������ ��� ���������� �����������
		float* h_matrix = (float*) malloc(memory_size);

		// ����������� ����������� � ����������� ������
		cudaMemcpy( h_matrix, d_matrix, memory_size, cudaMemcpyDeviceToHost);
		//outMatrix = (float *) malloc(sizeof(float) * size * size);

		// ���������� �����������
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


//// ������� ������� �������� ��������� � �������������� GPU
//int GPUSloveSystemOfLinearEquations(float* matrix, float* outResult, int size)
//{
//	// ��� ������
//	int error_code = 0;
//	// ����� ������ ������� (�������� + ���������).
//	int width = size + 1;
//	// ����� ������ ����������� ��� �������� �������.
//	unsigned int memory_size = sizeof(float) * width * size;
//	// ����� ���������� � ������������ ������������������� �� ������� ����� ��������� ��� ����������.
//		cudaSetDevice( cutGetMaxGflopsDeviceId() );
//
//	// ������������� �������
//	unsigned int timer = 0;
//	cutilCheckError( cutCreateTimer( &timer));
//	cutilCheckError( cutStartTimer( timer));
//
//	// ������������� ������ �� ����������
//	float* d_matrix;
//	int* d_row;
//	cutilSafeCall( cudaMalloc( (void**) &d_matrix, memory_size));
//	cutilSafeCall( cudaMalloc( (void**) &d_row, sizeof(int)));
//
//	// ����������� ������ �� ����������
//	cutilSafeCall( cudaMemcpy( d_matrix, matrix, memory_size, cudaMemcpyHostToDevice) );
//
//	// ����������� ���������� ����������.
//	// ����� ����������� ��� �������
//	dim3 grid(width / BLOCK_DIM, size / BLOCK_DIM);
//	// ������ ����������� ���� ����
//	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1); 
//	// ����� ����������� ���� ������� �������
//	dim3 column_grid(size / BLOCK_DIM); 
//	// ����� �� ������ ��������
//	dim3 single_grid(1); 
//	// ����� ����������� ���� ������ �������
//	dim3 row_grid(width / BLOCK_DIM); 
//	// ��������� �����
//	dim3 single_thread(1); 
//	// ��������� ���������� ��� ������ ������� ����� CPU � GPU 
//	int* h_row = (int*) malloc(sizeof(int)); 
//
//	// �������� ����. ���������� ���� ����� �������.
//	for (int row = 0; row < size; row ++)
//	{
//		// ��������, �� ����� �� 0 ������� �� ���������.
//		validateRows<<< single_grid, single_thread >>> (d_matrix, size, width, row, d_row);
//		cudaThreadSynchronize();
//
//		// ����������� ���������� � GPU �� CPU
//		cutilSafeCall( cudaMemcpy( h_row, d_row, sizeof(int), cudaMemcpyDeviceToHost) );
//		// ������� ���������� �������� � ���������� ���� - ��������� �����.
//		if(h_row[0] == -1) 
//		{
//			error_code = -1;
//			break;
//			// �� ��������� ������� ������� - ����� �������� �������.
//		} 
//		else if (h_row[0] != row) 
//		{
//			// �������� ������� ��� ������ �������
//			swapRows<<< row_grid, threads>>> (d_matrix, width, row, h_row[0]);
//			cudaThreadSynchronize();
//		} 
//		// ������������ ������
//		normalizeRow<<< row_grid, threads>>> (d_matrix, width, row); 
//		cudaThreadSynchronize();
//
//		// ��������� ������, ������� �������� ��� � ��� ���������� 
//		pass<<< grid, threads>>> (d_matrix, size, width, row); 
//		cudaThreadSynchronize(); 
//	}
//
//	// ��������, ������� �� ����������� ���� GPU
//	cutilCheckMsg("������ ���������� ����.");
//	cutilCheckError( cutStopTimer( timer));
//	printf( "����� ������� ������� �������� ���������: %f (ms)\n", cutGetTimerValue( timer));
//	cutilCheckError(cutDeleteTimer( timer));
//
//	if (error_code == 0) 
//	{
//		// ��������� ������ ��� ���������� �����������
//		float* h_matrix = (float*) malloc(memory_size);
//		// ����������� ����������� � ����������� ������
//		cutilSafeCall( cudaMemcpy( h_matrix, d_matrix, memory_size, cudaMemcpyDeviceToHost) );
//		// ���������� ����������� 
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
//// ���������� ������������ ���������� ������� � ��������������  GPU
//int GPUDeterminant(float* matrix, int size, float* determinant) 
//{
//	// ��� ������
//	int error_code = 0; 
//	// ���������, ������������ ���� ������������.
//	float result = 1.0; 
//	// ����� ������ ����������� ��� �������� �������.
//	unsigned int memory_size = sizeof(float) * size * size;
//
//	// ����� ���������� � ������������ ������������������� �� ������� ����� ��������� ��� ����������.
//		cudaSetDevice( cutGetMaxGflopsDeviceId() );
//	// ������������� �������
//	unsigned int timer = 0;
//	cutilCheckError( cutCreateTimer( &timer));
//	cutilCheckError( cutStartTimer( timer));
//	// ������������� ������ �� ����������
//	float* d_matrix;
//	int* d_row;
//	cutilSafeCall( cudaMalloc( (void**) &d_matrix, memory_size));
//	cutilSafeCall( cudaMalloc( (void**) &d_row, sizeof(int)));
//	// ����������� ������ �� ����������
//	cutilSafeCall( cudaMemcpy( d_matrix, matrix, memory_size, cudaMemcpyHostToDevice) );
//	// ����������� ���������� ����������.
//	// ����� ����������� ��� �������
//	dim3 grid(size / BLOCK_DIM, size / BLOCK_DIM); 
//	// ������ ����������� ���� ���� 
//	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1); 
//	// ����� ����������� ���� ������� �������
//	dim3 column_grid(size / BLOCK_DIM); 
//	// ����� �� ������ ��������
//	dim3 single_grid(1); 
//	// ����� ����������� ���� ������ �������
//	dim3 row_grid(size / BLOCK_DIM); 
//	// ��������� �����
//	dim3 single_thread(1); 
//	// ��������� ���������� ��� ������ ������� ����� CPU � GPU
//	int* h_row = (int*) malloc(sizeof(int)); 
//	// �������� ����. ���������� ���� ����� �������.
//	for (int row = 0; row < size; row ++)
//	{
//		// ��������, �� ����� �� 0 ������� �� ���������.
//		validateRows<<< single_grid, single_thread >>> (d_matrix, size, size, 
//			row, d_row);
//		cudaThreadSynchronize();
//		// ����������� ���������� � GPU �� CPU
//		cutilSafeCall( cudaMemcpy( h_row, d_row, sizeof(int), 
//			cudaMemcpyDeviceToHost) );
//		// ������� ���������� �������� � ���������� ���� - ��������� �����.
//		if(h_row[0] == -1) {
//			error_code = -1;
//			break;
//			// �� ��������� ������� ������� - ����� �������� �������.
//		} else if (h_row[0] != row) {
//			result = result * -1.0;
//			// �������� ������� ��� ������ �������
//			swapRows<<< row_grid, threads>>> (d_matrix, size, row, h_row[0]);
//			cudaThreadSynchronize();
//		} 
//		// ��������� ������, ������� �������� ��� � ��� ���������� 
//		pass_determinant<<< grid, threads>>> (d_matrix, size, row); 
//		cudaThreadSynchronize(); 
//	}
//	// ��������, ������� �� ����������� ���� GPU
//	cutilCheckMsg("������ ���������� ����.");
//	cutilCheckError( cutStopTimer( timer));
//	printf( "����� ������ ������������: %f (ms)\n", cutGetTimerValue( 
//		timer));
//	cutilCheckError(cutDeleteTimer( timer));
//	if (error_code == 0) 
//	{
//		// ��������� ������ ��� ���������� ����������� 
//		float* h_matrix = (float*) malloc(memory_size);
//		// ����������� ����������� � ����������� ������
//		cutilSafeCall( cudaMemcpy( h_matrix, d_matrix, memory_size, 
//			cudaMemcpyDeviceToHost) );
//		// ���������� �����������
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