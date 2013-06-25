// Kernels/pass_kernel.cu
// ������. �������� ��� �������� � ������� (����� �������������) � 0.
#include "../GPUMatrix.h"

__global__ void pass (float* matrix, int size, int width, int row)
{
	// ��������� �����-���� � ����������� ������
	__shared__ float block[BLOCK_DIM][BLOCK_DIM ];
	__shared__ float blockRow[1][BLOCK_DIM ];

	// ��������� ������� �������
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// �������� �� �����������
	if ((xIndex < row) || (xIndex >= width) || (yIndex == row) || 
		(yIndex >= size))
			return;

	// ���������� ������� �������� 
	unsigned int index_base = yIndex * width;
	unsigned int index_row = row * width;

	// ���������� ������� ��� ������� ������ 
	unsigned int index_currentRow = index_base + row;
	unsigned int index_currentElement = index_base + xIndex;   
	
	// ���������� �������� � ������� ������ 
	unsigned int index_baseElement = index_row + xIndex; 

	// ���������� ������������ ��� ������� ������ 
	float modifier = matrix[index_currentRow];

	// �����������
	block[threadIdx.y][threadIdx.x] = matrix[index_currentElement];
	blockRow[0][threadIdx.x] = matrix[index_baseElement];
	__syncthreads();

	// ���������� �����������   
	matrix[index_currentElement] = 	block[threadIdx.y][threadIdx.x] - (blockRow[0][threadIdx.x] * modifier);
}

// ������. �������� ��� �������� � �������(����� �������������) � 0.
__global__ void pass_determinant (float* matrix, int size, int row)
{
	// ��������� �����-���� � ����������� ������
	__shared__ float block[BLOCK_DIM][BLOCK_DIM];
	__shared__ float blockRow[1][BLOCK_DIM];

	// ��������� ������� �������
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// �������� �� �����������
	if ((xIndex < row) || (xIndex >= size) || (yIndex == row) || (yIndex >= size))
		return;

	// ���������� ������� �������� 
	unsigned int index_base = yIndex * size;
	unsigned int index_row = row * size;

	// ���������� ������� ��� ������� ������ 
	unsigned int index_currentRow = index_base + row;
	unsigned int index_currentElement = index_base + xIndex; 

	// ���������� �������� � ������� ������
	unsigned int index_baseRow = index_row + row; 
	unsigned int index_baseElement = index_row + xIndex; 

	// ���������� ������������ ��� ������� ������ 
	float modifier = matrix[index_currentRow] / matrix[index_baseRow];

	// �����������
	block[threadIdx.y][threadIdx.x] = matrix[index_currentElement];
	blockRow[0][threadIdx.x] = matrix[index_baseElement];
	__syncthreads();
	
	// ���������� �����������   
	matrix[index_currentElement] = 
	block[threadIdx.y][threadIdx.x] - (blockRow[0][threadIdx.x] * modifier);
}