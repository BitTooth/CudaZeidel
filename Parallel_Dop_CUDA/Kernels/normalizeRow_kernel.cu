#include "../GPUMatrix.h"

// Kernels/normalizeRow_kernel.cu
// ������������ ������. �������� ������������ ������� � �������.
__global__ void normalizeRow (float* matrix, int width, int row)
{
	// ��������� �����-���� � ����������� ������
	__shared__ float block[1][16];

	// ��������� ������� �������
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// �������� �� �����������
	if ((xIndex >= width) || (yIndex != 0)) 
		return;

	// ���������� ������� ��������������� ������ ������������� ������ 
	unsigned int baseIndex = row * width;

	// ���������� ��������� �������� (�������������)
	float keyElement = matrix[baseIndex + row];

	// �����������
	block[0][threadIdx.x] = matrix[baseIndex + xIndex] / keyElement; 
	__syncthreads(); 
	// ���������� ������
	matrix[baseIndex + xIndex] = block[0][threadIdx.x]; 
}