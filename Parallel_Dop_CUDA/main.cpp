#include "main.h"
#include <iostream>
using namespace std;


__declspec(dllexport) int CudaStartZeidel(const int &size, float &time, float *answer)
{
	cudaMain(size, time, answer);
	return 0;
}

__declspec(dllexport) int CudaTestZeidel(const int &size, float &time, float *answer)
{
	cudaTest(size, time, answer);
	return 0;
}

__declspec(dllexport) void CudaSetInitialTime(int t)
{
	cudaSetInitTime(t);
}

__declspec(dllexport) void SetProcessingUnit(bool Bl1_GPU, bool Bl2_GPU, bool Bl3_GPU)
{
	setProcessingUnit(Bl1_GPU, Bl2_GPU, Bl3_GPU);
}