#ifndef __MAIN_H
#define __MAIN_H
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include "CudaZeidel.h"

int cudaMain(const int &size, float &time, float *answer);
int cudaTest(const int &size, float &time, float *answer);
void cudaSetInitTime(int t);
#endif