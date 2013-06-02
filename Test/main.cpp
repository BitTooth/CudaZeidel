#include <iostream>
#include "CudaZeidel.h"
using namespace std;

int main()
{
	int size = 10000;
	float time;
	float *answer = new float[size];

	// SetProcessingUnit(false, false, false);
	// 
	// CudaStartZeidel(size, time, answer);
	// cout<<"time on CPU: "<<time<<endl;

	SetProcessingUnit(true, true, true);

	CudaStartZeidel(size, time, answer);
	cout<<"time on GPU: "<<time<<endl;

	system("PAUSE");
}