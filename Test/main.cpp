#include <iostream>
#include "CudaZeidel.h"
using namespace std;

int main()
{
	int size = 10;
	float time;
	float *answer = new float[size];

	// SetProcessingUnit(true, true, true);
	// CudaStartZeidel(size, time, answer);
	// cout<<"time on CPU: "<<time<<endl;
	// delete [] answer;

	// size = 300;
	// answer = new float[size];
	// SetProcessingUnit(true, true, true);
	// CudaStartZeidel(size, 100, time, answer);
	// cout<<"time on CPU: "<<time<<endl;
	// delete [] answer;
	// 
	// size = 6000;
	// answer = new float[size];
	// SetProcessingUnit(true, true, true);
	// CudaStartZeidel(size, 100, time, answer);
	// cout<<"time on CPU: "<<time<<endl;
	// delete [] answer;

	size = 1000;
	answer = new float[size];
	SetProcessingUnit(true, true, true);
	CudaStartZeidel(size, 100, time, answer);
	cout<<"time on GPU: "<<time<<endl;
	delete [] answer;

	size = 1000;
	answer = new float[size];
	SetProcessingUnit(false, false, false);
	CudaStartZeidel(size, 100, time, answer);
	cout<<"time on CPU: "<<time<<endl;
	delete [] answer;

	// SetProcessingUnit(true, true, true);
	// 
	// CudaStartZeidel(size, time, answer);
	// cout<<"time on GPU: "<<time<<endl;

	system("PAUSE");
}