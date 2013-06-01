#include <iostream>
#include "CudaZeidel.h"
using namespace std;

int main()
{
	int size = 1000;
	float time;
	float *answer = new float[size];

	CudaStartZeidel(size, time, answer);

	/*for (int i =0; i < size; ++i)
	{
		cout<<answer[i]<<endl;
	}*/

	system("PAUSE");
}