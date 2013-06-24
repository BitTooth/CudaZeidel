/**
	CudaStartZeidel

	Input:	size	- size of system
			r		- number of blocks
			time	- variable for returning the time
			answer	- array for returning the answer vector

	Output: 0 if successfull
*/
__declspec(dllexport) int CudaStartZeidel(const int &size, const int& r, float &time, float *answer);

/**
	SetProcessingUnit:	
		Setting processing unit for each block of algorithm.
		True means that block will be processed on GPU. 
		Default is true.

	Input:	Bl1_GPU, Bl2_GPU - use GPU for Block1 and Block2 respectfully
			BL3_GPU - not used
*/
__declspec(dllexport) void SetProcessingUnit(bool Bl1_GPU, bool Bl2_GPU, bool Bl3_GPU);


/**
	Test functions not used in release
*/
__declspec(dllexport) int CudaTestZeidel(const int &size, float &time, float *answer);
__declspec(dllexport) void CudaSetInitialTime(int t);
