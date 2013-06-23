
__declspec(dllexport) int CudaStartZeidel(const int &size, const int& r, float &time, float *answer);
__declspec(dllexport) int CudaTestZeidel(const int &size, float &time, float *answer);

/// Setting processing unit for each block of algorithm.
/// true means that block will be processed on GPU. Default is true
__declspec(dllexport) void SetProcessingUnit(bool Bl1_GPU, bool Bl2_GPU, bool Bl3_GPU);
__declspec(dllexport) void CudaSetInitialTime(int t);