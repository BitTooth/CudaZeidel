
__declspec(dllexport) int CudaStartZeidel(const int &size, float &time, float *answer);
__declspec(dllexport) int CudaTestZeidel(const int &size, float &time, float *answer);
__declspec(dllexport) void CudaSetInitialTime(int t);