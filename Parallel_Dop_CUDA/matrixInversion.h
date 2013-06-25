#ifndef MATRIXINV_H
#define MATRIXINV_H



#ifdef __cplusplus
   extern "C" {
#endif

/*
 * Set DOEXPORT to create the lib like
 * #define DOEXPORT
 * undef DOEXPORT to load the lib
 */
#define DOEXPORT



#ifdef DOEXPORT
	#if defined(__GNUC__)
		#define DLL_EXPORT
	#else
		#define DLL_EXPORT __declspec( dllexport )
	#endif
#else
	#if defined(__GNUC__)
		#define DLL_EXPORT
	#else
		#define DLL_EXPORT __declspec(dllimport)
	#endif
#endif

DLL_EXPORT int matrixInv (float* dataIn, 
                          float* dataOut, 
                          int size,
                          int useGPU);

#ifdef __cplusplus
   }
#endif

#endif
