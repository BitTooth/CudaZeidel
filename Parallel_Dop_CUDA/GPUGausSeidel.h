#ifndef GPUGAUSSEIDEL_H
#define GPUGAUSSEIDEL_H

#ifdef __cplusplus
   extern "C" {
#endif

#define BLOCKSIZE 16
#define BLOCKSIZEMINUS1 15

#define USELOOPUNROLLING 1  
#define AVOIDBANKCONFLICTS 0    //this just runs faster :X

int GPUGausSeidel (float* matrix, 
                   float* output, 
                   int size);

#ifdef __cplusplus
   }
#endif

#endif