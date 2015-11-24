#include <float.h>

#ifndef clusterKernels
#define clusterKernels
template<typename T> __global__ void kFill_with(T *m, T fill_value, int size);
#endif
