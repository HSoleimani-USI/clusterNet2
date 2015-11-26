#include <float.h>

#define COPY_BLOCK_SIZE 16

#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

#define PI 3.1415926535897932f

#define NUM_RND_BURNIN                      1000
#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

#ifndef clusterKernels
#define clusterKernels
template<typename T> __global__ void kFill_with(T *m, const T fill_value, int size);
template<typename T> __global__ void kTranspose(const T *A, T *out, int width, int height);
#endif
