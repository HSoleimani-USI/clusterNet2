#include <clusterKernels.cuh>
#include <math.h>

template __global__ void kTranspose<float>(const float *A, float *out, int width, int height);
template __global__ void kTranspose<int>(const int *A, int *out, int width, int height);
template __global__ void kTranspose<unsigned int>(const unsigned int *A, unsigned int *out, int width, int height);
template<typename T> __global__ void kTranspose(const T *A, T *out, int width, int height)
{
    __shared__ float block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the Matrix *tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = A[index_in];
    }

    __syncthreads();

    // write the transposed Matrix *tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        out[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

template __global__ void kFill_with<int>(int *m, const int fill_value, int size);
template __global__ void kFill_with<float>(float *m, const float fill_value, int size);
template __global__ void kFill_with<unsigned int>(unsigned int *m, const unsigned int fill_value, int size);
template __global__ void kFill_with<unsigned long long>(unsigned long long *m, const unsigned long long fill_value, int size);
template<typename T> __global__ void kFill_with(T *m, const T fill_value, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
       m[i] = fill_value;
}



template __global__ void kElementWise<0>(const float *A, const float *B,  float *out, const float scalar, int size);
template __global__ void kElementWise<1>(const float *A, const float *B,  float *out, const float scalar, int size);
template __global__ void kElementWise<2>(const float *A, const float *B,  float *out, const float scalar, int size);
template __global__ void kElementWise<3>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<4>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<5>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<6>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<7>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<8>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<9>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<10>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<11>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<12>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<13>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<14>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<15>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<16>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<17>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<18>(const float *A, const float *B, float *out, const float scalar, int size);
template<int operation> __global__ void kElementWise(const float *A, const float *B, float *out, const float scalar, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       switch(operation)
	   {
       	   case 0: out[i] = fabsf(A[i]); break;
       	   case 1: out[i] = __logf(A[i]); break;
       	   case 2: out[i] = sqrtf(A[i]); break;
       	   case 3: out[i] = powf(A[i],scalar); break;
       	   case 4: out[i] = A[i] + B[i]; break;
       	   case 5: out[i] = A[i] - B[i]; break;
       	   case 6: out[i] = fdividef(A[i], B[i]); break;
       	   case 7: out[i] = A[i] * B[i]; break;
       	   case 8: out[i] = 1.0f/(1.0f + expf(A[i])); break;
       	   case 9: out[i] = A[i]*(A[i]-1.0f); break;
       	   case 10: out[i] = A[i] > 0.0f ? A[i] : 0.0f; break;
       	   case 11: out[i] = A[i] > 0.0f ? 1.0f : 0.0f; break;
       	   case 12: out[i] = (float)(A[i] == B[i]); break;
       	   case 13: out[i] = (float)(A[i] < B[i]); break;
       	   case 14: out[i] = (float)(A[i] > B[i]); break;
       	   case 15: out[i] = (float)(A[i] >= B[i]); break;
       	   case 16: out[i] = (float)(A[i] <= B[i]); break;
       	   case 17: out[i] = (float)(A[i] != B[i]); break;
       	   case 18: out[i] = powf(A[i]-B[i],2.0f); break;
	   }
  }
}



