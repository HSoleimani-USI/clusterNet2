#include <clusterKernels.cuh>
#include <math.h>
#include <basicOps.cuh>

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



template __global__ void kElementWise<kabs>(const float *A, const float *B,  float *out, const float scalar, int size);
template __global__ void kElementWise<klog>(const float *A, const float *B,  float *out, const float scalar, int size);
template __global__ void kElementWise<ksqrt>(const float *A, const float *B,  float *out, const float scalar, int size);
template __global__ void kElementWise<kpow>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kadd>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<ksub>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kdiv>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kmul>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<klogistic>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<klogistic_grad>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<krectified>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<krectified_grad>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<keq>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<klt>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kgt>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kle>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kge>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kne>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<ksquared_diff>(const float *A, const float *B, float *out, const float scalar, int size);
template<int operation> __global__ void kElementWise(const float *A, const float *B, float *out, const float scalar, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
       switch(operation)
	   {
       	   case kabs: out[i] = fabsf(A[i]); break;
       	   case klog: out[i] = __logf(A[i]); break;
       	   case ksqrt: out[i] = sqrtf(A[i]); break;
       	   case kpow: out[i] = powf(A[i],scalar); break;
       	   case kadd: out[i] = A[i] + B[i]; break;
       	   case ksub: out[i] = A[i] - B[i]; break;
       	   case kdiv: out[i] = fdividef(A[i], B[i]); break;
       	   case kmul: out[i] = A[i] * B[i]; break;
       	   case klogistic: out[i] = 1.0f/(1.0f + expf(A[i])); break;
       	   case klogistic_grad: out[i] = A[i]*(A[i]-1.0f); break;
       	   case krectified: out[i] = A[i] > 0.0f ? A[i] : 0.0f; break;
       	   case krectified_grad: out[i] = A[i] > 0.0f ? 1.0f : 0.0f; break;
       	   case keq: out[i] = (float)(A[i] == B[i]); break;
       	   case klt: out[i] = (float)(A[i] < B[i]); break;
       	   case kgt: out[i] = (float)(A[i] > B[i]); break;
       	   case kge: out[i] = (float)(A[i] >= B[i]); break;
       	   case kle: out[i] = (float)(A[i] <= B[i]); break;
       	   case kne: out[i] = (float)(A[i] != B[i]); break;
       	   case ksquared_diff: out[i] = powf(A[i]-B[i],2.0f); break;
	   }
  }
}

template __global__ void kVectorWise<kvadd>(float *A, float *v, float *out, const float scalar, int rows, int size);
template <int operation> __global__ void kVectorWise(float *A, float *v, float *out, const float scalar, int rows, int size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int offset = 0;
	for (unsigned int i = idx;i < size; i += numThreads)
	{
		offset = (i / rows);
		switch(operation)
		{
			case kvadd: out[i] =  A[i] + v[offset]; break;
		}
	}
}


//for column major data
__global__ void kSlice(float *A, float *out, int rows_A, int cols_A, int rstart, int rend, int cstart, int cend)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int rows_out = (rend - rstart);
  int cols_out = (cend - cstart);
  int size = rows_out*cols_out;

  int current_col = 0;
  int offset = 0;
  int current_row = 0;
  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  current_col = i / rows_out;
	  current_row = i - (current_col*rows_out);

	  offset = (rows_A*(current_col+cstart)) + current_row + rstart;
	  out[i] = A[offset];
  }
}



