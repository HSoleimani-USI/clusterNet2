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
        unsigned int index_in = xIndex * height + yIndex;
        block[threadIdx.y][threadIdx.x] = A[index_in];
    }

    __syncthreads();

    // write the transposed Matrix *tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = xIndex*width + yIndex ;
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
template __global__ void kElementWise<ktanh>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<ktanh_grad>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kELU>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kELU_grad>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<krectified>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<krectified_grad>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<keq>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<klt>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kgt>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kle>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kge>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kne>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<ksquared_diff>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kdropout>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<ksmul>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kssub>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<ksgt>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kcopy>(const float *A, const float *B, float *out, const float scalar, int size);
template __global__ void kElementWise<kmod>(const float *A, const float *B, float *out, const float scalar, int size);
template<int operation> __global__ void kElementWise(const float *A, const float *B, float *out, const float scalar, int size)
{
  const unsigned int numThreads = blockDim.x * gridDim.x;
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (unsigned int i = idx;i < size; i += numThreads)
  {
	  //this switch operation will be removed by the compiler upon instantiation of the template
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
       	   case klogistic: out[i] = 1.0f/(1.0f + expf(-A[i])); break;
       	   case klogistic_grad: out[i] = A[i]*(1.0f-A[i]); break;
       	   case kELU: out[i] = A[i] > 0.0f ? A[i] : expf(A[i])-1.0f; break;
       	   case kELU_grad: out[i] = A[i] > 0.0f ? 1.0f : A[i] + 1.0f; break;
       	   case krectified: out[i] = A[i] > 0.0f ? A[i] : 0.0f; break;
       	   case krectified_grad: out[i] = A[i] > 0.0f ? 1.0f : 0.0f; break;
       	   case keq: out[i] = (float)(A[i] == B[i]); break;
       	   case klt: out[i] = (float)(A[i] < B[i]); break;
       	   case kgt: out[i] = (float)(A[i] > B[i]); break;
       	   case kge: out[i] = (float)(A[i] >= B[i]); break;
       	   case kle: out[i] = (float)(A[i] <= B[i]); break;
       	   case kne: out[i] = (float)(A[i] != B[i]); break;
       	   case ksquared_diff: out[i] = powf(A[i]-B[i],2.0f); break;

       	   case ksmul: out[i] = A[i] * scalar; break;
       	   case kssub: out[i] = A[i] - scalar; break;
       	   case kdropout: out[i] = B[i] > scalar ? A[i] : 0.0f; break;
       	   case kcopy: out[i] = A[i]; break;


       	   case ksgt: out[i] = (float)(A[i] > scalar); break;
       	   case kmod: out[i] = (float)((int)A[i] % (int)scalar); break;

       	   case ktanh: out[i] = tanhf(A[i]); break;
       	   case ktanh_grad: out[i] = 1.0f - (A[i]*A[i]); break;
	   }
  }
}


template __global__ void kVectorWise<kvadd>(float *A, float *v, float *out, const float scalar, int rows, int cols);
template __global__ void kVectorWise<kvsub>(float *A, float *v, float *out, const float scalar, int rows, int cols);
template __global__ void kVectorWise<ktmatrix>(float *A, float *v, float *out, const float scalar, int rows, int cols);
template <int operation > __global__ void kVectorWise(float *A, float *v, float *out, const float scalar, int rows, int cols)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	for (unsigned int i = idx; i < rows*cols; i += numThreads)
	{
		//this switch operation will be removed by the compiler upon instantiation of the template
		switch(operation)
		{
			case kvadd: out[i] =  A[i] + v[i - ((i / cols)*cols)]; break;
			case kvsub: out[i] =  A[i] - v[i - ((i / cols)*cols)]; break;
			case ktmatrix: out[i] = i-((i / cols)*cols) == (int)v[(i / cols)] ? 1.0f : 0.0f; break;
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
	  current_row = i / cols_out;
	  current_col = i - (current_row*cols_out);

	  offset = (cols_A*(current_row+rstart)) + current_col + cstart;
	  out[i] = A[offset];
  }
}


__global__ void kSoftMax(float* A, float* out, const unsigned int rows, const unsigned int cols)
{
	float col_value = 0.0f;
	int row_offset = 0;

	__shared__ float max_values[256];
	__shared__ float row_sums[256];

	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		//fill with min values
		max_values[threadIdx.x] = -FLT_MAX;
		row_sums[threadIdx.x] = 0.0f;

		row_offset = row*cols;
		 //calc max value of the row
		for (unsigned int i = threadIdx.x; i < cols; i+=blockDim.x)
		{
			col_value = A[i + row_offset];
			max_values[threadIdx.x] = fmaxf(max_values[threadIdx.x],col_value);
		}

		reduceByValue<rmax>(max_values,threadIdx.x,blockDim.x);

		for (unsigned int i = threadIdx.x; i < cols; i+=blockDim.x)
		{
			col_value = A[i + row_offset];
			row_sums[threadIdx.x] += __expf(col_value-max_values[0]);
		}

		reduceByValue<rsum>(row_sums,threadIdx.x,blockDim.x);

		//calc the value of each element in the row
		for (unsigned int i = threadIdx.x; i < cols; i+=blockDim.x)
		{
			//subtraction by max exponential values makes this softmax numerically stable
			out[row_offset + i] = __expf(A[row_offset + i] - max_values[0])/row_sums[0];
		}

	}
}

template __global__ void kReduceToRows<rsum>(float* A, float* out, const unsigned int rows, const unsigned int cols);
template __global__ void kReduceToRows<rmax>(float* A, float* out, const unsigned int rows, const unsigned int cols);
template __global__ void kReduceToRows<rmean>(float* A, float* out, const unsigned int rows, const unsigned int cols);
template <int reduction>__global__ void kReduceToRows(float* A, float* out, const unsigned int rows, const unsigned int cols)
{
	float col_value = 0.0f;
	int row_offset = 0;
	__shared__ float col_reductions[256];

	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		switch(reduction)
		{
			case rmax: col_reductions[threadIdx.x] = -FLT_MAX; break;
			case rsum: col_reductions[threadIdx.x] = 0.0f; break;
			case rmean: col_reductions[threadIdx.x] = 0.0f; break;
		}

		row_offset = row*cols;
		for (unsigned int i = threadIdx.x; i < cols; i+=blockDim.x)
		{
			col_value = A[i + row_offset];
			switch(reduction)
			{
				case rmax: col_reductions[threadIdx.x] = fmaxf(col_reductions[threadIdx.x],col_value); break;
				case rsum: col_reductions[threadIdx.x] = col_reductions[threadIdx.x]+col_value; break;
				case rmean: col_reductions[threadIdx.x] = col_reductions[threadIdx.x]+col_value; break;
			}
		}

		reduceByValue<reduction>(col_reductions,threadIdx.x,blockDim.x);

		if(threadIdx.x == 0)
		{
			switch(reduction)
			{
				case rmean:
					out[row] = col_reductions[0]/(float)cols;
					break;
				default:
					out[row] = col_reductions[0];
					break;
			}

		}

	}
}

template __global__ void kReduceToCols<rsum>(float* A, float* out, const unsigned int rows, const unsigned int cols);
template __global__ void kReduceToCols<rmax>(float* A, float* out, const unsigned int rows, const unsigned int cols);
template __global__ void kReduceToCols<rmean>(float* A, float* out, const unsigned int rows, const unsigned int cols);
template <int reduction>__global__ void kReduceToCols(float* A, float* out, const unsigned int rows, const unsigned int cols)
{
	float row_value = 0.0f;
	__shared__ float row_reductions[32];

	for (unsigned int col = blockIdx.x; col < cols; col += gridDim.x)
	{
		switch(reduction)
		{
			case rmax: row_reductions[threadIdx.x] = -FLT_MAX; break;
			case rsum: row_reductions[threadIdx.x] = 0.0f; break;
			case rmean: row_reductions[threadIdx.x] = 0.0f; break;
		}

		for (unsigned int row = threadIdx.x; row < rows; row+=blockDim.x)
		{
			row_value = A[col + (cols*row)];
			switch(reduction)
			{
				case rmax: row_reductions[threadIdx.x] = fmaxf(row_reductions[threadIdx.x],row_value); break;
				case rsum: row_reductions[threadIdx.x] = row_reductions[threadIdx.x]+row_value; break;
				case rmean: row_reductions[threadIdx.x] = row_reductions[threadIdx.x]+row_value; break;
			}
		}

		reduceByValue<reduction>(row_reductions,threadIdx.x,blockDim.x);

		if(threadIdx.x == 0)
		{
			switch(reduction)
			{
				case rmean:
					out[col] = row_reductions[0]/(float)rows;
					break;
				default:
					out[col] = row_reductions[0];
					break;
			}

		}

	}
}

template __device__ float reduction_action<rsum>(float a, float b);
template __device__ float reduction_action<rmax>(float a, float b);
template __device__ float reduction_action<rmean>(float a, float b);
template<int action> __device__ float reduction_action(float a, float b)
{
	//this switch operation will be removed by the compiler upon instantiation of the template
	switch(action)
	{
		case rsum: return a+b;
		case rmean: return a+b;
		case rmax: return fmaxf(a,b);
		default: return 0.0f;
	}
}

//reductions in shared memory. These are very fast, especially for 32 or less elements.
template __device__ void reduceByValue<rsum>(float* sdata, const unsigned int tid, const unsigned int threads);
template __device__ void reduceByValue<rmax>(float* sdata, const unsigned int tid, const unsigned int threads);
template __device__ void reduceByValue<rmean>(float* sdata, const unsigned int tid, const unsigned int threads);
template <int action> __device__ void reduceByValue(float* sdata, const unsigned int tid, const unsigned int threads)
{

	  //Synchronize threads to share shared memory data
	  __syncthreads();

	  float agg = sdata[tid];

	  // do reduction in shared mem
	  if (threads >= 1024) { if (tid < 512) { sdata[tid] = agg = reduction_action<action>(agg, sdata[tid + 512]); } __syncthreads(); }
	  if (threads >= 512) { if (tid < 256) { sdata[tid] = agg = reduction_action<action>(agg, sdata[tid + 256]); } __syncthreads(); }
	  if (threads >= 256) { if (tid < 128) { sdata[tid] = agg = reduction_action<action>(agg, sdata[tid + 128]); } __syncthreads(); }
	  if (threads >= 128) { if (tid <  64) { sdata[tid] = agg = reduction_action<action>(agg, sdata[tid + 64]);  } __syncthreads(); }

	  if (threads == 32){
	    if (tid < 16)
	    {
	      // now that we are using warp-synchronous programming (below)
	      // we need to declare our shared memory volatile so that the compiler
	      // doesn't reorder stores to it and induce incorrect behavior.
	      volatile float* smem = sdata;
	      if (threads >=  32) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 16]); }
	      if (threads >=  16) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 8]); }
	      if (threads >=   8) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 4]);; }
	      if (threads >=   4) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 2]); }
	      if (threads >=   2) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 1]); }
	    }
	  }
	  else
	  {
	    if (tid < 32)
	    {
	      // now that we are using warp-synchronous programming (below)
	      // we need to declare our shared memory volatile so that the compiler
	      // doesn't reorder stores to it and induce incorrect behavior.
	      volatile float* smem = sdata;
	      if (threads >=  64) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 32]); }
	      if (threads >=  32) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 16]); }
	      if (threads >=  16) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 8]); }
	      if (threads >=   8) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 4]); }
	      if (threads >=   4) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 2]); }
	      if (threads >=   2) { smem[tid] = agg = reduction_action<action>(agg, smem[tid + 1]); }
	    }
	  }

	  __syncthreads();

}

__device__ void reduceToArgmax(float* svalues, float* skeys,const  unsigned int tid, const unsigned int threads)
{

	//Synchronize threads to share shared memory data
	__syncthreads();

  	float maxValue = svalues[tid];

	if (threads >= 1024)
	{
		if (tid < 512){ if(maxValue < svalues[tid + 512]){svalues[tid] = maxValue = svalues[tid + 512];  skeys[tid] = skeys[tid + 512]; }}
		__syncthreads();
	}
	if (threads >= 512)
	{
		if (tid < 256){ if(maxValue < svalues[tid + 256]){svalues[tid] = maxValue = svalues[tid + 256];  skeys[tid] = skeys[tid + 256]; }}
		__syncthreads();
	}
	if (threads >= 256)
	{
		if (tid < 128){ if(maxValue < svalues[tid + 128]){svalues[tid] = maxValue = svalues[tid + 128];  skeys[tid] = skeys[tid + 128]; }}
		__syncthreads();
	}
	if (threads >= 128)
	{
		if (tid < 64){ if(maxValue < svalues[tid + 64]){svalues[tid] = maxValue = svalues[tid + 64];  skeys[tid] = skeys[tid + 64]; }}
		__syncthreads();
	}

  	if(threads == 32)
  	{
		if (tid < 16)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile float* smemMax = svalues;
			volatile float* smemArgMax = skeys;
			if (threads >=  32) if(maxValue < smemMax[tid + 16]){smemMax[tid] = maxValue = smemMax[tid + 16];  smemArgMax[tid] = smemArgMax[tid + 16]; }
			if (threads >=  16) if(maxValue < smemMax[tid +  8]){smemMax[tid] = maxValue = smemMax[tid +  8];  smemArgMax[tid] = smemArgMax[tid +  8]; }
			if (threads >=   8) if(maxValue < smemMax[tid +  4]){smemMax[tid] = maxValue = smemMax[tid +  4];  smemArgMax[tid] = smemArgMax[tid +  4]; }
			if (threads >=   4) if(maxValue < smemMax[tid +  2]){smemMax[tid] = maxValue = smemMax[tid +  2];  smemArgMax[tid] = smemArgMax[tid +  2]; }
			if (threads >=   2) if(maxValue < smemMax[tid +  1]){smemMax[tid] = maxValue = smemMax[tid +  1];  smemArgMax[tid] = smemArgMax[tid +  1]; }
		}
  	}
	else
	{
		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behavior.
			volatile float* smemMax = svalues;
			volatile float* smemArgMax = skeys;
			if (threads >=  64) if(maxValue < smemMax[tid + 32]){smemMax[tid] = maxValue = smemMax[tid + 32];  smemArgMax[tid] = smemArgMax[tid + 32]; }
			if (threads >=  32) if(maxValue < smemMax[tid + 16]){smemMax[tid] = maxValue = smemMax[tid + 16];  smemArgMax[tid] = smemArgMax[tid + 16]; }
			if (threads >=  16) if(maxValue < smemMax[tid +  8]){smemMax[tid] = maxValue = smemMax[tid +  8];  smemArgMax[tid] = smemArgMax[tid +  8]; }
			if (threads >=   8) if(maxValue < smemMax[tid +  4]){smemMax[tid] = maxValue = smemMax[tid +  4];  smemArgMax[tid] = smemArgMax[tid +  4]; }
			if (threads >=   4) if(maxValue < smemMax[tid +  2]){smemMax[tid] = maxValue = smemMax[tid +  2];  smemArgMax[tid] = smemArgMax[tid +  2]; }
			if (threads >=   2) if(maxValue < smemMax[tid +  1]){smemMax[tid] = maxValue = smemMax[tid +  1];  smemArgMax[tid] = smemArgMax[tid +  1]; }
		}

	}

	__syncthreads();
}

__global__ void kArgmax(float* A, float* vout, const unsigned int rows, const unsigned int cols)
{
	float col_value = 0.0f;
	int row_offset = 0;

	__shared__ float max_values[256];
	__shared__ float max_ids[256];

	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		//fill with min values
		max_values[threadIdx.x] = -FLT_MAX;
		max_ids[threadIdx.x] = -1.0f;

		row_offset = row*cols;
		 //calc max value of the row
		for (unsigned int i = threadIdx.x; i < cols; i+=blockDim.x)
		{
			col_value = A[i + row_offset];
			if(col_value > max_values[threadIdx.x])
			{
				max_values[threadIdx.x] = col_value;
				max_ids[threadIdx.x] = i;
			}
		}

		reduceToArgmax(max_values, max_ids, threadIdx.x, blockDim.x);

		if(threadIdx.x == 0)
			vout[row] = max_ids[0];

	}
}



/*
RMSProp = 0,
Momentum = 1,
PlainSGD = 2,
RMSPropInit = 3
*/

template __global__ void kRMSprop<RMSProp>(float *RMS, float *grad, float *w, float RMS_multiplier, float learning_rate, int size);
template __global__ void kRMSprop<RMSPropInit>(float *RMS, float *grad, float *w, float RMS_multiplier, float learning_rate, int size);
template <int action> __global__ void kRMSprop(float *RMS, float *grad, float *w, float RMS_multiplier, float learning_rate, int size)
{
	  const unsigned int numThreads = blockDim.x * gridDim.x;
	  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	  const float rms_reciprocal = 1.0f - RMS_multiplier;

	  float RMS_value = 0.0f;
	  float grad_value = 0.0f;

	  for (unsigned int i = idx;i < size; i += numThreads)
	  {
		  grad_value = grad[i];

		  switch(action)
		  {
		  	  case RMSProp:
				  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
				  grad_value = learning_rate*fdividef(grad_value,(__fsqrt_rn(RMS_value)+1.0e-08f));
				  RMS[i] = RMS_value;
				  w[i] -= grad_value;
		  		  break;
		  	  case RMSPropInit:
				  RMS_value = (RMS_multiplier*RMS[i]) + (powf(grad_value,2.0f)*rms_reciprocal);
				  grad_value = learning_rate*grad_value;
				  RMS[i] = RMS_value;
				  w[i] -= grad_value;
				  break;
		  }


	  }
}
__global__ void kEmbeddingLookup(float *embeddings, float *idx_batch, float *out, int rows, int cols, int embeddings_cols)
{
	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		for (unsigned int col = blockIdx.y; col < cols; col += gridDim.y)
		{
			int embedding_start_idx = ((int)idx_batch[(cols*row) + col])*embeddings_cols;
			out[(((row*cols) + col)*embeddings_cols)  + threadIdx.x] = embeddings[embedding_start_idx + threadIdx.x];


		}
	}
}


__global__ void kEmbeddingUpdate(float *embeddings, float *idx_batch, float *grad, float *RMS,
								 float RMS_multiplier, float learning_rate, int rows, int cols, int embeddings_cols)
{
	const float rms_reciprocal = 1.0f - RMS_multiplier;
	float RMS_value = 0.0f;
	float grad_value = 0.0f;
	for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x)
	{
		for (unsigned int col = blockIdx.y; col < cols; col += gridDim.y)
		{
			int embedding_start_idx = ((int)idx_batch[(cols*row) + col])*embeddings_cols;
			grad_value =  grad[(((row*cols) + col)*embeddings_cols)  + threadIdx.x];
			RMS_value = (RMS_multiplier*RMS[embedding_start_idx + threadIdx.x]) + (powf(grad_value,2.0f)*rms_reciprocal);

			grad_value = learning_rate*fdividef(grad_value,(__fsqrt_rn(RMS_value)+1.0e-08f));
			RMS[embedding_start_idx + threadIdx.x] = RMS_value;
			embeddings[embedding_start_idx + threadIdx.x] -= grad_value;
		}
	}
}

