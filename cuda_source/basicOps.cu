#include <basicOps.cuh>
#include <clusterKernels.cuh>
#include <iostream>     // std::cout

using std::cout;
using std::endl;

template Matrix<int> *Matrix<int>::to_host();
template Matrix<float> *Matrix<float>::to_host();
template <typename T> Matrix<T> *Matrix<T>::to_host()
{
	Matrix<T> *out = (Matrix<T>*)malloc(sizeof(Matrix<T>));
	T *cpu_data;

	cpu_data = (T*)malloc(bytes);
	CUDA_CHECK_RETURN(cudaMemcpy(cpu_data,data,bytes,cudaMemcpyDefault));
	out->rows = rows;
	out->cols = cols;
	out->bytes = bytes;
	out->size = size;
	out->data = cpu_data;
  
  return out;
}


template void to_host(Matrix<int> *gpu, int *cpu);
template void to_host(Matrix<float> *gpu, float *cpu);
template <typename T> void to_host(Matrix<T> *gpu, T *cpu)
{ CUDA_CHECK_RETURN(cudaMemcpy(cpu,gpu->data,gpu->bytes,cudaMemcpyDefault)); }

template Matrix<int> *to_pinned(int rows, int cols, int *cpu);
template Matrix<float> *to_pinned(int rows, int cols, float *cpu);
template <typename T> Matrix<T> *to_pinned(int rows, int cols, T *cpu)
{
	int size = rows*cols;
	size_t bytes = sizeof(T)*size;
	Matrix<T> *out = (Matrix<T>*)malloc(sizeof(Matrix<T>));
	T *pinned_ptr;
	cudaHostAlloc(&pinned_ptr, bytes, cudaHostAllocPortable);
	CUDA_CHECK_RETURN(cudaMemcpy(pinned_ptr,cpu,bytes,cudaMemcpyDefault));

	out->bytes = bytes;
	out->size = size;
	out->rows = rows;
	out->cols = cols;
	out->data = pinned_ptr;

	return out;
}


template Matrix<int> *empty<int>(int rows, int cols);
template Matrix<float> *empty<float>(int rows, int cols);
template <typename T> Matrix<T> *empty(int rows, int cols)
{
  T *gpu_data;
  int size = rows*cols;
  size_t bytes = rows*cols*sizeof(T);
  CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_data, bytes));
  
  Matrix<T> *out = (Matrix<T>*)malloc(sizeof(Matrix<T>));
  out->rows = rows;
  out->cols = cols;
  out->bytes = bytes;
  out->size = size;
  out->data = gpu_data;

  return out;
}

template void to_gpu(unsigned int *cpu, Matrix<unsigned int> *gpu);
template void to_gpu(int *cpu, Matrix<int> *gpu);
template void to_gpu(float *cpu, Matrix<float> *gpu);
template<typename T> void to_gpu(T *cpu, Matrix<T> *gpu)
{
    CUDA_CHECK_RETURN(cudaMemcpy(gpu->data,cpu,gpu->bytes,cudaMemcpyDefault));
  	//to_col_major<T>(gpu,gpu);
}


template Matrix<unsigned long long> *fill_matrix(int rows, int cols, unsigned long long fill_value);
template Matrix<unsigned int> *fill_matrix(int rows, int cols, unsigned int fill_value);
template Matrix<int> *fill_matrix(int rows, int cols, int fill_value);
template Matrix<float> *fill_matrix(int rows, int cols, float fill_value);
template <typename T> Matrix<T> *fill_matrix(int rows, int cols, T fill_value)
{
  if(rows < 1 || cols < 1)
  {
    printf("Error: Dimensions must be greater than zero!\n");
  }
 
  Matrix<T> *out = empty<T>(rows, cols);

  kFill_with<T><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(out->data,fill_value,out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
 
  return out;
}


template void transpose(Matrix<float> *A, Matrix<float> *out, int rows, int cols);
template <typename T> void transpose(Matrix<T> *A, Matrix<T> *out, int rows, int cols)
{
  // setup execution parameters
  int grid_x = rows / COPY_BLOCK_SIZE;
  if (rows % COPY_BLOCK_SIZE)
    grid_x++;

  int grid_y = cols / COPY_BLOCK_SIZE;
  if (cols % COPY_BLOCK_SIZE)
    grid_y++;

  dim3 grid(grid_x, grid_y, 1);
  dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);
  kTranspose<T><<< grid, threads >>>(A->data, out->data, rows, cols);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());

}

template Matrix<float> *to_col_major(Matrix<float> *A);
template <typename T> Matrix<T> *to_col_major(Matrix<T> *A)
{
  Matrix<T> *out = empty<T>(A->rows,A->cols);
  transpose<T>(A, out, A->cols,A->rows);
  return out;
}

template void to_col_major(Matrix<unsigned int> *A, Matrix<unsigned int> *out);
template void to_col_major(Matrix<float> *A, Matrix<float> *out);
template <typename T> void to_col_major(Matrix<T> *A, Matrix<T> *out)
{
	transpose<T>(A, out, A->cols,A->rows);
}

template Matrix<float> *to_row_major(Matrix<float> *A);
template <typename T> Matrix<T> *to_row_major(Matrix<T> *A)
{
  Matrix<T> *out = empty<T>(A->rows,A->cols);
  transpose<T>(A, out, A->rows,A->cols);

  return out;
}

template Matrix<unsigned int> *transpose(Matrix<unsigned int> *A);
template Matrix<float> *transpose(Matrix<float> *A);
template <typename T> Matrix<T> *transpose(Matrix<T> *A)
{
  Matrix<T> *out = empty<T>(A->cols,A->rows);
  transpose<T>(A, out, A->rows,A->cols);

  out->rows = A->cols;
  out->cols = A->rows;
  return out;
}



template void elementWiseUnary<kabs>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWiseUnary<klog>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWiseUnary<ksqrt>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWiseUnary<kpow>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWiseUnary<klogistic>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWiseUnary<klogistic_grad>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWiseUnary<krectified>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWiseUnary<krectified_grad>(Matrix<float> *A, Matrix<float>*out, float scalar);
template <int action> void elementWiseUnary(Matrix<float> *A, Matrix<float>*out, float scalar)
{
  kElementWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, NULL, out->data,scalar, out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void elementWise<kadd>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<ksub>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<kdiv>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<kmul>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<keq>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<klt>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<kgt>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<kge>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<kle>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<kne>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template void elementWise<ksquared_diff>(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);
template <int action> void elementWise(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar)
{
  kElementWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, B->data, out->data,scalar, out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void vectorWise<kvadd>(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out, float scalar);
template <int action> void vectorWise(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out, float scalar)
{
  kVectorWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, v->data, out->data, scalar, out->cols, out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend)
{
  kSlice<<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, out->data, A->rows, A->cols, rstart, rend, cstart, cend);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void softmax(Matrix<float> *A, Matrix<float> *out)
{
    kSoftMax<<<A->rows > 1024 ? 1024 : A->rows, 256>>>(A->data, out->data, A->rows, A->cols);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

}
