#include <basicOps.cuh>
#include <clusterKernels.cuh>

template Matrix<int> *Matrix<int>::to_host();
template Matrix<float> *Matrix<float>::to_host();
template <typename T> Matrix<T> *Matrix<T>::to_host()
{
	Matrix<T> *row_major;
	row_major = to_row_major<T>(this);

	Matrix<T> *out = (Matrix<T>*)malloc(sizeof(Matrix<T>));
	T *cpu_data;

	cpu_data = (T*)malloc(bytes);
	CUDA_CHECK_RETURN(cudaMemcpy(cpu_data,row_major->data,row_major->bytes,cudaMemcpyDefault));
	out->rows = row_major->rows;
	out->cols = row_major->cols;
	out->bytes = row_major->bytes;
	out->size = row_major->size;
	out->data = cpu_data;

	CUDA_CHECK_RETURN(cudaFree(row_major->data));
	free(row_major);
  
  return out;
}


template void to_host(Matrix<int> *gpu, int *cpu);
template void to_host(Matrix<float> *gpu, float *cpu);
template <typename T> void to_host(Matrix<T> *gpu, T *cpu)
{  
	Matrix<T> *row_major;
	row_major = to_row_major<T>(gpu);
	CUDA_CHECK_RETURN(cudaMemcpy(cpu,row_major->data,row_major->bytes,cudaMemcpyDefault));
	CUDA_CHECK_RETURN(cudaFree(row_major->data));
	free(row_major);
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
  	to_col_major<T>(gpu,gpu);
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
  //cudaFree(A->data);
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

