#include <BasicOpsCUDA.cuh>
#include <clusterKernels.cuh>
#include <hdf5.h>

template Matrix<int> *to_host(Matrix<int> *in);
template Matrix<float> *to_host(Matrix<float> *in);
template <typename T> Matrix<T> *to_host(Matrix<T> *in)
{
	Matrix<T> *out = (Matrix<T>*)malloc(sizeof(Matrix<T>));
	T *cpu_data = (T*)malloc(in->bytes);

	//this is a bit confusing; we assume that the host data is col major while it is row major
	//that is why we tranpose the data if it is not transposed on the GPU
	if(!in->isRowMajor)
	{

		Matrix<T> *helper = transpose(in);
		CUDA_CHECK_RETURN(cudaMemcpy(cpu_data,helper->data,in->bytes,cudaMemcpyDefault));
		CUDA_CHECK_RETURN(cudaFree(helper->data));
		free(helper);
	}
	else
	{
		CUDA_CHECK_RETURN(cudaMemcpy(cpu_data,in->data,in->bytes,cudaMemcpyDefault));
	}

	out->rows = in->rows;
	out->cols = in->cols;
	out->bytes = in->bytes;
	out->size = in->size;
	out->data = cpu_data;
	out->isRowMajor = true;

  return out;
}


template void free_matrix(Matrix<float> *in);
template <typename T> void free_matrix(Matrix<T> *in)
{
	CUDA_CHECK_RETURN(cudaFree(in->data));
	free(in);
}



//to host where we already have created a gpu buffer
template void to_host(Matrix<int> *gpu, int *cpu);
template void to_host(Matrix<float> *gpu, float *cpu);
template <typename T> void to_host(Matrix<T> *gpu, T *cpu)
{ CUDA_CHECK_RETURN(cudaMemcpy(cpu,gpu->data,gpu->bytes,cudaMemcpyDefault)); }

//pinned memory needed for asynchronous copies between CPU and GPU
//pinned memory makes sure that we do not have to allocate a page CPU buffer before the copy
//this makes the copy faster and asynchronous with respect to the caller
template Matrix<int> *to_pinned(int rows, int cols, int *cpu);
template Matrix<float> *to_pinned(int rows, int cols, float *cpu);
template <typename T> Matrix<T> *to_pinned(int rows, int cols, T *cpu){ return to_pinned<T>(rows, cols, cpu,sizeof(T)*rows*cols); }

template Matrix<int> *to_pinned(int rows, int cols, int *cpu, size_t bytes_to_copy);
template Matrix<float> *to_pinned(int rows, int cols, float *cpu, size_t bytes_to_copy);
template <typename T> Matrix<T> *to_pinned(int rows, int cols, T *cpu, size_t bytes_to_copy)
{
	int size = rows*cols;
	size_t bytes = sizeof(T)*size;
	Matrix<T> *out = (Matrix<T>*)malloc(sizeof(Matrix<T>));
	T *pinned_ptr;
	CUDA_CHECK_RETURN(cudaHostAlloc(&pinned_ptr, bytes, cudaHostAllocPortable));
	for(int i = 0; i < rows*cols; i++){ pinned_ptr[i] = 0.0f;}
	CUDA_CHECK_RETURN(cudaMemcpy(pinned_ptr,cpu,bytes_to_copy,cudaMemcpyDefault));

	out->bytes = bytes;
	out->size = size;
	out->rows = rows;
	out->cols = cols;
	out->data = pinned_ptr;
	out->isRowMajor = true;

	return out;
}

template Matrix<float> *zeros<float>(int rows, int cols);
template <typename T> Matrix<T> *zeros(int rows, int cols)
{
	return fill_matrix<T>(rows, cols, (T)0.0f);
}


template Matrix<float> *ones<float>(int rows, int cols);
template <typename T> Matrix<T> *ones(int rows, int cols)
{
	return fill_matrix<T>(rows, cols, (T)1.0f);
}

template Matrix<unsigned int> *empty<unsigned int>(int rows, int cols);
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
  out->isRowMajor = true;

  return out;
}

template void to_gpu(unsigned int *cpu, Matrix<unsigned int> *gpu);
template void to_gpu(int *cpu, Matrix<int> *gpu);
template void to_gpu(float *cpu, Matrix<float> *gpu);
template<typename T> void to_gpu(T *cpu, Matrix<T> *gpu)
{
    CUDA_CHECK_RETURN(cudaMemcpy(gpu->data,cpu,gpu->bytes,cudaMemcpyDefault));

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

template void sortbykey(Matrix<float> *keys, Matrix<float> *values);
template <typename T> void sortbykey(Matrix<T> *keys, Matrix<T> *values)
{
	thrust::device_ptr<T> d_values(values->data);
	thrust::device_ptr<T> d_keys(keys->data);

	thrust::sort_by_key(d_keys, d_keys + keys->size, d_values);
}


float sum(Matrix<float> *A)
{
	thrust::device_ptr<float> d_values(A->data);
	return thrust::reduce(d_values, d_values+A->size);
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

//the column major format has increasing indexes along its columns:
/*
 * 			[0 3 6]
 * 			[1 4 7]
 * 			[2 5 8]
 */

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
	out->isRowMajor = false;
}

//the row major format has increasing indexes along its rows:
/*
 * 			[0 1 2]
 * 			[3 4 5]
 * 			[6 7 8]
 */
template Matrix<float> *to_row_major(Matrix<float> *A);
template <typename T> Matrix<T> *to_row_major(Matrix<T> *A)
{
  Matrix<T> *out = empty<T>(A->rows,A->cols);
  transpose<T>(A, out, A->rows,A->cols);
  out->isRowMajor = true;

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

//elementwise operation with a single matrix argument
template void elementWise<kabs>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<klog>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<ksqrt>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<klogistic>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<klogistic_grad>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<ktanh>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<ktanh_grad>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<kELU>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<kELU_grad>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<krectified>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<krectified_grad>(Matrix<float> *A, Matrix<float>*out);
template void elementWise<kcopy>(Matrix<float> *A, Matrix<float>*out);
template <int action> void elementWise(Matrix<float> *A, Matrix<float>*out)
{
  check_for_same_dimensions(A,out);
  kElementWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, NULL, out->data,0.0f, out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void elementWise<kpow>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWise<ksmul>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWise<kssub>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWise<ksgt>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void elementWise<kmod>(Matrix<float> *A, Matrix<float>*out, float scalar);
template <int action> void elementWise(Matrix<float> *A, Matrix<float>*out, float scalar)
{
  check_for_same_dimensions(A,out);
  kElementWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, NULL, out->data,scalar, out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

//elementwise operation with a two matrix arguments
template void elementWise<kadd>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<ksub>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<kdiv>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<kmul>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<keq>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<klt>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<kgt>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<kge>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<kle>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<kne>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void elementWise<ksquared_diff>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template <int action> void elementWise(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
  check_for_same_dimensions(A,B);
  check_for_same_dimensions(A,out);

  kElementWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, B->data, out->data,0.0f, out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void elementWise<kdropout>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar);
template <int action> void elementWise(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar)
{
  check_for_same_dimensions(A,B);
  check_for_same_dimensions(A,out);


  kElementWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, B->data, out->data,scalar, out->size);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

//vectorwise operation between matrix and vector
//this is equivalent to broadcasting in numpy
template void vectorWise<kvadd>(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out);
template void vectorWise<kvsub>(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out);
template <int action> void vectorWise(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out)
{
  check_matrix_vector_op(A, v);
  kVectorWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, v->data, out->data, 0.0f, out->rows, out->cols);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void vectorWise<ktmatrix>(Matrix<float> *v, Matrix<float>*out);
template <int action> void vectorWise(Matrix<float> *v, Matrix<float>*out)
{
  check_matrix_vector_op(out, v);
  kVectorWise<action><<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(NULL, v->data, out->data, 0.0f, out->rows, out->cols);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

//slice rows and columns
//equivalent to python slicing, e.h. X[3:4,6:9] is equivalent to slice(X,out, 3,4,6,9)
void slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend)
{
  kSlice<<<out->size/THREADS_PER_BLOCKS + 1, THREADS_PER_BLOCKS>>>(A->data, out->data, A->rows, A->cols, rstart, rend, cstart, cend);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void reduceToRows<rmax>(Matrix<float> *A, Matrix<float> *vout);
template void reduceToRows<rsum>(Matrix<float> *A, Matrix<float> *vout);
template void reduceToRows<rmean>(Matrix<float> *A, Matrix<float> *vout);
template <int reduction> void reduceToRows(Matrix<float> *A, Matrix<float> *vout)
{
    kReduceToRows<reduction><<<A->rows > 1024 ? 1024 : A->rows, 256>>>(A->data, vout->data, A->rows, A->cols);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void reduceToCols<rmax>(Matrix<float> *A, Matrix<float> *vout);
template void reduceToCols<rsum>(Matrix<float> *A, Matrix<float> *vout);
template void reduceToCols<rmean>(Matrix<float> *A, Matrix<float> *vout);
template <int reduction> void reduceToCols(Matrix<float> *A, Matrix<float> *vout)
{
	kReduceToCols<reduction><<<A->cols > 1024 ? 1024 : A->cols, 32>>>(A->data, vout->data, A->rows, A->cols);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template float reduceToValue<rsum>(Matrix<float> *A);
template float reduceToValue<rmax>(Matrix<float> *A);
template float reduceToValue<rmean>(Matrix<float> *A);
template <int reduction> float reduceToValue(Matrix<float> *A)
{
	Matrix<float> *vout = empty<float>(A->rows, 1);
	float retValue = reduceToValue<reduction>(A, vout);
	CUDA_CHECK_RETURN(cudaFree(vout->data));
	free(vout);
	return retValue;
}

template float reduceToValue<rsum>(Matrix<float> *A, Matrix<float> *vout_rows);
template float reduceToValue<rmax>(Matrix<float> *A, Matrix<float> *vout_rows);
template float reduceToValue<rmean>(Matrix<float> *A, Matrix<float> *vout_rows);
template <int reduction> float reduceToValue(Matrix<float> *A, Matrix<float> *vout_rows)
{
	reduceToRows<reduction>(A, vout_rows);
	Matrix<float> *value = empty<float>(1,1);
    kReduceToRows<reduction><<<1, 256>>>(vout_rows->data, value->data, 1, A->rows);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    float retValue = 0.0f;

	CUDA_CHECK_RETURN(cudaMemcpy(&retValue,value->data,value->bytes,cudaMemcpyDefault));

	cudaFree(value->data);
	free(value);

    return retValue;
}


//this softmax is numerically stable
void softmax(Matrix<float> *A, Matrix<float> *out)
{
	check_for_same_dimensions(A, out);
    kSoftMax<<<A->rows > 1024 ? 1024 : A->rows, 256>>>(A->data, out->data, A->rows, A->cols);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


void argmax(Matrix<float> *A, Matrix<float> *out)
{
	check_matrix_vector_op(A, out);
    kArgmax<<<A->rows > 1024 ? 1024 : A->rows, 256>>>(A->data, out->data, A->rows, A->cols);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void lookup(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *out)
{
	assert(embedding->cols <=1024);
	dim3 grid(idx_batch->rows, idx_batch->cols,1);
	kEmbeddingLookup<<<grid, embedding->cols>>>(embedding->data, idx_batch->data, out->data, idx_batch->rows, idx_batch->cols, embedding->cols);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

void embeddingUpdate(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *grad, Matrix<float> *RMS, float RMS_momentum, float learning_rate)
{
	assert(embedding->cols <=1024);
	dim3 grid(idx_batch->rows, idx_batch->cols,1);
	kEmbeddingUpdate<<<grid, embedding->cols>>>(embedding->data, idx_batch->data, grad->data, RMS->data,
												  RMS_momentum, learning_rate, idx_batch->rows, idx_batch->cols, embedding->cols);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template void WeightUpdate<RMSProp>(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate);
template void WeightUpdate<RMSPropInit>(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate);
template <int action> void WeightUpdate(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate)
{
	check_for_same_dimensions(RMS, grad);
	check_for_same_dimensions(RMS, w);
	int threads = 256;
	int blocks = (RMS->size/threads) + 1;
	kRMSprop<action><<<blocks,threads>>>(RMS->data, grad->data, w->data, RMS_multiplier, learning_rate, RMS->size);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}


template<typename T> struct switch_value {};
template<> struct switch_value<int>{ enum { value = 1 }; };
template<> struct switch_value<float>{enum { value = 2 }; };

template Matrix<int> *read_hdf5(const char *filepath);
template Matrix<int> *read_hdf5(const char *filepath, const char *tag);
template Matrix<float> *read_hdf5(const char *filepath);
template Matrix<float> *read_hdf5(const char *filepath, const char *tag);

template <typename T> Matrix<T> *read_hdf5(const char *filepath){ return read_hdf5<T>(filepath,"/Default"); }
template <typename T> Matrix<T> *read_hdf5(const char *filepath, const char *tag)
{
	   hid_t       file_id, dataset_id;

	   file_id = H5Fopen(filepath, H5F_ACC_RDWR, H5P_DEFAULT);
	   dataset_id = H5Dopen2(file_id, tag, H5P_DEFAULT);

	   hid_t dspace = H5Dget_space(dataset_id);
	   hsize_t dims[2];
	   H5Sget_simple_extent_dims(dspace, dims, NULL);
	   size_t bytes = sizeof(T)*dims[0]*dims[1];

	   T *data;
	   CUDA_CHECK_RETURN(cudaHostAlloc(&data, bytes, cudaHostAllocPortable));

	   switch(switch_value<T>::value)
	   {
		   case 1:
			   H5Dread(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
			   break;
		   case 2:
			   H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
			   break;
	   }
	   H5Dclose(dataset_id);
	   H5Fclose(file_id);

	   Matrix<T> *out = (Matrix<T>*)malloc(sizeof(Matrix<T>));
	   out->rows = (int)dims[0];
	   out->cols= (int)dims[1];
	   out->bytes = bytes;
	   out->data = data;
	   out->size = (int)(dims[0]*dims[1]);
	   out->isRowMajor = true;

	   return out;
}

bool check_for_same_dimensions(Matrix<float> *A, Matrix<float> *B)
{
	if(A && B)
	{
		if(A->rows == B->rows && A->cols == B->cols) return true;
		else
		{
			cout << "Matrices do not have the same dimension: " << A->rows << "x" << A->cols << " vs " << B->rows << "x" << B->cols << endl;
			throw "Matricies do not have same dimension!";
		}
	}
	else
		return true;
}

bool check_matrix_multiplication(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2)
{
	int A_rows = A->rows, A_cols = A->cols, B_rows = B->rows, B_cols = B->cols;
	if (T1){ A_rows = A->cols; A_cols = A->rows; }
	if (T2){ B_rows = B->cols; B_cols = B->rows; }

	if(A_rows == out->rows && A_cols == B_rows && B_cols == out->cols) return true;
	else
	{
		cout << "Matrices are not aligned: " << A_rows<< "x" << A_cols << " dot " << B_rows << "x" << B_cols << " -->"  << out->rows << "x" << out->cols <<endl;
		throw "Matrices are not aligned!";
	}

}

bool check_matrix_vector_op(Matrix<float> *A, Matrix<float> *vec)
{
	if(A && vec)
	{
		if((A->rows == vec->rows && vec->cols == 1) ||
		   (A->cols == vec->rows && vec->cols == 1) ||
		   (A->rows == vec->cols && vec->rows == 1) ||
		   (A->cols == vec->cols && vec->rows == 1)) return true;
		else
		{
			cout << "Matrix vector opt does not align: " << A->rows << "x" << A->cols << " vs " << vec->rows << "x" << vec->cols << endl;
			throw "Matrix vector opt does not align!";
		}
	}
	else return true;
}




void print_matrix(Matrix<float> *A, int end_rows, int end_cols){ print_matrix(A,0,end_rows,0,end_cols); }
void print_matrix(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col)
{
	for(int row = start_row; row< end_row; row++)
	{
		printf("[");
		for(int col =start_col; col < end_col; col++)
		{
		  if(A->data[(row*A->cols)+col] < 0.0f)
			  printf("% f ",A->data[(row*A->cols)+col]);
		  else
			  printf("%f ",A->data[(row*A->cols)+col]);
		}
		printf("]\n");
	}
	printf("\n");
}

void printmat(Matrix<float> *A)
{
  Matrix<float> * m = to_host(A);
  print_matrix(m,A->rows,A->cols);
  free(m->data);
  free(m);

}

void printdim(Matrix<float> *A)
{
	cout << A->rows << "x" << A->cols << endl;
}

void printsum(Matrix<float> *A)
{
	cout << sum(A) << endl;
}

void printhostmat(Matrix<float> *A){ print_matrix(A,A->rows,A->cols); }
void printmat(Matrix<float> *A, int end_rows, int end_cols)
{
  Matrix<float> * m = to_host(A);
  print_matrix(m, end_rows, end_cols);
  free(m->data);
  free(m);

}

void printmat(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col)
{
  Matrix<float> * m = to_host(A);
  print_matrix(m, start_row, end_row, start_col, end_col);
  free(m->data);
  free(m);

}

Matrix<float> *get_view(Matrix<float> *A, int rstart, int rend)
{
	assert(rstart < A->rows);
	assert(rstart >= 0);
	assert(rend <= A->rows);
	assert(A->isRowMajor);

	Matrix<float> *ret = new Matrix<float>();
	ret->rows = rend-rstart;
	ret->cols = A->cols;
	ret->size = ret->rows*ret->cols;
	ret->bytes = sizeof(float)*ret->size;

	ret->data = &(A->data)[rstart*A->cols];
	ret->isRowMajor = true;

	return ret;
}
