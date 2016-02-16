#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


using std::cout;
using std::endl;

#ifndef basicOps_H
#define basicOps_H


#define TILE_DIM (32)
#define BLOCK_ROWS (8)

#define RDM_NUMBERS_PER_THREAD (1024)
#define THREADS_PER_BLOCKS (512)
#define BLOCKS (4096)

#define DOT_BLOCKS (128)
#define TILE_SIZE (32)
#define DOT_REPS (4)

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)


#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }



typedef enum WeightUpdateType_t
{
	RMSProp = 0,
	Momentum = 1,
	PlainSGD = 2,
	RMSPropInit = 3
} WeightUpdateType_t;

typedef enum Operations_t
{
	kabs = 0,
	klog = 1,
	ksqrt = 2,
	kpow = 3,
	kadd = 4,
	ksub = 5,
	kdiv = 6,
	kmul = 7,
	klogistic = 8,
	klogistic_grad = 9,
	krectified = 10,
	krectified_grad = 11,
	keq = 12,
	klt = 13,
	kgt = 14,
	kge = 15,
	kle = 16,
	kne = 17,
	ksquared_diff = 18,



	kvadd = 19,
	kvsub = 20,
	ktmatrix = 21,


	ksmul = 22,
	ksgt = 23,

	kdropout = 24,
	kcopy = 25,
	kssub = 26,
	kELU = 27,
	kELU_grad = 28,

} Operations_t;


typedef enum Reduction_t
{
	rmax,
	rsum,
	rmean

} Reduction_t;

template<typename T> class Matrix
{
  public:
    int rows;
    int cols;
    size_t bytes;
    int size;
    T *data;
    bool isRowMajor;
    Matrix<T> *to_host();
    void free_matrix();
};


template<typename T> Matrix<T> *fill_matrix(int rows, int cols, T fill_value);
template<typename T> Matrix<T> *empty(int rows, int cols);
template<typename T> Matrix<T> *zeros(int rows, int cols);
template<typename T> Matrix<T> *ones(int rows, int cols);
template<typename T> void to_host(Matrix<T> *gpu, T *cpu);
template<typename T> void to_gpu(T *cpu, Matrix<T> *gpu);
template <typename T> Matrix<T> *to_pinned(int rows, int cols, T *cpu);
template <typename T> Matrix<T> *to_pinned(int rows, int cols, T *cpu, size_t bytes_to_copy);


template <typename T> void sortbykey(Matrix<T> *keys, Matrix<T> *values);
float sum(Matrix<float> *A);

template <typename T> void transpose(Matrix<T> *A, Matrix<T> *out, int rows, int cols);
template <typename T> Matrix<T> *to_col_major(Matrix<T> *A);
template <typename T> void to_col_major(Matrix<T> *A, Matrix<T> *out);
template <typename T> Matrix<T> *to_row_major(Matrix<T> *A);
template <typename T> Matrix<T> *transpose(Matrix<T> *A);

template <int action> void elementWise(Matrix<float> *A, Matrix<float>*out);
template <int action> void elementWise(Matrix<float> *A, Matrix<float>*out, float scalar);
template <int action> void elementWise(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out);
template <int action> void elementWise(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);

template <int action> void vectorWise(Matrix<float> *v, Matrix<float>*out);
template <int action> void vectorWise(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out);

void slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend);
void softmax(Matrix<float> *A, Matrix<float> *out);
void argmax(Matrix<float> *A, Matrix<float> *out);

template <int action> void WeightUpdate(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate);

template <int reduction> void reduceToCols(Matrix<float> *A, Matrix<float> *vout);
template <int reduction> void reduceToRows(Matrix<float> *A, Matrix<float> *vout);
template <int reduction> float reduceToValue(Matrix<float> *A);
template <int reduction> float reduceToValue(Matrix<float> *A, Matrix<float> *vout_rows);


bool check_matrix_vector_op(Matrix<float> *A, Matrix<float> *vec);
bool check_for_same_dimensions(Matrix<float> *A, Matrix<float> *B);
bool check_matrix_multiplication(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2);

Matrix<float> *read_hdf5(const char *filepath);
Matrix<float> *read_hdf5(const char *filepath, const char *tag);

Matrix<float> *get_view(Matrix<float> *A, int rstart, int rend);


void print_matrix(Matrix<float> *A, int end_rows, int end_cols);
void print_matrix(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col);
void printmat(Matrix<float> *A);
void printhostmat(Matrix<float> *A);
void printdim(Matrix<float> *A);
void printsum(Matrix<float> *A);
void printmat(Matrix<float> *A, int end_rows, int end_cols);
void printmat(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col);


#endif
