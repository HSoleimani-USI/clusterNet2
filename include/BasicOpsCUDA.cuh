#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <Matrix.h>
#include <BasicOpsWrapper.h>


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







template <typename T> Matrix<T> *to_host(Matrix<T> *in);
template <typename T> void free_matrix(Matrix<T> *in);
template<typename T> Matrix<T> *fill_matrix(int rows, int cols, T fill_value);
template<typename T> Matrix<T> *empty(int rows, int cols);
template<typename T> Matrix<T> *zeros(int rows, int cols);
template<typename T> Matrix<T> *ones(int rows, int cols);
template<typename T> void to_host(Matrix<T> *gpu, T *cpu);
template<typename T> void to_gpu(T *cpu, Matrix<T> *gpu);
template <typename T> Matrix<T> *get_pinned(int rows, int cols);
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


void lookup(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *out);
void embeddingUpdate(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *grad, Matrix<float> *RMS, float RMS_momentum, float learning_rate);

template <int action> void WeightUpdate(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate);

template <int reduction> void reduceToCols(Matrix<float> *A, Matrix<float> *vout);
template <int reduction> void reduceToRows(Matrix<float> *A, Matrix<float> *vout);
template <int reduction> float reduceToValue(Matrix<float> *A);
template <int reduction> float reduceToValue(Matrix<float> *A, Matrix<float> *vout_rows);


#endif
