#include <stdio.h>

#ifndef basicOps_H
#define basicOps_H


#define TILE_DIM (32)
#define BLOCK_ROWS (8)
#define COPY_BLOCK_SIZE 16

#define RDM_NUMBERS_PER_THREAD (1024)
#define THREADS_PER_BLOCKS (512)
#define BLOCKS (4096)

#define DOT_BLOCKS (128)
#define TILE_SIZE (32)
#define DOT_REPS (4)


#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

template<typename T> struct Matrix
{
  public:
    int rows;
    int cols;
    size_t bytes;
    int size;
    T *data;
    Matrix<T> *to_host();
};

template<typename T> Matrix<T> *fill_matrix(int rows, int cols, T fill_value);
template<typename T> Matrix<T> *empty(int rows, int cols);
template<typename T> void to_host(Matrix<T> *gpu, T *cpu);
template<typename T> void to_gpu(T *cpu, Matrix<T> *gpu);


#endif
