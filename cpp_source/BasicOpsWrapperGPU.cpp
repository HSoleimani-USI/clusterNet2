/*
 * BasicOpsWrapperGPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "BasicOpsWrapperGPU.h"
#include <BasicOpsCUDA.cuh>

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

Matrix<float> *BasicOpsWrapperGPU::fill_matrix(int rows, int cols, float fill_value){ return ::fill_matrix<float>(rows, cols, fill_value); }
Matrix<float> *BasicOpsWrapperGPU::empty(int rows, int cols){ return ::empty<float>(rows, cols); }
Matrix<float> *BasicOpsWrapperGPU::zeros(int rows, int cols){ return ::zeros<float>(rows, cols); }
Matrix<float> *BasicOpsWrapperGPU::ones(int rows, int cols){ return ::ones<float>(rows, cols); }
void BasicOpsWrapperGPU::to_host(Matrix<float> *gpu, float *cpu){ ::to_host<float>(gpu, cpu); }
void BasicOpsWrapperGPU::to_gpu(float *cpu, Matrix<float> *gpu){ ::to_gpu<float>(cpu, gpu); }
Matrix<float> *BasicOpsWrapperGPU::to_pinned(int rows, int cols, float *cpu){ return ::to_pinned<float>(rows, cols, cpu); }
Matrix<float> *BasicOpsWrapperGPU::to_pinned(int rows, int cols, float *cpu, size_t bytes_to_copy){ return ::to_pinned<float>(rows, cols, cpu, bytes_to_copy); }

void BasicOpsWrapperGPU::transpose(Matrix<float> *A, Matrix<float> *out, int rows, int cols){ ::transpose<float>(A,out,rows,cols); }
Matrix<float> *BasicOpsWrapperGPU::to_col_major(Matrix<float> *A){ return ::to_col_major<float>(A); }
void BasicOpsWrapperGPU::to_col_major(Matrix<float> *A, Matrix<float> *out){ ::to_col_major<float>(A, out); }
Matrix<float> *BasicOpsWrapperGPU::to_row_major(Matrix<float> *A){ return ::to_row_major<float>(A); }
Matrix<float> *BasicOpsWrapperGPU::transpose(Matrix<float> *A){ return ::transpose<float>(A); }

void BasicOpsWrapperGPU::slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend){ ::slice(A, out, rstart, rend, cstart, cend); }
void BasicOpsWrapperGPU::softmax(Matrix<float> *A, Matrix<float> *out){ ::softmax(A, out); }
void BasicOpsWrapperGPU::argmax(Matrix<float> *A, Matrix<float> *out){ ::argmax(A, out); }
void BasicOpsWrapperGPU::lookup(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *out){ ::lookup(embedding, idx_batch, out); }
void BasicOpsWrapperGPU::embeddingUpdate(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *grad, Matrix<float> *RMS, float RMS_momentum, float learning_rate)
{ ::embeddingUpdate(embedding, idx_batch, grad, RMS, RMS_momentum, learning_rate); }

bool BasicOpsWrapperGPU::check_matrix_vector_op(Matrix<float> *A, Matrix<float> *vec){ return ::check_matrix_vector_op(A, vec);}
bool BasicOpsWrapperGPU::check_for_same_dimensions(Matrix<float> *A, Matrix<float> *B){ return ::check_for_same_dimensions(A, B); }
bool BasicOpsWrapperGPU::check_matrix_multiplication(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2)
{ return ::check_matrix_multiplication(A, B, out, T1, T2); }
Matrix<float> *BasicOpsWrapperGPU::get_view(Matrix<float> *A, int rstart, int rend){ return ::get_view(A, rstart, rend); }


//map those to util file
Matrix<float> *BasicOpsWrapperGPU::read_hdf5(const char *filepath){ return ::read_hdf5<float>(filepath); }
Matrix<float> *BasicOpsWrapperGPU::read_hdf5(const char *filepath, const char *tag){ return ::read_hdf5<float>(filepath, tag); }
void BasicOpsWrapperGPU::print_matrix(Matrix<float> *A, int end_rows, int end_cols){ ::print_matrix(A, end_rows, end_cols); }
void BasicOpsWrapperGPU::print_matrix(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col)
{ ::print_matrix(A, start_row, end_row, start_col, end_col); }
void BasicOpsWrapperGPU::printmat(Matrix<float> *A){ ::printmat(A); }
void BasicOpsWrapperGPU::printhostmat(Matrix<float> *A){ ::printhostmat(A); }
void BasicOpsWrapperGPU::printdim(Matrix<float> *A){ ::printdim(A); }
void BasicOpsWrapperGPU::printsum(Matrix<float> *A){ ::printsum(A); }
void BasicOpsWrapperGPU::printmat(Matrix<float> *A, int end_rows, int end_cols){ ::printmat(A, end_rows, end_cols); }
void BasicOpsWrapperGPU::printmat(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col)
{ ::printmat(A, start_row, end_row, start_col, end_col); }


void BasicOpsWrapperGPU::sqrt(Matrix<float> *A, Matrix<float> *out){ elementWise<ksqrt>(A, out); }
void BasicOpsWrapperGPU::abs(Matrix<float> *A, Matrix<float> *out){ elementWise<kabs>(A, out); }
void BasicOpsWrapperGPU::log(Matrix<float> *A, Matrix<float> *out){ elementWise<klog>(A, out); }
void BasicOpsWrapperGPU::logistic(Matrix<float> *A, Matrix<float> *out){ elementWise<klogistic>(A, out); }
void BasicOpsWrapperGPU::logistic_grad(Matrix<float> *A, Matrix<float> *out){ elementWise<klogistic_grad>(A, out); }
void BasicOpsWrapperGPU::tanh(Matrix<float> *A, Matrix<float> *out){ elementWise<ktanh>(A, out); }
void BasicOpsWrapperGPU::tanh_grad(Matrix<float> *A, Matrix<float> *out){ elementWise<ktanh_grad>(A, out); }
void BasicOpsWrapperGPU::ELU(Matrix<float> *A, Matrix<float> *out){ elementWise<kELU>(A, out); }
void BasicOpsWrapperGPU::ELU_grad(Matrix<float> *A, Matrix<float> *out){ elementWise<kELU_grad>(A, out); }
void BasicOpsWrapperGPU::rectified(Matrix<float> *A, Matrix<float> *out){ elementWise<krectified>(A, out); }
void BasicOpsWrapperGPU::rectified_grad(Matrix<float> *A, Matrix<float> *out){ elementWise<krectified_grad>(A, out); }
void BasicOpsWrapperGPU::copy(Matrix<float> *A, Matrix<float> *out){ elementWise<kcopy>(A, out); }

void BasicOpsWrapperGPU::pow(Matrix<float> *A, Matrix<float> *out, float scalar){ elementWise<kpow>(A, out, scalar); }
void BasicOpsWrapperGPU::mul(Matrix<float> *A, Matrix<float> *out, float scalar){ elementWise<ksmul>(A, out, scalar); }
void BasicOpsWrapperGPU::sub(Matrix<float> *A, Matrix<float> *out, float scalar){ elementWise<kssub>(A, out, scalar); }
void BasicOpsWrapperGPU::greater_than(Matrix<float> *A, Matrix<float> *out, float scalar){ elementWise<ksgt>(A, out, scalar); }
void BasicOpsWrapperGPU::mod(Matrix<float> *A, Matrix<float> *out, float scalar){ elementWise<kmod>(A, out, scalar); }

void BasicOpsWrapperGPU::add(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<kadd>(A, B, out); }
void BasicOpsWrapperGPU::sub(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<ksub>(A, B, out); }
void BasicOpsWrapperGPU::div(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<kdiv>(A, B, out); }
void BasicOpsWrapperGPU::mul(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<kmul>(A, B, out); }
void BasicOpsWrapperGPU::equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<keq>(A, B, out); }
void BasicOpsWrapperGPU::less_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<klt>(A, B, out); }
void BasicOpsWrapperGPU::greater_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<kgt>(A, B, out); }
void BasicOpsWrapperGPU::less_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<kle>(A, B, out); }
void BasicOpsWrapperGPU::not_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<kne>(A, B, out); }
void BasicOpsWrapperGPU::squared_diff(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ elementWise<ksquared_diff>(A, B, out); }

void BasicOpsWrapperGPU::dropout(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar){ elementWise<kdropout>(A, B, out, scalar); }


void BasicOpsWrapperGPU::vadd(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out){ vectorWise<kvadd>(A, v, out); }
void BasicOpsWrapperGPU::vsub(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out){ vectorWise<kvsub>(A, v, out); }
void BasicOpsWrapperGPU::get_t_matrix(Matrix<float> *v, Matrix<float> *out){ vectorWise<ktmatrix>(v, out); }


void BasicOpsWrapperGPU::WeightUpdate_RMSProp(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate)
{ WeightUpdate<RMSProp>(RMS, grad, w, RMS_multiplier, learning_rate); }

void BasicOpsWrapperGPU::mean_of_cols(Matrix<float> *A, Matrix<float> *vout){ reduceToRows<rmean>(A, vout); }
void BasicOpsWrapperGPU::sum_of_cols(Matrix<float> *A, Matrix<float> *vout){ reduceToRows<rsum>(A, vout); }
void BasicOpsWrapperGPU::max_of_cols(Matrix<float> *A, Matrix<float> *vout){ reduceToRows<rmax>(A, vout); }
void BasicOpsWrapperGPU::mean_of_rows(Matrix<float> *A, Matrix<float> *vout){ reduceToCols<rmean>(A, vout); }
void BasicOpsWrapperGPU::sum_of_rows(Matrix<float> *A, Matrix<float> *vout){ reduceToCols<rsum>(A, vout); }
void BasicOpsWrapperGPU::max_of_rows(Matrix<float> *A, Matrix<float> *vout){ reduceToCols<rmax>(A, vout); }

float BasicOpsWrapperGPU::mean(Matrix<float> *A){ return reduceToValue<rmean>(A); }
float BasicOpsWrapperGPU::sum(Matrix<float> *A){ return reduceToValue<rsum>(A); }
float BasicOpsWrapperGPU::max(Matrix<float> *A){ return reduceToValue<rmax>(A); }
