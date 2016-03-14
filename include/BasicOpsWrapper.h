/*
 * BasicOpsWrapper.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#ifndef BASICOPSWRAPPER_H_
#define BASICOPSWRAPPER_H_

#include <Matrix.h>



typedef enum WeightUpdateType_t
{
	RMSProp = 0,
	Momentum = 1,
	PlainSGD = 2,
	RMSPropInit = 3
} WeightUpdateType_t;

class BasicOpsWrapper
{
public:
	BasicOpsWrapper(){};
	~BasicOpsWrapper(){};

	virtual void free_matrix(Matrix<float> *A) = 0;

	virtual Matrix<float> *fill_matrix(int rows, int cols, float fill_value) = 0;
	virtual Matrix<float> *empty(int rows, int cols) = 0;
	virtual Matrix<float> *zeros(int rows, int cols) = 0;
	virtual Matrix<float> *ones(int rows, int cols) = 0;

	virtual Matrix<float> *to_host(Matrix<float> *gpu) = 0; //  for cpu and gpu
	virtual void to_host(Matrix<float> *gpu, float *cpu) = 0;
	virtual void to_gpu(float *cpu, Matrix<float> *gpu) = 0;\
	//  pin (mapped) memory, cpu doent need to create new memory
	//  one copy instead of 2 copies (zeros copy)
	virtual Matrix<float> *to_pinned(int rows, int cols, float *cpu) = 0;
	virtual Matrix<float> *to_pinned(int rows, int cols, float *cpu, size_t bytes_to_copy) = 0;



	virtual Matrix<float> *transpose(Matrix<float> *A) = 0;

	virtual void transpose(Matrix<float> *A, Matrix<float> *out, int rows, int cols) = 0;

	//  BLAS uses only col_major
	virtual Matrix<float> *to_col_major(Matrix<float> *A) = 0;
	virtual void to_col_major(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual Matrix<float> *to_row_major(Matrix<float> *A) = 0;


	// select a slice from matrix

	virtual void abs(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void log(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void sqrt(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void logistic(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void logistic_grad(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void tanh(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void tanh_grad(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void ELU(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void ELU_grad(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void rectified(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void rectified_grad(Matrix<float> *A, Matrix<float> *out) = 0;
	virtual void copy(Matrix<float> *A, Matrix<float> *out) = 0;

	virtual void pow(Matrix<float> *A, Matrix<float> *out, float scalar) = 0;
	virtual void mul(Matrix<float> *A, Matrix<float> *out, float scalar) = 0;
	virtual void sub(Matrix<float> *A, Matrix<float> *out, float scalar) = 0;
	virtual void greater_than(Matrix<float> *A, Matrix<float> *out, float scalar) = 0;
	virtual void mod(Matrix<float> *A, Matrix<float> *out, float scalar) = 0;

	virtual void add(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void sub(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void div(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void mul(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void less_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void greater_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void greater_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void less_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void not_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;
	virtual void squared_diff(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) = 0;

	virtual void dropout(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar) = 0;

	virtual void vadd(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out) = 0;
	virtual void vsub(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out) = 0;
	virtual void get_t_matrix(Matrix<float> *v, Matrix<float> *out) = 0;

	virtual void slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend) = 0;
	//  normalized each element on the row (e ^ each element in the row / e^ sum of one row)
	// gives soft probabilty distribution
	virtual void softmax(Matrix<float> *A, Matrix<float> *out) = 0;
	// gives us the index of the maximum in each row
	virtual void argmax(Matrix<float> *A, Matrix<float> *out) = 0;

	//  we have a big table of all possibilites. look at the row and we copy on the matrix then.
	virtual void lookup(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *out) = 0;
	// embeddedwords...updating our table based on the matrix
	virtual void embeddingUpdate(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *grad, Matrix<float> *RMS, float RMS_momentum, float learning_rate) = 0;


    // gradient descent with some acceleration methods
    // ---
    // reduce to sum  of values in the row
	/*
	template <int action> void WeightUpdate(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate) = 0;


	template <int reduction> void reduceToCols(Matrix<float> *A, Matrix<float> *vout) = 0;
	template <int reduction> void reduceToRows(Matrix<float> *A, Matrix<float> *vout) = 0;
	template <int reduction> float reduceToValue(Matrix<float> *A) = 0;
	template <int reduction> float reduceToValue(Matrix<float> *A, Matrix<float> *vout_rows) = 0;
	*/

	virtual void WeightUpdate_RMSProp(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate) = 0;

	virtual void mean_of_cols(Matrix<float> *A, Matrix<float> *vout) = 0;
	virtual void sum_of_cols(Matrix<float> *A, Matrix<float> *vout) = 0;
	virtual void max_of_cols(Matrix<float> *A, Matrix<float> *vout) = 0;
	virtual void mean_of_rows(Matrix<float> *A, Matrix<float> *vout) = 0;
	virtual void sum_of_rows(Matrix<float> *A, Matrix<float> *vout) = 0;
	virtual void max_of_rows(Matrix<float> *A, Matrix<float> *vout) = 0;


	virtual float mean(Matrix<float> *A) = 0;
	virtual float sum(Matrix<float> *A) = 0;
	virtual float max(Matrix<float> *A) = 0;

	// to check the dimention
	virtual bool check_matrix_vector_op(Matrix<float> *A, Matrix<float> *vec) = 0;
	virtual bool check_for_same_dimensions(Matrix<float> *A, Matrix<float> *B) = 0;
	virtual bool check_matrix_multiplication(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2) = 0;


	//map those to util file
	virtual Matrix<float> *read_hdf5(const char *filepath) = 0;
	virtual Matrix<float> *read_hdf5(const char *filepath, const char *tag) = 0;


	// getting pointers from the starting row to nth rows
	virtual Matrix<float> *get_view(Matrix<float> *A, int rstart, int rend) = 0;


	virtual void print_matrix(Matrix<float> *A, int end_rows, int end_cols) = 0;
	virtual void print_matrix(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col) = 0;
	virtual void printmat(Matrix<float> *A) = 0;
	virtual void printhostmat(Matrix<float> *A) = 0;
	virtual void printdim(Matrix<float> *A) = 0;
	virtual void printsum(Matrix<float> *A) = 0;
	virtual void printmat(Matrix<float> *A, int end_rows, int end_cols) = 0;
	virtual void printmat(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col) = 0;


};

#endif /* BASICOPSWRAPPER_H_ */
