/*
 * BasicOpsWrapperCPU.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#ifndef BASICOPSWRAPPERCPU_H_
#define BASICOPSWRAPPERCPU_H_

#include <Matrix.h>
#include <BasicOpsWrapper.h>

class BasicOpsWrapperCPU : public BasicOpsWrapper
{
public:

	BasicOpsWrapperCPU(){};
	~BasicOpsWrapperCPU(){};

	void free_matrix(Matrix<float> *A);
	Matrix<float> *fill_matrix(int rows, int cols, float fill_value);
	Matrix<float> *empty(int rows, int cols);
	Matrix<float> *zeros(int rows, int cols);
	Matrix<float> *ones(int rows, int cols);
	void to_host(Matrix<float> *gpu, float *cpu);
	Matrix<float> *to_host(Matrix<float> *gpu);
	void to_gpu(float *cpu, Matrix<float> *gpu);
	Matrix<float> *to_pinned(int rows, int cols, float *cpu);
	Matrix<float> *to_pinned(int rows, int cols, float *cpu, size_t bytes_to_copy);
	Matrix<float> *get_pinned(int rows, int cols);

	void transpose(Matrix<float> *A, Matrix<float> *out, int rows, int cols);
	Matrix<float> *to_col_major(Matrix<float> *A);
	void to_col_major(Matrix<float> *A, Matrix<float> *out);
	Matrix<float> *to_row_major(Matrix<float> *A);
	Matrix<float> *transpose(Matrix<float> *A);

	void slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend);
	void softmax(Matrix<float> *A, Matrix<float> *out);
	void argmax(Matrix<float> *A, Matrix<float> *out);


	void lookup(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *out);
	void embeddingUpdate(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *grad, Matrix<float> *RMS, float RMS_momentum, float learning_rate);

	void abs(Matrix<float> *A, Matrix<float> *out);
	void log(Matrix<float> *A, Matrix<float> *out);
	void sqrt(Matrix<float> *A, Matrix<float> *out);
	void logistic(Matrix<float> *A, Matrix<float> *out);
	void logistic_grad(Matrix<float> *A, Matrix<float> *out);
	void tanh(Matrix<float> *A, Matrix<float> *out);
	void tanh_grad(Matrix<float> *A, Matrix<float> *out);
	void ELU(Matrix<float> *A, Matrix<float> *out);
	void ELU_grad(Matrix<float> *A, Matrix<float> *out);
	void rectified(Matrix<float> *A, Matrix<float> *out);
	void rectified_grad(Matrix<float> *A, Matrix<float> *out);
	void copy(Matrix<float> *A, Matrix<float> *out);


	void pow(Matrix<float> *A, Matrix<float> *out, float scalar);
	void mul(Matrix<float> *A, Matrix<float> *out, float scalar);
	void sub(Matrix<float> *A, Matrix<float> *out, float scalar);
	void greater_than(Matrix<float> *A, Matrix<float> *out, float scalar);
	void mod(Matrix<float> *A, Matrix<float> *out, float scalar);

	template <int action> void elementWise(Matrix<float> *A, Matrix<float>*out);
	template <int action> void elementWise(Matrix<float> *A, Matrix<float>*out, float scalar);
	template <int action> void elementWise(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out);
	template <int action> void elementWise(Matrix<float> *A, Matrix<float> *B, Matrix<float>*out, float scalar);

	template <int action> void vectorWise(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out);
	template <int action> void vectorWise(Matrix<float> *v, Matrix<float>*out);

	void add(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void sub(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void div(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void mul(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void less_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void greater_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void greater_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void less_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void not_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void squared_diff(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);

	void dropout(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar);

	void vadd(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out);
	void vsub(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out);
	void get_t_matrix(Matrix<float> *v, Matrix<float> *out);

	void WeightUpdate_RMSProp(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate);

	void reduceToRowsMean(Matrix<float> *A, Matrix<float> *vout);
	void reduceToRowsSum(Matrix<float> *A, Matrix<float> *vout);
	void reduceToRowsMax(Matrix<float> *A, Matrix<float> *vout);
	void reduceToColsMean(Matrix<float> *A, Matrix<float> *vout);
	void reduceToColsSum(Matrix<float> *A, Matrix<float> *vout);
	void reduceToColsMax(Matrix<float> *A, Matrix<float> *vout);

	float mean(Matrix<float> *A);
	float sum(Matrix<float> *A);
	float max(Matrix<float> *A);

	//non-abstract
	void exp(Matrix<float> *A, Matrix<float> *out);

};

#endif /* BASICOPSWRAPPERCPU_H_ */
