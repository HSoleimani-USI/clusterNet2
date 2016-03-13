/*
 * BasicOpsWrapperGPU.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#ifndef BASICOPSWRAPPERGPU_H_
#define BASICOPSWRAPPERGPU_H_

#include <BasicOpsWrapper.h>


class BasicOpsWrapperGPU : public BasicOpsWrapper
{
public:
	BasicOpsWrapperGPU(){};
	~BasicOpsWrapperGPU(){};

	Matrix<float> *fill_matrix(int rows, int cols, float fill_value);
	Matrix<float> *empty(int rows, int cols);
	Matrix<float> *zeros(int rows, int cols);
	Matrix<float> *ones(int rows, int cols);
	void to_host(Matrix<float> *gpu, float *cpu);
	void to_gpu(float *cpu, Matrix<float> *gpu);
	Matrix<float> *to_pinned(int rows, int cols, float *cpu);
	Matrix<float> *to_pinned(int rows, int cols, float *cpu, size_t bytes_to_copy);

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



	bool check_matrix_vector_op(Matrix<float> *A, Matrix<float> *vec);
	bool check_for_same_dimensions(Matrix<float> *A, Matrix<float> *B);
	bool check_matrix_multiplication(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2);
	Matrix<float> *get_view(Matrix<float> *A, int rstart, int rend);


	//map those to util file
	Matrix<float> *read_hdf5(const char *filepath);
	Matrix<float> *read_hdf5(const char *filepath, const char *tag);
	void print_matrix(Matrix<float> *A, int end_rows, int end_cols);
	void print_matrix(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col);
	void printmat(Matrix<float> *A);
	void printhostmat(Matrix<float> *A);
	void printdim(Matrix<float> *A);
	void printsum(Matrix<float> *A);
	void printmat(Matrix<float> *A, int end_rows, int end_cols);
	void printmat(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col);


};

#endif /* BASICOPSWRAPPERGPU_H_ */
