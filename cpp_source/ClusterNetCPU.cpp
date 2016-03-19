/*
 * ClusterNetCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "ClusterNetCPU.h"
#include "cblas.h"
#include <BasicOpsWrapperCPU.h>


ClusterNetCPU::ClusterNetCPU()
{
	// TODO Auto-generated constructor stub

	uniform = std::uniform_real_distribution<float>(0.0f,1.0f);
	gaussian = std::normal_distribution<float>(0.0f,1.0f);
	normal_distribution = std::normal_distribution<float>(0.0f,1.0f);

	OPS = new BasicOpsWrapperCPU();

}

void ClusterNetCPU::setRandomState(int seed)
{
	generator_uniform = std::default_random_engine(seed);
	generator_gaussian = std::default_random_engine(seed);
	generator_normal = std::default_random_engine(seed);
}

Matrix<float> *ClusterNetCPU::rand(int rows, int cols)
{
	Matrix<float> *ret = OPS->empty(rows,cols);

	for(int i = 0; i < ret->size; i++)
		ret->data[i] = uniform(generator_uniform);

	return ret;
}

Matrix<float> *ClusterNetCPU::randn(int rows, int cols)
{
	Matrix<float> *ret = OPS->empty(rows,cols);

	for(int i = 0; i < ret->size; i++)
		ret->data[i] = gaussian(generator_gaussian);

	return ret;
}


Matrix<float> *ClusterNetCPU::normal(int rows, int cols, float mean, float std)
{
	Matrix<float> *ret = OPS->empty(rows,cols);
	normal_distribution = std::normal_distribution<float>(mean,std);

	for(int i = 0; i < ret->size; i++)
		ret->data[i] = normal_distribution(generator_gaussian);

	return ret;
}

void ClusterNetCPU::dropout(Matrix<float> *A, Matrix <float> *out, const float dropout)
{
	for(int i = 0; i < out->size; i++)
		out->data[i] = uniform(generator_uniform);

	OPS->dropout(A, out, out, dropout);
}

Matrix<float> *ClusterNetCPU::get_uniformsqrt_weight(int input, int output)
{
	Matrix<float> *out = rand(input,output);
	float range = 8.0f*sqrtf(6.0f/((float)input + output));
	OPS->mul(out,out,range);
	OPS->sub(out,out,range/2.0f);

	return out;
}

void ClusterNetCPU::Tdot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ dot(A, B, out, true, false); }
void ClusterNetCPU::dotT(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ dot(A, B, out, false, true); }
void ClusterNetCPU::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ dot(A, B, out, false, false); }
void ClusterNetCPU::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	int A_rows = A->rows, A_cols = A->cols, B_rows = B->rows, B_cols = B->cols;
	if (T1){ A_rows = A->cols; A_cols = A->rows; }
	if (T2){ B_cols = B->rows; B_rows = B->cols; }

	OPS->check_matrix_multiplication(A, B, out, T1, T2);



		cblas_sgemm(CblasRowMajor,
					 T1 ? CblasTrans : CblasNoTrans,
					 T2 ? CblasTrans : CblasNoTrans,
					 A_rows, B_cols, A_cols, alpha, A->data,
					 A->cols, B->data, B->cols,
					 beta, out->data, out->cols);
}
