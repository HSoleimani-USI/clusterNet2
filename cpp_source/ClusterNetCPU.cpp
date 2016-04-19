/*
 * ClusterNetCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "ClusterNetCPU.h"
#ifdef PHI
	#include <mkl.h>
#else
	#include "cblas.h"
#endif
#include <BasicOpsWrapperCPU.h>

ClusterNetCPU::ClusterNetCPU()
{
	// TODO Auto-generated constructor stub
	OPS = new BasicOpsWrapperCPU();

}

void ClusterNetCPU::setRandomState(int seed)
{
	srand(seed);
}

Matrix<float> *ClusterNetCPU::rand(int rows, int cols)
{
	Matrix<float> *ret = OPS->empty(rows,cols);

	for(int i = 0; i < ret->size; i++)
		ret->data[i] =(float)((double) rand() / (RAND_MAX+1) * (2));

	return ret;
}

Matrix<float> *ClusterNetCPU::randn(int rows, int cols)
{
	Matrix<float> *ret = OPS->empty(rows,cols);

	for(int i = 0; i < ret->size; i++)
	{
		float rdm = (float)((double) rand() / (RAND_MAX+1) * (2));
		ret->data[i] =  1.0f/(1.0f + expf((-0.07056* (rdm*rdm*rdm)) - (1.5976*rdm)));
	}

	return ret;
}


Matrix<float> *ClusterNetCPU::normal(int rows, int cols, float mean, float std)
{
	Matrix<float> *ret = OPS->empty(rows,cols);

	for(int i = 0; i < ret->size; i++)
	{
		float rdm = (float)((double) rand() / (RAND_MAX+1) * (2));
		ret->data[i] =  1.0f/(1.0f + expf((-0.07056* (rdm*rdm*rdm)) - (1.5976*rdm)));
	}

	return ret;
}

void ClusterNetCPU::dropout(Matrix<float> *A, Matrix <float> *out, const float dropout)
{
	for(int i = 0; i < out->size; i++)
		out->data[i] = vargen_uniform();

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
	int ldout = out->cols, ldA = A->cols, ldB = B->cols;
	float *xA = A->data;
	float *xB = B->data;
	float *xout = out->data;
	if (T1){ A_rows = A->cols; A_cols = A->rows; }
	if (T2){ B_cols = B->rows; B_rows = B->cols; }

	OPS->check_matrix_multiplication(A, B, out, T1, T2);

	#pragma offload target(mic:0) \
	in(xA, xB, xout:length(0) alloc_if(0) free_if(0)) \
	in(T1, T2, A_rows, B_cols, A_cols, alpha, beta)
	{

		cblas_sgemm(CblasRowMajor,
				 T1 ? CblasTrans : CblasNoTrans,
				 T2 ? CblasTrans : CblasNoTrans,
				 A_rows, B_cols, A_cols, alpha,
				 xA, ldA, xB, ldB,
				 beta, xout, ldout);
	}
}
