/*
 * ClusterNetCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "ClusterNetCPU.h"

#ifdef PHI
	#include <mkl.h>
	#include <mkl_vsl.h>
#endif
#ifdef CPU
	#include <cblas.h>
#endif

#include <BasicOpsWrapperCPU.h>
#include <math.h>

ClusterNetCPU::ClusterNetCPU()
{
	// TODO Auto-generated constructor stub
	OPS = new BasicOpsWrapperCPU();

#ifdef PHI
	// Initialize RNG
	vslNewStream(&rdm_uniform,	VSL_BRNG_MT19937,1 );
	vslNewStream(&rdm_standard_normal,	VSL_BRNG_MT19937,1 );
	vslNewStream(&rdm_normal,	VSL_BRNG_MT19937,1 );
	//vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, rdm_standard_normal, N, 0.0f,1.0f);
#endif


}

void ClusterNetCPU::setRandomState(int seed)
{
	srand(seed);
}

Matrix<float> *ClusterNetCPU::rand(int rows, int cols)
{
	Matrix<float> *ret = OPS->empty(rows,cols);

	int size = ret->size;
	float *xret = ret->data;

	int seed = ::rand();

#ifdef PHI
	#pragma offload target(mic:0) \
	in(xret : length(0) alloc_if(0) free_if(0)) \
	in(size, seed) 
{ 

	vslNewStream(&rdm_uniform,	VSL_BRNG_MT19937,seed );
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,	rdm_uniform, size, xret ,0.0f, 1.0f);
}
#else
	#pragma omp parallel for
	for(int i = 0; i < size; i++)
		xret[i] =(float)((double) ::rand() / (RAND_MAX));
#endif



	return ret;
}

Matrix<float> *ClusterNetCPU::randn(int rows, int cols)
{
	Matrix<float> *ret = OPS->empty(rows,cols);

	int size = ret->size;
	float *xret = ret->data;
	int seed = ::rand();

 #ifdef PHI
	#pragma offload target(mic:0) \
	in(xret : length(0) alloc_if(0) free_if(0)) \
	in(size, seed) 
{ 

	//vslNewStream(&rdm_standard_normal,	VSL_BRNG_MT19937,seed );
	//vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, rdm_standard_normal, size, xret, 0.0f,1.0f);
	vslNewStream(&rdm_uniform,	VSL_BRNG_MT19937,seed );
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,	rdm_uniform, size, xret ,0.0f, 1.0f);

	#pragma omp parallel for
	for(int i = 0; i < size; i++)
	{
		float rdm = xret[i];
		xret[i] = rdm > 0.5f ? sqrtf(-1.57079632679*logf(1-powf((2*rdm-1),2))) :
				    -sqrtf(-1.57079632679*logf(1-powf((1-2*rdm),2)));
	}

}
#else

	#pragma omp parallel for
	for(int i = 0; i < size; i++)
	{
		float rdm = (float)((double) ::rand() / (RAND_MAX) );
		xret[i] = rdm > 0.5f ? sqrtf(-1.57079632679*logf(1-powf((2*rdm-1),2))) :
				    -sqrtf(-1.57079632679*logf(1-powf((1-2*rdm),2)));
	}
#endif
	return ret;
}


Matrix<float> *ClusterNetCPU::normal(int rows, int cols, float mean, float std)
{
	Matrix<float> *ret = OPS->empty(rows,cols);
	int size = ret->size;
	float *xret = ret->data;
	int seed = ::rand();

 #ifdef PHI
	#pragma offload target(mic:0) \
	in(xret : length(0) alloc_if(0) free_if(0)) \
	in(size, seed, mean, std) 
{ 

	//vslNewStream(&rdm_normal,	VSL_BRNG_MT19937,seed );
	//vRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, rdm_standard_normal, size, xret, mean,std);
	vslNewStream(&rdm_uniform,	VSL_BRNG_MT19937,seed );
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,	rdm_uniform, size, xret ,0.0f, 1.0f);

	#pragma omp parallel for
	for(int i = 0; i < size; i++)
	{
		float rdm = xret[i];
		xret[i] =  1.0f/(1.0f + expf((-0.07056* (rdm*rdm*rdm)) - (1.5976*rdm)));
	}

}
#else



	#pragma omp parallel for
	for(int i = 0; i < size; i++)
	{
		float rdm = (float)((double) ::rand() / (RAND_MAX) * (2));
		xret[i] =  1.0f/(1.0f + expf((-0.07056* (rdm*rdm*rdm)) - (1.5976*rdm)));
	}
#endif

	return ret;
}

void ClusterNetCPU::dropout(Matrix<float> *A, Matrix <float> *out, const float dropout)
{
        int size = out->size;
	float *xout = out->data;

#ifdef PHI
	#pragma offload target(mic:0) \
	in(xout : length(0) alloc_if(0) free_if(0)) \
        in(size)
#endif

	#pragma omp parallel for
	for(int i = 0; i < size; i++)
		xout[i] = (float)((double) ::rand() / (RAND_MAX) * (2));

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

        const char chrT1 = T1 ? 'N' : 'T';
        const char chrT2 = T2 ? 'N' : 'T';

	//OPS->check_matrix_multiplication(A, B, out, T1, T2);

#ifdef PHI
	__assume_aligned(xA,64);
	__assume_aligned(xB,64);
	__assume_aligned(xout,64);
	#pragma offload target(mic:0) \
	in(xA, xB, xout:length(0) alloc_if(0) free_if(0)) \
	in(T1, T2, A_rows, B_cols, A_cols, alpha, beta) \
	in(ldA,ldB, ldout)
#endif
	{

#ifdef PHI
		cblas_sgemm(CblasRowMajor,
				 T1 ? CblasTrans : CblasNoTrans,
				 T2 ? CblasTrans : CblasNoTrans,
				 A_rows, B_cols, A_cols, alpha,
				 xA, ldA, xB, ldB,
				 beta, xout, ldout);
#endif
	}
}
