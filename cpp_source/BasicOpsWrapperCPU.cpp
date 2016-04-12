/*
 * BasicOpsWrapperCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "BasicOpsWrapperCPU.h"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

Matrix<float> *BasicOpsWrapperCPU::fill_matrix(int rows, int cols, float fill_value)
{
	Matrix<float> *ret = empty(rows, cols);
	
	int size = ret->size;
	float *data = ret->data;
	#pragma offload target(mic:0) in(size) in(data : length(0) alloc_if(0) free_if(0))
	{
		#pragma omp parallel for
		for(int i = 0; i < size; i++)
			data[i] = fill_value;

	}
	return ret;
}

Matrix<float> *BasicOpsWrapperCPU::empty(int rows, int cols)
{
	Matrix<float> *ret = new Matrix<float>();
	{
		ret->data = (float*)malloc(sizeof(float)*rows*cols);
	}
	ret->rows = rows;
	ret->cols = cols;
	ret->size = rows*cols;
	ret->bytes = rows*cols*sizeof(float);
	ret->isRowMajor = true;
	float *data = ret-> data;
	int size = rows*cols;

	
	#pragma offload target(mic:0) in(size) inout(data: length(size) alloc_if(1) free_if(0)) 
	{
	}
	
	return ret;

}


Matrix<float> *BasicOpsWrapperCPU::zeros(int rows, int cols)
{
	return fill_matrix(rows,cols,0);
}

Matrix<float> *BasicOpsWrapperCPU::ones(int rows, int cols)
{
	return fill_matrix(rows,cols,1);
}

template void BasicOpsWrapperCPU::elementWise<kdropout>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar);
template <int action> void BasicOpsWrapperCPU::elementWise(Matrix<float> *a, Matrix<float> *b, Matrix<float> *c, float scalar)
{
	int size = a->size;
	float *A = a->data;
	float *B = b->data;
	float *out = c->data;

	#pragma offload target(mic:0) \
	in(A,B,out : length(0) alloc_if(0) free_if(0)) \
	in(scalar)

	#pragma omp parallel for
	for(int i=0; i < size ;i++)
	{
		switch(action)
		{
    	   case kdropout: out[i] = B[i] > scalar ? A[i] : 0.0f; break;
		}
	}
}

//BasicOpsWrapperCPU::elementWise operation with a single matrix argument
template void BasicOpsWrapperCPU::elementWise<kabs>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<klog>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<ksqrt>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<klogistic>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<klogistic_grad>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<ktanh>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<ktanh_grad>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<kELU>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<kELU_grad>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<krectified>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<krectified_grad>(Matrix<float> *A, Matrix<float>*out);
template void BasicOpsWrapperCPU::elementWise<kcopy>(Matrix<float> *A, Matrix<float>*out);
template <int action> void BasicOpsWrapperCPU::elementWise(Matrix<float> *a, Matrix<float>*c)
{
		int size = a->size;
		float *A = a->data;
		float *out = c->data;

		#pragma offload target(mic:0) \
		in(A,out : length(0) alloc_if(0) free_if(0))


		#pragma omp parallel for
		for(int i=0; i < size ;i++)
		{
			switch(action)
			{
			   case kabs: out[i] = fabsf(A[i]); break;
			   case klog: out[i] = __logf(A[i]); break;
			   case ksqrt: out[i] = sqrtf(A[i]); break;
			   case kpow: out[i] = powf(A[i],scalar); break;
			   case klogistic: out[i] = 1.0f/(1.0f + expf(-A[i])); break;
			   case klogistic_grad: out[i] = A[i]*(1.0f-A[i]); break;
			   case kELU: out[i] = A[i] > 0.0f ? A[i] : expf(A[i])-1.0f; break;
			   case kELU_grad: out[i] = A[i] > 0.0f ? 1.0f : A[i] + 1.0f; break;
			   case krectified: out[i] = A[i] > 0.0f ? A[i] : 0.0f; break;
			   case krectified_grad: out[i] = A[i] > 0.0f ? 1.0f : 0.0f; break;
			   case kcopy: out[i] = A[i]; break;
			   case ktanh: out[i] = tanhf(A[i]); break;
			   case ktanh_grad: out[i] = 1.0f - (A[i]*A[i]); break;
			   case kexp: out[i] = exp(A[i]); break;

			}
		}
}

template void BasicOpsWrapperCPU::elementWise<kpow>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void BasicOpsWrapperCPU::elementWise<ksmul>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void BasicOpsWrapperCPU::elementWise<kssub>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void BasicOpsWrapperCPU::elementWise<ksgt>(Matrix<float> *A, Matrix<float>*out, float scalar);
template void BasicOpsWrapperCPU::elementWise<kmod>(Matrix<float> *A, Matrix<float>*out, float scalar);
template <int action> void BasicOpsWrapperCPU::elementWise(Matrix<float> *a, Matrix<float>*c, float scalar)
{
		int size = a->size;
		float *A = a->data;
		float *out = c->data;

		#pragma offload target(mic:0) \
		in(A,out : length(0) alloc_if(0) free_if(0)) \
		in(scalar)

		#pragma omp parallel for
		for(int i=0; i < size ;i++)
		{
			switch(action)
			{
			   case kpow: out[i] = powf(A[i],scalar); break;
			   case ksmul: out[i] = A[i] * scalar; break;
			   case kssub: out[i] = A[i] - scalar; break;
			   case ksgt: out[i] = (float)(A[i] > scalar); break;
			   case kmod: out[i] = (float)((int)A[i] % (int)scalar); break;

			}
		}
}

//BasicOpsWrapperCPU::elementWise operation with a two matrix arguments
template void BasicOpsWrapperCPU::elementWise<kadd>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<ksub>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<kdiv>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<kmul>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<keq>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<klt>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<kgt>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<kge>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<kle>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<kne>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template void BasicOpsWrapperCPU::elementWise<ksquared_diff>(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template <int action> void BasicOpsWrapperCPU::elementWise(Matrix<float> *a, Matrix<float> *b, Matrix<float> *c)
{
	int size = a->size;
	float *A = a->data;
	float *B = b->data;
	float *out = c->data;

	#pragma offload target(mic:0) \
	in(A,B,out : length(0) alloc_if(0) free_if(0))


	#pragma omp parallel for
	for(int i=0; i < size ;i++)
	{
		switch(action)
		{
		   case kadd: out[i] = A[i] + B[i]; break;
		   case ksub: out[i] = A[i] - B[i]; break;
		   case kdiv: out[i] = fdividef(A[i], B[i]); break;
		   case kmul: out[i] = A[i] * B[i]; break;
		   case keq: out[i] = (float)(A[i] == B[i]); break;
		   case klt: out[i] = (float)(A[i] < B[i]); break;
		   case kgt: out[i] = (float)(A[i] > B[i]); break;
		   case kge: out[i] = (float)(A[i] >= B[i]); break;
		   case kle: out[i] = (float)(A[i] <= B[i]); break;
		   case kne: out[i] = (float)(A[i] != B[i]); break;
       	   case ksquared_diff: out[i] = powf(A[i]-B[i],2.0f); break;

		}
	}
}


void BasicOpsWrapperCPU::add(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<kadd>(A,B,out); }
void BasicOpsWrapperCPU::sub(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<ksub>(A,B,out); }
void BasicOpsWrapperCPU::div(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<kdiv>(A,B,out); }
void BasicOpsWrapperCPU::mul(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<kmul>(A,B,out); }
void BasicOpsWrapperCPU::equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<keq>(A,B,out); }
void BasicOpsWrapperCPU::less_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<klt>(A,B,out); }
void BasicOpsWrapperCPU::greater_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<kgt>(A,B,out); }
void BasicOpsWrapperCPU::greater_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<kge>(A,B,out); }
void BasicOpsWrapperCPU::less_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<kle>(A,B,out); }
void BasicOpsWrapperCPU::not_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<kne>(A,B,out); }
void BasicOpsWrapperCPU::squared_diff(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ elementWise<ksquared_diff>(A,B,out, 2.0f); }
void BasicOpsWrapperCPU::dropout(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar)
{ elementWise<kdropout>(A,B,out,scalar); }

void BasicOpsWrapperCPU::vadd(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out)
{
	check_matrix_vector_op(A, v);
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] + v->data[i - ((i / out->cols)*out->cols)];
	}
}

void BasicOpsWrapperCPU::vsub(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out)
{
	check_matrix_vector_op(A, v);
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] =  A->data[i] - v->data[i - ((i / out->cols)*out->cols)];
	}
}


void BasicOpsWrapperCPU::get_t_matrix(Matrix<float> *v, Matrix<float> *out)
{
	for(int i=0; i < out->size ;i++)
	{
		out->data[i] = i - ((i / out->cols)*out->cols) == (int)v->data[(i / out->cols)] ? 1.0f : 0.0f;
	}
}





void BasicOpsWrapperCPU::slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend)
{
  
  int rows_out = (rend - rstart);
  int cols_out = (cend - cstart);
  int size = rows_out*cols_out;

  int current_col = 0;
  int offset = 0;
  int current_row = 0;
  for (unsigned int i = 0;i < size; i++)
  {
	  current_row = i / cols_out;
	  current_col = i - (current_row*cols_out);

	  offset = (A->cols * (current_row+rstart)) + current_col + cstart;
	  out->data[i] = A->data[offset];
  }

}


void BasicOpsWrapperCPU::reduceToRowsMean(Matrix<float> *A, Matrix<float> *vout)
{
	reduceToRowsSum(A, vout);
	for(int i = 0; i < vout->size; i++)
	{
		vout->data[i] /= A->cols;
	}
}



void BasicOpsWrapperCPU::reduceToRowsSum(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < vout->size ;i++)
    {
		vout->data[i] = 0.0f;
    }

	for(int i=0; i < A->size ;i++)
		vout->data[i/A->cols] += A->data[i];

}


void BasicOpsWrapperCPU::reduceToRowsMax(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < vout->size ;i++)
    {
		vout->data[i] = -std::numeric_limits<float>::max();
    }
	for(int i=0; i < A->size ;i++)
		vout->data[i/A->cols] = std::max(vout->data[i/A->cols],A->data[i]);
}


void BasicOpsWrapperCPU::reduceToColsMean(Matrix<float> *A, Matrix<float> *vout)
{
	reduceToColsSum(A, vout);
	for(int i = 0; i < vout->size; i++)
		vout->data[i] /= A->rows;
}


void BasicOpsWrapperCPU::reduceToColsSum(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < vout->size ;i++)
    {
		vout->data[i] = 0.0f;
    }
	//must be either rows or cols here; run tests and change this if the tests fail
	for(int i=0; i < A->size ;i++)
		vout->data[i % A->cols] += A->data[i];


}



void BasicOpsWrapperCPU::reduceToColsMax(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < vout->size ;i++)
    {
		vout->data[i] = -std::numeric_limits<float>::max();
    }
	for(int i=0; i < A->size ;i++)
		vout->data[i % A->cols] = std::max(vout->data[i % A->cols],A->data[i]);

}


float BasicOpsWrapperCPU::mean(Matrix<float> *A)
{

	return sum(A)/A->size;

}


float BasicOpsWrapperCPU::sum(Matrix<float> *A)
{

	float *a = A->data;
	float sumValue = 0.0f;
	cout << "size: " << A->size << endl;
	#pragma offload target(mic:0)\
	in(A:length(0) alloc_if(0) free_if(0))\
	inout(sumValue)
	{
		#pragma omp parallel for
		for(int i=0; i < A->size ;i++)
	    {
			sumValue += A->data[i];
		}
	}
	cout << "sum value: " << sumValue  << endl;
	return sumValue;
}

float BasicOpsWrapperCPU::max(Matrix<float> *A)
{

	float maxValue = -std::numeric_limits<float>::max();
	for(int i=0; i < A->size ;i++)
    {
		maxValue = std::max(maxValue,A->data[i]);
	}
	return maxValue;
}




void BasicOpsWrapperCPU::pow(Matrix<float> *A, Matrix<float> *out, float scalar)
{ elementWise<kpow>(A,out, scalar); }


void BasicOpsWrapperCPU::mul(Matrix<float> *A, Matrix<float> *out, float scalar)
{ elementWise<ksmul>(A,out,scalar); }


void BasicOpsWrapperCPU::sub(Matrix<float> *A, Matrix<float> *out, float scalar)
{ elementWise<kssub>(A,out, scalar); }



void BasicOpsWrapperCPU::greater_than(Matrix<float> *A, Matrix<float> *out, float scalar)
{ elementWise<ksgt>(A,out, scalar); }


void BasicOpsWrapperCPU::mod(Matrix<float> *A, Matrix<float> *out, float scalar)
{ elementWise<kmod>(A,out,scalar); }



Matrix<float> *BasicOpsWrapperCPU::transpose(Matrix<float> *A)
{
	Matrix<float> *out = empty(A->cols, A->rows);
	transpose(A, out, A->rows, A->cols);
	return out;
}
void BasicOpsWrapperCPU::transpose(Matrix<float> *A, Matrix<float> *out, int rows, int cols)
{
	for(int row=0; row < rows ;row++)
	{
		for(int col=0; col < cols ;col++)
		{
			out->data[col + row*cols] = A->data[row + col*rows]; 
		}
	}
}

void BasicOpsWrapperCPU::exp(Matrix<float> *A, Matrix<float> *out)
{ elementWise<kexp>(A,out); }
void BasicOpsWrapperCPU::abs(Matrix<float> *A, Matrix<float> *out)
{ elementWise<kabs>(A,out); }
void BasicOpsWrapperCPU::log(Matrix<float> *A, Matrix<float> *out)
{ elementWise<klog>(A,out); }
void BasicOpsWrapperCPU::sqrt(Matrix<float> *A, Matrix<float> *out)
{ elementWise<ksqrt>(A,out); }
void BasicOpsWrapperCPU::logistic(Matrix<float> *A, Matrix<float> *out)
{ elementWise<klogistic>(A,out); }
void BasicOpsWrapperCPU::logistic_grad(Matrix<float> *A, Matrix<float> *out)
{ elementWise<klogistic_grad>(A,out); }
void BasicOpsWrapperCPU::tanh(Matrix<float> *A, Matrix<float> *out)
{ elementWise<ktanh>(A,out); }
void BasicOpsWrapperCPU::tanh_grad(Matrix<float> *A, Matrix<float> *out)
{ elementWise<ktanh_grad>(A,out); }
void BasicOpsWrapperCPU::ELU(Matrix<float> *A, Matrix<float> *out)
{ elementWise<kELU>(A,out); }
void BasicOpsWrapperCPU::ELU_grad(Matrix<float> *A, Matrix<float> *out)
{ elementWise<kELU_grad>(A,out); }
void BasicOpsWrapperCPU::rectified(Matrix<float> *A, Matrix<float> *out)
{ elementWise<krectified>(A,out); }
void BasicOpsWrapperCPU::rectified_grad(Matrix<float> *A, Matrix<float> *out)
{ elementWise<krecitfied_grad>(A,out); }
void BasicOpsWrapperCPU::copy(Matrix<float> *A, Matrix<float> *out)
{ elementWise<kcopy>(A,out); }

void BasicOpsWrapperCPU::softmax(Matrix<float> *A, Matrix<float> *out)
{
	check_for_same_dimensions(A, out);

	Matrix<float> *vsum = empty(A->rows, 1);
	reduceToRowsMax(A, vsum);

	for(int i=0; i < A->size ;i++)
		out->data[i] = A->data[i] - vsum->data[i/A->cols];

	exp(out, out);
	reduceToRowsSum(out, vsum);

	for(int i=0; i < A->size ;i++)
		out->data[i] /= vsum->data[i/A->cols];

	free(vsum);
}




void BasicOpsWrapperCPU::to_host(Matrix<float> *gpu, float *cpu)
{
int size = gpu->size;
float *data = gpu->data;


	#pragma offload target(mic:0) \
	out(data: length(size) alloc_if(0) free_if(0)) 
	{
	}
	std::memcpy(cpu, data, size*sizeof(float));

}
Matrix<float> *BasicOpsWrapperCPU::to_host(Matrix<float> *gpu)
{
	Matrix<float> *out = empty(gpu->rows, gpu->cols);
	to_host(gpu, out->data);
	return out;
}

void BasicOpsWrapperCPU::to_gpu(float *cpu, Matrix<float> *gpu)
{

	float *A = gpu->data;
	int size = gpu->size;
	std::memcpy(gpu->data, cpu, gpu->bytes);
	#pragma offload target(mic:0) \
	in(A: length(size) alloc_if(0) free_if(0)) 
	{
	}
}
Matrix<float> *BasicOpsWrapperCPU::to_pinned(int rows, int cols, float *cpu)
{
	Matrix<float> *out = empty(rows, cols);
	to_host(out, cpu);
	return out;
}
Matrix<float> *BasicOpsWrapperCPU::to_pinned(int rows, int cols, float *cpu, size_t bytes_to_copy)
{
	for(int i =0; i < 10; i++)
		cout << cpu[i] << " ";
	cout << endl;
	Matrix<float> *out = empty(rows, cols);
	std::memcpy(out->data,cpu, bytes_to_copy);
	cout << sum(out) << endl;
	return out;
}
Matrix<float> *BasicOpsWrapperCPU::get_pinned(int rows, int cols){ return empty(rows, cols); }



Matrix<float> *BasicOpsWrapperCPU::to_col_major(Matrix<float> *A)
{
	Matrix<float> *out = empty(A->rows,A->cols);
	to_col_major(A, out);

	return out;
}
void BasicOpsWrapperCPU::to_col_major(Matrix<float> *A, Matrix<float> *out)
{
	if(A->isRowMajor)
		transpose(A, out, A->cols, A->rows);
	else
		copy(A, out);
}

Matrix<float> *BasicOpsWrapperCPU::to_row_major(Matrix<float> *A)
{
	Matrix<float> *out = empty(A->rows,A->cols);

	if(!A->isRowMajor)
			transpose(A, out, A->cols, A->rows);
		else
			copy(A, out);

	return out;
}


void BasicOpsWrapperCPU::WeightUpdate_RMSProp(Matrix<float> *RMS, Matrix<float> *grad, Matrix<float> *w, float RMS_multiplier, float learning_rate)
{
	float rms_reciprocal = 1.0f - RMS_multiplier;
	float grad_value = 0.0f;
	float RMS_value = 0.0f;

	for(int i = 0; i < w->size; i++)
	{
		grad_value = grad->data[i];
		RMS_value = (RMS_multiplier*RMS->data[i]) + (std::pow(grad_value,2.0f)*rms_reciprocal);
		grad_value = learning_rate*grad_value/((std::sqrt(RMS_value)+1.0e-08f));
		RMS->data[i] = RMS_value;
		w->data[i] -= grad_value;
	}
}


void BasicOpsWrapperCPU::free_matrix(Matrix<float> *A){ free(A->data); free(A);}

void BasicOpsWrapperCPU::lookup(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *out){}
void BasicOpsWrapperCPU::embeddingUpdate(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *grad, Matrix<float> *RMS, float RMS_momentum, float learning_rate){}


void BasicOpsWrapperCPU::argmax(Matrix<float> *A, Matrix<float> *out)
{
	Matrix<float> *vmaxbuffer = empty(out->rows, out->cols);
	for(int i=0; i < vmaxbuffer->size ;i++)
	{
		vmaxbuffer->data[i] = -std::numeric_limits<float>::max();
		out->data[i] = -1.0f;
	}

	for(int i=0; i < A->size ;i++)
	{
		if(A->data[i] > vmaxbuffer->data[i/A->cols])
		{
			vmaxbuffer->data[i/A->cols] = A->data[i];
			out->data[i/A->cols] = (float)(i % A->cols);
		}
	}
}

