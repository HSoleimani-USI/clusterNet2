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
		in(A,out : length(0) alloc_if(0) free_if(0)) \


		#pragma omp parallel for
		for(int i=0; i < size ;i++)
		{
			switch(action)
			{
			   case kabs: out[i] = fabsf(A[i]); break;
			   case klog: out[i] = __logf(A[i]); break;
			   case ksqrt: out[i] = sqrtf(A[i]); break;
			   case klogistic: out[i] = 1.0f/(1.0f + expf(-A[i])); break;
			   case klogistic_grad: out[i] = A[i]*(1.0f-A[i]); break;
			   case kELU: out[i] = A[i] > 0.0f ? A[i] : expf(A[i])-1.0f; break;
			   case kELU_grad: out[i] = A[i] > 0.0f ? 1.0f : A[i] + 1.0f; break;
			   case krectified: out[i] = A[i] > 0.0f ? A[i] : 0.0f; break;
			   case krectified_grad: out[i] = A[i] > 0.0f ? 1.0f : 0.0f; break;
			   case kcopy: out[i] = A[i]; break;
			   case ktanh: out[i] = tanhf(A[i]); break;
			   case ktanh_grad: out[i] = 1.0f - (A[i]*A[i]); break;
			   case kexp: out[i] = expf(A[i]); break;

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
		   case kdiv: out[i] = A[i] / B[i]; break;
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



//vectorwise operation between matrix and vector
//this is equivalent to broadcasting in numpy
template void BasicOpsWrapperCPU::vectorWise<kvadd>(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out);
template void BasicOpsWrapperCPU::vectorWise<kvsub>(Matrix<float> *A, Matrix<float> *v, Matrix<float>*out);
template <int action> void BasicOpsWrapperCPU::vectorWise(Matrix<float> *a, Matrix<float> *b, Matrix<float>*c)
{
	int size = a->size;
	float *A = a->data;
	float *v = b->data;
	float *out = c->data;
	int rows = c->rows;
	int cols = c->cols;

	#pragma offload target(mic:0) \
	in(A,v,out : length(0) alloc_if(0) free_if(0)) \
	in(size, cols, rows)


	#pragma omp parallel for
	for(int i = 0; i < size ;i++)
	{
		switch(action)
		{
			case kvadd: out[i] =  A[i] + v[i - ((i / cols)*cols)]; break;
			case kvsub: out[i] =  A[i] - v[i - ((i / cols)*cols)]; break;
		}
	}
}

template void BasicOpsWrapperCPU::vectorWise<ktmatrix>(Matrix<float> *v, Matrix<float>*out);
template <int action> void BasicOpsWrapperCPU::vectorWise(Matrix<float> *a, Matrix<float>*c)
{
	int size = a->size;
	float *v = a->data;
	float *out = c->data;
	int rows = c->rows;
	int cols = c->cols;

	#pragma offload target(mic:0) \
	in(v,out : length(0) alloc_if(0) free_if(0)) \
	in(size, cols, rows)


	#pragma omp parallel for
	for(int i = 0; i < size ;i++)
	{
		switch(action)
		{
			case ktmatrix: out[i] = i-((i / cols)*cols) == (int)v[(i / cols)] ? 1.0f : 0.0f; break;
		}
	}
}


void BasicOpsWrapperCPU::vadd(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out)
{ vectorWise<kadd>(A,v,out); }

void BasicOpsWrapperCPU::vsub(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out)
{ vectorWise<ksub>(A,v,out); }

void BasicOpsWrapperCPU::get_t_matrix(Matrix<float> *v, Matrix<float> *out)
{ vectorWise<ktmatrix>(v,out); }


void BasicOpsWrapperCPU::slice(Matrix<float> *a, Matrix<float>*c, int rstart, int rend, int cstart, int cend)
{
  
	int rows_out = (rend - rstart);
	int cols_out = (cend - cstart);
	int size = rows_out*cols_out;

	int cols = a->cols;
	float *A = a->data;
	float *out = c->data;


	#pragma offload target(mic:0) \
	in(A,out : length(0) alloc_if(0) free_if(0)) \
	in(size, rstart, rend, cstart, cend, cols_out, rows_out, cols)

	#pragma omp parallel for
	for (unsigned int i = 0;i < size; i++)
	{
	  int current_row = i / cols_out;
	  int current_col = i - (current_row*cols_out);

	  int offset = (cols * (current_row+rstart)) + current_col + cstart;
	  out[i] = A[offset];
	}

}


void BasicOpsWrapperCPU::reduceToRowsMean(Matrix<float> *A, Matrix<float> *vout)
{
	reduceToRowsSum(A, vout);

	float *out = vout->data;
	int size = vout->size;
	int cols = A->cols;

	#pragma offload target(mic:0)\
	in(out:length(0) alloc_if(0) free_if(0)) \
	in(size)

	#pragma omp parallel for
	for(int i = 0; i < size; i++)
	{
		out[i] /= cols;
	}
}



void BasicOpsWrapperCPU::reduceToRowsSum(Matrix<float> *a, Matrix<float> *vout)
{

	float *A = a->data;
	float *out = vout->data;
	int vsize = vout->size;
	int Asize = a->size;
	int cols = a->cols;

	#pragma offload target(mic:0)\
	in(A,out:length(0) alloc_if(0) free_if(0)) \
	in(vsize,Asize)
	{

		#pragma omp parallel for
		for(int i=0; i < vsize ;i++)
		{
			out[i] = 0.0f;
		}

		#pragma omp parallel for
		for(int i=0; i < Asize ;i++)
			out[i/cols] += A[i];
	}

}


void BasicOpsWrapperCPU::reduceToRowsMax(Matrix<float> *a, Matrix<float> *vout)
{

	float *A = a->data;
	float *out = vout->data;
	int vsize = vout->size;
	int Asize = a->size;
	int cols = a->cols;

	#pragma offload target(mic:0)\
	in(A,out:length(0) alloc_if(0) free_if(0)) \
	in(vsize,Asize)
	{

		#pragma omp parallel for
		for(int i=0; i < vsize ;i++)
		{
			out[i] = -std::numeric_limits<float>::max();
		}

		#pragma omp parallel for
		for(int i=0; i < Asize ;i++)
			out[i/cols] = std::max(out[i/cols],A[i]);
	}

}


void BasicOpsWrapperCPU::reduceToColsMean(Matrix<float> *a, Matrix<float> *vout)
{
	reduceToColsSum(a, vout);

	float *out = vout->data;
	int size = vout->size;
	int rows = a->rows;

	#pragma offload target(mic:0)\
	in(out:length(0) alloc_if(0) free_if(0)) \
	in(size)

	#pragma omp parallel for
	for(int i = 0; i < size; i++)
		out[i] /= rows;
}


void BasicOpsWrapperCPU::reduceToColsSum(Matrix<float> *a, Matrix<float> *vout)
{


	float *A = a->data;
	float *out = vout->data;
	int vsize = vout->size;
	int Asize = a->size;
	int cols = a->cols;

	#pragma offload target(mic:0)\
	in(A,out:length(0) alloc_if(0) free_if(0)) \
	in(vsize,Asize)
	{

		#pragma omp parallel for
		for(int i=0; i < vsize ;i++)
		{
			out[i] = 0.0f;
		}
		//must be either rows or cols here; run tests and change this if the tests fail

		#pragma omp parallel for
		for(int i=0; i < Asize ;i++)
			out[i % cols] += A[i];
	}

}



void BasicOpsWrapperCPU::reduceToColsMax(Matrix<float> *a, Matrix<float> *vout)
{
	float *A = a->data;
	float *out = vout->data;
	int vsize = vout->size;
	int Asize = a->size;
	int cols = a->cols;

	#pragma offload target(mic:0)\
	in(A,out:length(0) alloc_if(0) free_if(0)) \
	in(vsize,Asize)
	{

		#pragma omp parallel for
		for(int i=0; i < vsize ;i++)
		{
			out[i] = -std::numeric_limits<float>::max();
		}

		#pragma omp parallel for
		for(int i=0; i < Asize ;i++)
			out[i % cols] = std::max(out[i % cols],A[i]);

	}

}


float BasicOpsWrapperCPU::mean(Matrix<float> *A)
{

	return sum(A)/A->size;

}


float BasicOpsWrapperCPU::sum(Matrix<float> *a)
{

	float *A = a->data;
	float sumValue = 0.0f;
	int size = a->size;

	#pragma offload target(mic:0)\
	in(A:length(0) alloc_if(0) free_if(0)) \
	in(size) \
	inout(sumValue)

	{
		#pragma omp parallel for
		for(int i=0; i < size ;i++)
	    {
			sumValue += A[i];
		}
	}
	return sumValue;
}

float BasicOpsWrapperCPU::max(Matrix<float> *a)
{

	float *A = a->data;
	float maxValue = -std::numeric_limits<float>::max();
	int size = a->size;

	#pragma offload target(mic:0)\
	in(A:length(0) alloc_if(0) free_if(0)) \
	in(size) \
	inout(maxValue)

	#pragma omp parallel for
	for(int i=0; i < size ;i++)
    {
		maxValue = std::max(maxValue,A[i]);
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
void BasicOpsWrapperCPU::transpose(Matrix<float> *a, Matrix<float> *c, int rows, int cols)
{
	float *A = a->data;
	float *out = c->data;
	int size = a->size;


	#pragma offload target(mic:0)\
	in(A:length(0) alloc_if(0) free_if(0)) \
	in(size, rows, cols)

	#pragma omp parallel for
	for(int row=0; row < rows ;row++)
	{
		for(int col=0; col < cols ;col++)
		{
			out[col + row*cols] = A[row + col*rows];
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
{ elementWise<krectified_grad>(A,out); }
void BasicOpsWrapperCPU::copy(Matrix<float> *A, Matrix<float> *out)
{ elementWise<kcopy>(A,out); }

void BasicOpsWrapperCPU::softmax(Matrix<float> *a, Matrix<float> *c)
{
	check_for_same_dimensions(a, c);

	Matrix<float> *Vsum = empty(a->rows, 1);
	reduceToRowsMax(a, Vsum);


	float *A = a->data;
	float *out = c->data;
	float *vsum = Vsum->data;
	int size = a->size;
	int cols = a->cols;

	int vsum_size = Vsum->size;

	#pragma offload target(mic:0)\
	in(A,out,vsum:length(0) alloc_if(0) free_if(0)) \
	in(size, cols)
	{
		#pragma omp parallel for
		for(int i=0; i < size ;i++)
			out[i] = A[i] - vsum[i/cols];

	}
		exp(c, c);
		reduceToRowsSum(c, Vsum);


	#pragma offload target(mic:0)\
	in(A,out,vsum:length(0) alloc_if(0) free_if(0)) \
	in(size, cols)
	{
		#pragma omp parallel for
		for(int i=0; i < size ;i++)
			out[i] /= vsum[i/cols];

	}

	//free vsum
	#pragma offload target(mic:0)\
	in(vsum:length(vsum_size) alloc_if(0) free_if(1))
	{
	}
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
	Matrix<float> *out = empty(rows, cols);
	std::memcpy(out->data,cpu, bytes_to_copy);
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



	float *xRMS = RMS->data;
	float *xgrad = grad->data;
	float *xw = w->data;
	int size = w->size;

	#pragma offload target(mic:0)\
	in(xRMS,xgrad,xw:length(0) alloc_if(0) free_if(0)) \
	in(size, rms_reciprocal, grad_value, RMS_value)

	#pragma omp parallel for
	for(int i = 0; i < size; i++)
	{
		grad_value = xgrad[i];
		RMS_value = (RMS_multiplier*xRMS[i]) + (std::pow(grad_value,2.0f)*rms_reciprocal);
		grad_value = learning_rate*grad_value/((std::sqrt(RMS_value)+1.0e-08f));
		xRMS[i] = RMS_value;
		xw[i] -= grad_value;
	}
}


void BasicOpsWrapperCPU::free_matrix(Matrix<float> *A)
{
	float *xA = A->data;
	int size = A->size;

	#pragma offload target(mic:0)\
	in(xA:length(size) alloc_if(0) free_if(1))
	{
	}
}

void BasicOpsWrapperCPU::lookup(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *out){}
void BasicOpsWrapperCPU::embeddingUpdate(Matrix<float> *embedding, Matrix<float> *idx_batch, Matrix<float> *grad, Matrix<float> *RMS, float RMS_momentum, float learning_rate){}


void BasicOpsWrapperCPU::argmax(Matrix<float> *A, Matrix<float> *out)
{

	Matrix<float> *vmaxbuffer = empty(out->rows, out->cols);

	float *xvmaxbuffer = vmaxbuffer->data;

	float *xA = A->data;
	float *xout = out->data;
	int size = A->size;
	int cols = A->cols;
	int vmaxbuffer_size = vmaxbuffer->size;

	#pragma offload target(mic:0)\
	in(xA,xout,xvmaxbuffer:length(0) alloc_if(0) free_if(0)) \
	in(size, cols,vmaxbuffer_size)
	{
		#pragma omp parallel for
		for(int i=0; i < vmaxbuffer_size ;i++)
		{
			xvmaxbuffer[i] = -std::numeric_limits<float>::max();
			xout[i] = -1.0f;
		}

		#pragma omp parallel for
		for(int i=0; i < size ;i++)
		{
			if(xA[i] > xvmaxbuffer[i/cols])
			{
				xvmaxbuffer[i/cols] = xA[i];
				xout[i/cols] = (float)(i % cols);
			}
		}
	}
}

