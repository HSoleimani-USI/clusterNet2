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
	#pragma offload target(mic:0) in(size) in(data : length(size) alloc_if(0) free_if(0))
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


void BasicOpsWrapperCPU::add(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
	int size = A->size;
	float *a = A->data;
	float *b = B->data;
	float *c = out->data;
	#pragma offload target(mic:0) \ 
	in(a : length(size) alloc_if(0) free_if(0)) \
	in(b : length(size) alloc_if(0) free_if(0) )\
	in(c : length(size) alloc_if(0) free_if(0)) \
	in(size)
	{

		#pragma omp parallel for
		for(int i=0; i < size ;i++)
		{
			c[i] = a[i] + b[i];
		}
	}

}

void BasicOpsWrapperCPU::sub(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] - B->data[i];
	}

}


void BasicOpsWrapperCPU::div(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] / B->data[i];
	}
}


void BasicOpsWrapperCPU::mul(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ 

	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] * B->data[i];
	}
}


void BasicOpsWrapperCPU::equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
	for(int i=0; i < A->size ;i++)
	{
 		 out->data[i] = (float)(A->data[i] == B->data[i]);

 	}
}


void BasicOpsWrapperCPU::less_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] < B->data[i]);
	}
}



void BasicOpsWrapperCPU::greater_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] > B->data[i]);
	}
}


void BasicOpsWrapperCPU::greater_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] >= B->data[i]);
	}
}


void BasicOpsWrapperCPU::less_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] <= B->data[i]);
	}
}



void BasicOpsWrapperCPU::not_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] != B->data[i]);
	}
}



void BasicOpsWrapperCPU::squared_diff(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = std::pow(A->data[i] - B->data[i],2.0f);
	}
}


void BasicOpsWrapperCPU::dropout(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = B->data[i] > scalar ? A->data[i] : 0.0f;
	}
}


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

	float sumValue = 0.0f;
	cout << "size: " << A->size << endl;
	#pragma omp parallel for
	for(int i=0; i < A->size ;i++)
    {
		sumValue += A->data[i];
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
{
	for(int i=0; i < A->size ;i++)
    {
		out->data[i] = powf(A->data[i],scalar);
	}
}


void BasicOpsWrapperCPU::mul(Matrix<float> *A, Matrix<float> *out, float scalar)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] * scalar;
	}

}


void BasicOpsWrapperCPU::sub(Matrix<float> *A, Matrix<float> *out, float scalar)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] - scalar;
	}

}



void BasicOpsWrapperCPU::greater_than(Matrix<float> *A, Matrix<float> *out, float scalar)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] > scalar);
	}
}


void BasicOpsWrapperCPU::mod(Matrix<float> *A, Matrix<float> *out, float scalar)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)((int)A->data[i] % (int)scalar);
	}

}



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
{ for(int i=0; i < A->size ;i++){ out->data[i] = ::exp(A->data[i]); } }
void BasicOpsWrapperCPU::abs(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = std::abs(A->data[i]); } }
void BasicOpsWrapperCPU::log(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = std::log(A->data[i]); } }
void BasicOpsWrapperCPU::sqrt(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = std::sqrt(A->data[i]); } }
void BasicOpsWrapperCPU::logistic(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = 1.0f/(1.0f+::exp(-A->data[i])); } }
void BasicOpsWrapperCPU::logistic_grad(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = A->data[i]*(1.0f-A->data[i]); } }
void BasicOpsWrapperCPU::tanh(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = std::tanh(A->data[i]); } }
void BasicOpsWrapperCPU::tanh_grad(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = 1.0f - (A->data[i]*A->data[i]); } }
void BasicOpsWrapperCPU::ELU(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = A->data[i] > 0.0f ? A->data[i] : expf(A->data[i])-1.0f; } }
void BasicOpsWrapperCPU::ELU_grad(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = A->data[i] > 0.0f ? 1.0f : A->data[i] + 1.0f;} }
void BasicOpsWrapperCPU::rectified(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = A->data[i] > 0.0f ? A->data[i] : 0.0f;} }
void BasicOpsWrapperCPU::rectified_grad(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = A->data[i] > 0.0f ? 1.0f : 0.0f;} }
void BasicOpsWrapperCPU::copy(Matrix<float> *A, Matrix<float> *out)
{ for(int i=0; i < A->size ;i++){ out->data[i] = A->data[i];} }

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

