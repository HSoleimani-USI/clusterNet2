/*
 * BasicOpsWrapperCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "BasicOpsWrapperCPU.h"
#include <cmath>
#include <cstdlib>




Matrix<float> *BasicOpsWrapperCPU::fill_matrix(int rows, int cols, float fill_value)
{
	Matrix<float> *ret = empty(rows, cols);
	
	for(int i = 0; i < ret->size; i++)
		ret->data[i] = fill_value;

	return ret;
}

Matrix<float> *BasicOpsWrapperCPU::empty(int rows, int cols)
{
	Matrix<float> *ret = new Matrix<float>();
	ret->data = (float*)malloc(sizeof(float)*rows*cols);
	ret->rows = rows;
	ret->cols = cols;
	ret->size = rows*cols;
	ret->bytes = rows*cols*sizeof(float);

	
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
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] + B->data[i];
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
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] + v->data[i - ((i / out->cols)*out->cols)];
	}
}

void BasicOpsWrapperCPU::vsub(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out)
{
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
  return 0;

}


void mean_of_cols(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < A->size ;i++)
	{
		vout->data = median(A->data[i],'c');
	}


}



void sum_of_cols(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < A->size ;i++)
	{
		vout->data[i] = vout->data[i]+A->data[i];
    }
 
    return 0;

}


void max_of_cols(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < A->size ; i++)
	{
		out->data = max(A->data[i],[],1)
	}
	return 0;
}


void mean_of_rows(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < A->size ;i++)
	{
		vout->data = median(A->data[i],'r');
	}

	return 0;

}


void sum_of_rows(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < A->size ;i++)
    {
		vout->data[i] = 0;
	    for(int j=0; j < A->size ;i++)
	  	vout->data[i] = vout->data[i] + A->data[i][j];
    }

    return 0;

}



void max_of_rows(Matrix<float> *A, Matrix<float> *vout)
{
	for(int i=0; i < A->size ; i++)
	{
		out->data = max(A->data[i],[],1)
	}
	return 0;

}


float mean(Matrix<float> *A)
{

	for(int i=0; i < A->size ;i++)
    {
    	sum = sum + A->data[i];

    {
    average = sum / (float)A->size);
    }

}


float sum(Matrix<float> *A)
{
	for(int i=0; i < A->size ;i++)
    {
	totalsum = sum_of_rows(*A) + sum_of_cols(*A)
	}
}























