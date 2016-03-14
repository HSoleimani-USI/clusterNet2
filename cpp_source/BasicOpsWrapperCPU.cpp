/*
 * BasicOpsWrapperCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "BasicOpsWrapperCPU.h"




Matrix<float> *fill_matrix(int rows, int cols, float fill_value)
{
	Matrix<float> *ret = empty(rows, cols);
	
	for(int i = 0; i < ret->size; i++)
		ret->data[i] = fill_value;

	return ret;
}

Matrix<float> *empty(int rows, int cols)
{
	Matrix<float> *ret = new Matrix<float>();
	ret->data = (float*)malloc(sizeof(float)*rows*cols);
	ret->rows = rows;
	ret->cols = cols;
	ret->size = rows*cols;
	ret->bytes = rows*cols*size(float);

	
	return ret;

}


Matrix<float> *zeros(int rows, int cols) 
{

	Matrix<float> *ret = fill_matrix(rows,cols,0)
	
	
	return ret;
}

Matrix<float> *ones(int rows, int cols) 
{

	Matrix<float> *ret = fill_matrix(rows,cols,1)
	
	
	return ret;
}


void add(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] + B->data[i];
	}
	return 0;

}

void sub(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] - B->data[i];
	}
	return 0;

}


void div(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{
	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] / B->data[i];
	}
    return 0;
}


void mul(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{ 

	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] * B->data[i];
	}
    return 0;
}


void equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out) 
{
	for(int i=0; i < A->size ;i++)
	{
 		 out->data[i] = (float)(A->data[i] == B->data[i]);

 	}
    return 0;
}


void less_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] < B->data[i]);
	}
	return 0;
}



void greater_than(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] > B->data[i]);
	}
	return 0;
}


void greater_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] >= B->data[i]);
	}
	return 0;
}


void less_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] <= B->data[i]);
	}
	return 0;
}



void not_equal(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = (float)(A->data[i] != B->data[i]);
	}
	return 0;
}



void squared_diff(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = powf(A->data[i] - B->data[i],2.0f);
	}
	return 0;
}


void dropout(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, float scalar)
{	
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = B->data[i] > scalar ? A->data[i] : 0.0f;
	}
	return 0;
}


void vadd(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = A->data[i] + v->data[i - ((i / cols)*cols)];
	}
	return 0;
}

void vsub(Matrix<float> *A, Matrix<float> *v, Matrix<float> *out)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] =  A->data[i] - v->data[i - ((i / cols)*cols)];
	}
	return 0;
}


void get_t_matrix(Matrix<float> *v, Matrix<float> *out)
{
	for(int i=0; i < A->size ;i++)
	{
		out->data[i] = i - ((i / cols)*cols) == (int)v->data[(i / cols)] ? 1.0f : 0.0f;
	}
	return 0;
}





void slice(Matrix<float> *A, Matrix<float>*out, int rstart, int rend, int cstart, int cend)
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

























