/*
 * BasicOpsWrapperCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "BasicOpsWrapper.h"
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include <vector>
#include <string>
#include <string.h> // memcpy
#include <iostream>
#include <sstream>
#include <fstream>

#ifdef HDF5
	#include <hdf5.h>
#endif

using std::cout;
using std::endl;

void BasicOpsWrapper::printsum(Matrix<float> *A){ printf("%f\n",sum(A)); }
void BasicOpsWrapper::printdim(Matrix<float> *A){ printf("%ix%i\n",A->rows, A->cols); }
void BasicOpsWrapper::printhostmat(Matrix<float> *A){ print_matrix(A,A->rows,A->cols); }
void BasicOpsWrapper::print_matrix(Matrix<float> *A, int end_rows, int end_cols){ print_matrix(A,0,end_rows,0,end_cols); }
void BasicOpsWrapper::print_matrix(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col)
{
	for(int row = start_row; row< end_row; row++)
	{
		printf("[");
		for(int col =start_col; col < end_col; col++)
		{
		  if(A->data[(row*A->cols)+col] < 0.0f)
			  printf("% f ",A->data[(row*A->cols)+col]);
		  else
			  printf("%f ",A->data[(row*A->cols)+col]);
		}
		printf("]\n");
	}
	printf("\n");
}


void BasicOpsWrapper::printmat(Matrix<float> *A)
{
  Matrix<float> * m = to_host(A);
  print_matrix(m,A->rows,A->cols);
  free(m->data);
  free(m);

}

void BasicOpsWrapper::printmat(Matrix<float> *A, int end_rows, int end_cols)
{
  Matrix<float> * m = to_host(A);
  print_matrix(m, end_rows, end_cols);
  free(m->data);
  free(m);

}

void BasicOpsWrapper::printmat(Matrix<float> *A, int start_row, int end_row, int start_col, int end_col)
{
  Matrix<float> * m = to_host(A);
  print_matrix(m, start_row, end_row, start_col, end_col);
  free(m->data);
  free(m);

}




Matrix<float> *BasicOpsWrapper::get_view(Matrix<float> *A, int rstart, int rend)
{
	assert(rstart < A->rows);
	assert(rstart >= 0);
	assert(rend <= A->rows);
	assert(A->isRowMajor);

	Matrix<float> *ret = new Matrix<float>();
	ret->rows = rend-rstart;
	ret->cols = A->cols;
	ret->size = ret->rows*ret->cols;
	ret->bytes = sizeof(float)*ret->size;

	ret->data = &(A->data)[rstart*A->cols];
	ret->isRowMajor = true;

	return ret;
}

bool BasicOpsWrapper::check_for_same_dimensions(Matrix<float> *A, Matrix<float> *B)
{
	if(A && B)
	{
		if(A->rows == B->rows && A->cols == B->cols) return true;
		else
		{
			cout << "Matrices do not have the same dimension: " << A->rows << "x" << A->cols << " vs " << B->rows << "x" << B->cols << endl;
			throw "Matricies do not have same dimension!";
		}
	}
	else
		return true;
}

bool BasicOpsWrapper::check_matrix_multiplication(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2)
{
	int A_rows = A->rows, A_cols = A->cols, B_rows = B->rows, B_cols = B->cols;
	if (T1){ A_rows = A->cols; A_cols = A->rows; }
	if (T2){ B_rows = B->cols; B_cols = B->rows; }

	if(A_rows == out->rows && A_cols == B_rows && B_cols == out->cols) return true;
	else
	{
		cout << "Matrices are not aligned: " << A_rows<< "x" << A_cols << " dot " << B_rows << "x" << B_cols << " -->"  << out->rows << "x" << out->cols <<endl;
		throw "Matrices are not aligned!";
	}

}

bool BasicOpsWrapper::check_matrix_vector_op(Matrix<float> *A, Matrix<float> *vec)
{
	if(A && vec)
	{
		if((A->rows == vec->rows && vec->cols == 1) ||
		   (A->cols == vec->rows && vec->cols == 1) ||
		   (A->rows == vec->cols && vec->rows == 1) ||
		   (A->cols == vec->cols && vec->rows == 1)) return true;
		else
		{
			cout << "Matrix vector opt does not align: " << A->rows << "x" << A->cols << " vs " << vec->rows << "x" << vec->cols << endl;
			throw "Matrix vector opt does not align!";
		}
	}
	else return true;
}


Matrix<float> *BasicOpsWrapper::read_hdf5(const char *filepath){ return read_hdf5(filepath,"/Default"); }
Matrix<float> *BasicOpsWrapper::read_hdf5(const char *filepath, const char *tag)
{
#ifdef HDF5
	   hid_t       file_id, dataset_id;
	   file_id = H5Fopen(filepath, H5F_ACC_RDWR, H5P_DEFAULT);
	   dataset_id = H5Dopen2(file_id, tag, H5P_DEFAULT);

	   hid_t dspace = H5Dget_space(dataset_id);
	   hsize_t dims[2];
	   H5Sget_simple_extent_dims(dspace, dims, NULL);
	   size_t bytes = sizeof(float)*dims[0]*dims[1];

	   Matrix<float> *out = get_pinned((int)dims[0], (int)dims[1]);
	   H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data);


	   H5Dclose(dataset_id);
	   H5Fclose(file_id);


	   return out;
#else
	   return 0;
#endif

}

Matrix<float> *BasicOpsWrapper::read_csv (const char* filename)
{
	std::ifstream  dStream(filename);
	long columns = 0;
	long rows = 0;
	std::vector<float> X;

	std::string line;
	while(std::getline(dStream,line))
	{
		std::stringstream  lineStream(line);
		std::string cell;
		while(std::getline(lineStream,cell,','))
		{
			X.push_back(::atof(cell.c_str()));

			if(rows == 0)
				columns++;
		}
	rows++;
	}

	float *data = (float*)malloc(sizeof(float)*columns*rows);
	memcpy(data,&X[0], columns*rows*sizeof(float));
	Matrix<float> *out = to_pinned(rows, columns, data);

	
	cout << "post pinned data" << endl;
	 for(int i =0; i < 10; i++)
		cout << out->data[i] << " ";
	cout << endl;
	//std::vector<float>().swap( X );

	return out;
}


