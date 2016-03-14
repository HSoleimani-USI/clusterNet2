/*
 * BasicOpsWrapperCPU.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#include "BasicOpsWrapperCPU.h"




Matrix<float> *fill_matrix(int rows, int cols, float fill_value)
{
	Matrix<float> *ret = new Matrix<float>();
	ret->data = (float*)malloc(sizeof(float)*rows*cols);
	ret->rows = rows;
	ret->cols = cols;
	ret->size = rows*cols;
	ret->bytes = rows*cols*size(float);

	for(int i = 0; i < ret->size; i++)
		ret->data[i] = fill_value;

	return ret;
}