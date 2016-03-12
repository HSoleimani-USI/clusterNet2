/*
 * Matrix.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */


#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdio.h>

template<typename T> class Matrix
{
  public:
    int rows;
    int cols;
    size_t bytes;
    int size;
    T *data;
    bool isRowMajor;
};

#endif /* MATRIX_H_ */
