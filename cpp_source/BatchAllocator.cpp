/*
 * BatchHandler.cpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include "BatchAllocator.h"
#include <Timer.cuh>


//we need to fetch the current batch in python, because the struct values with the dimensions does not update automatically
Matrix<float> *BatchAllocator::get_current_batchX()
{ return batchX; }

Matrix<float> *BatchAllocator::get_current_batchY()
{ return batchY; }
