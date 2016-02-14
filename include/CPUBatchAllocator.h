/*
 * BatchHandler.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include <cuda_runtime_api.h>
#include <basicOps.cuh>
#include <boost/swap.hpp>
#include <BatchAllocator.h>


#ifndef CPUBatchAllocator_H_
#define CPUBatchAllocator_H_

class CPUBatchAllocator : public BatchAllocator
{
public:
	CPUBatchAllocator();
	CPUBatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size);
	void allocate_next_batch_async();
	void replace_current_with_next_batch();
};

#endif /* CPUBatchAllocator_H_ */
