/*
 * BatchHandler.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include <cuda_runtime_api.h>
#include <BasicOpsCUDA.cuh>
#include <boost/swap.hpp>
#include <BatchAllocator.h>


#ifndef GPUBatchAllocator_H_
#define GPUBatchAllocator_H_

class GPUBatchAllocator : public BatchAllocator
{
public:
	GPUBatchAllocator();
	GPUBatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size);
	void allocate_next_batch_async();
	void replace_current_with_next_batch();
};

#endif /* GPUBatchAllocator_H_ */
