/*
 * BatchHandler.cpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include "GPUBatchAllocator.h"
#include <Timer.cuh>


GPUBatchAllocator::GPUBatchAllocator(){}
GPUBatchAllocator::GPUBatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size)
{

	BATCH_SIZE = batch_size;
	BATCHES = rows/batch_size;

	batch_bufferX = empty<float>(BATCHES*batch_size,colsX);
	batch_bufferY = empty<float>(BATCHES*batch_size,colsY);

	CUDA_CHECK_RETURN(cudaMemcpy(batch_bufferX->data, X,sizeof(float)*BATCHES*batch_size*colsX, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(batch_bufferY->data, y,sizeof(float)*BATCHES*batch_size*colsY, cudaMemcpyHostToDevice));

	CURRENT_BATCH = 0;
	EPOCH = 0;

	batchX = empty<float>(BATCH_SIZE, colsX);
	batchY = empty<float>(BATCH_SIZE, colsY);

	nextbatchX = empty<float>(BATCH_SIZE, colsX);
	nextbatchY = empty<float>(BATCH_SIZE, colsY);

	cudaStreamCreate(&streamX);
	cudaStreamCreate(&streamY);

	allocate_next_batch_async();

}


void GPUBatchAllocator::allocate_next_batch_async(){}
void GPUBatchAllocator::replace_current_with_next_batch()
{
	batchX = get_view(batch_bufferX, CURRENT_BATCH*BATCH_SIZE, (CURRENT_BATCH+1)*BATCH_SIZE);
	batchY = get_view(batch_bufferY, CURRENT_BATCH*BATCH_SIZE, (CURRENT_BATCH+1)*BATCH_SIZE);

	CURRENT_BATCH += 1;
	if(CURRENT_BATCH == BATCHES)
	{
		CURRENT_BATCH = 0;
		EPOCH += 1;
	}
}


