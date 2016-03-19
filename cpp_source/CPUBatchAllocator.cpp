/*
 * BatchHandler.cpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include "CPUBatchAllocator.h"


CPUBatchAllocator::CPUBatchAllocator(ClusterNet *gpu){GPU = gpu;}
CPUBatchAllocator::CPUBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size)
{
	GPU = gpu;
	BATCH_SIZE = batch_size;
	BATCHES = rows/batch_size;

	batch_bufferX = GPU->OPS->to_pinned(BATCHES*batch_size,colsX, X, sizeof(float)*BATCHES*batch_size*colsX);
	batch_bufferY = GPU->OPS->to_pinned(BATCHES*batch_size,colsY, y,sizeof(float)*BATCHES*batch_size*colsY);
	CURRENT_BATCH = 0;
	EPOCH = 0;

	batchX = GPU->OPS->empty(BATCH_SIZE, colsX);
	batchY = GPU->OPS->empty(BATCH_SIZE, colsY);

	nextbatchX = GPU->OPS->empty(BATCH_SIZE, colsX);
	nextbatchY = GPU->OPS->empty(BATCH_SIZE, colsY);

	cudaStreamCreate(&streamX);
	cudaStreamCreate(&streamY);

	allocate_next_batch_async();

}

void CPUBatchAllocator::allocate_next_batch_async()
{
		cudaMemcpyAsync(nextbatchX->data,&batch_bufferX->data[BATCH_SIZE*batch_bufferX->cols*CURRENT_BATCH], nextbatchX->bytes, cudaMemcpyHostToDevice,streamX);
		cudaMemcpyAsync(nextbatchY->data,&batch_bufferY->data[BATCH_SIZE*batch_bufferY->cols*CURRENT_BATCH], nextbatchY->bytes, cudaMemcpyHostToDevice,streamY);
}

void CPUBatchAllocator::replace_current_with_next_batch()
{

	cudaStreamSynchronize(streamX);
	cudaStreamSynchronize(streamY);
	boost::swap(batchX,nextbatchX);
	boost::swap(batchY,nextbatchY);

	CURRENT_BATCH += 1;

	if(CURRENT_BATCH == BATCHES)
	{
		CURRENT_BATCH = 0;
		EPOCH += 1;
	}
}


