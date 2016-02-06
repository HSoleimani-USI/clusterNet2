/*
 * BatchHandler.cpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include "BatchAllocator.h"


BatchAllocator::BatchAllocator(){}
BatchAllocator::BatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size)
{

	BATCH_SIZE = batch_size;
	BATCHES = rows/batch_size;

	pinned_bufferX = to_pinned<float>(BATCHES*batch_size,colsX, X, sizeof(float)*BATCHES*batch_size*colsX);
	pinned_bufferY = to_pinned<float>(BATCHES*batch_size,colsY, y,sizeof(float)*BATCHES*batch_size*colsY);
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

//we need to fetch the current batch in python, because the struct values with the dimensions does not update automatically
Matrix<float> *BatchAllocator::get_current_batchX()
{ return batchX; }

Matrix<float> *BatchAllocator::get_current_batchY()
{ return batchY; }

void BatchAllocator::allocate_next_batch_async()
{
		cudaMemcpyAsync(nextbatchX->data,&pinned_bufferX->data[BATCH_SIZE*pinned_bufferX->cols*CURRENT_BATCH], nextbatchX->bytes, cudaMemcpyHostToDevice,streamX);
		cudaMemcpyAsync(nextbatchY->data,&pinned_bufferY->data[BATCH_SIZE*pinned_bufferY->cols*CURRENT_BATCH], nextbatchY->bytes, cudaMemcpyHostToDevice,streamY);
}

void BatchAllocator::replace_current_with_next_batch()
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


