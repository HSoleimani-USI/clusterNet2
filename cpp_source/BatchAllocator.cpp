/*
 * BatchHandler.cpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include "BatchAllocator.h"

BatchAllocator::BatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size)
{
	pinned_bufferX = to_pinned<float>(rows,colsX, X);
	pinned_bufferY = to_pinned<float>(rows,colsY, y);

	BATCH_SIZE = batch_size;
	BATCHES = (rows/batch_size) +1;
	OFFBATCH_SIZE = rows - ((BATCHES-1)*BATCH_SIZE);

	CURRENT_BATCH = 0;
	EPOCH = 0;

	batchX = empty<float>(BATCH_SIZE, colsX);
	batchY = empty<float>(BATCH_SIZE, colsY);

	nextbatchX = empty<float>(BATCH_SIZE, colsX);
	nextbatchY = empty<float>(BATCH_SIZE, colsY);
	nextoffbatchX = empty<float>(OFFBATCH_SIZE, colsX);
	nextoffbatchY = empty<float>(OFFBATCH_SIZE, colsY);

	BYTES_X = colsX*BATCH_SIZE*sizeof(float);
	OFFBYTES_X = colsX*OFFBATCH_SIZE*sizeof(float);

	BYTES_Y = colsY*BATCH_SIZE*sizeof(float);
	OFFBYTES_Y = colsY*OFFBATCH_SIZE*sizeof(float);

	cudaStreamCreate(&streamX);
	cudaStreamCreate(&streamY);

}

int BatchAllocator::get_current_batch_size(){
	cout << "a" << endl;
	return CURRENT_BATCH == 0 && EPOCH > 0 ? OFFBATCH_SIZE : BATCH_SIZE; }

void BatchAllocator::allocate_next_batch_async()
{
	cout << "copy to : " << nextbatchX->data << endl;
	cudaMemcpyAsync(nextbatchX->data,&pinned_bufferX->data[BATCH_SIZE*pinned_bufferX->cols*CURRENT_BATCH], nextbatchX->bytes, cudaMemcpyHostToDevice,streamX);
	cudaMemcpyAsync(nextbatchY->data,&pinned_bufferY->data[BATCH_SIZE*pinned_bufferY->cols*CURRENT_BATCH], nextbatchY->bytes, cudaMemcpyHostToDevice,streamY);
}

void BatchAllocator::replace_current_with_next_batch()
{

	cudaStreamSynchronize(streamX);
	cudaStreamSynchronize(streamY);



	if(CURRENT_BATCH < BATCHES-2)
	{

		boost::swap(batchX,nextbatchX);
		cout << CURRENT_BATCH << ": " << "swap normal" << " dim: " << batchX->rows << endl;
		cout << "copy prepared for : " << nextbatchX->data << endl;
		batchX->rows = BATCH_SIZE;

		CURRENT_BATCH += 1;
	}
	else if(CURRENT_BATCH == BATCHES-2)
	{

		boost::swap(batchX,nextbatchX);
		boost::swap(nextbatchX,nextoffbatchX);
		cout << CURRENT_BATCH << ": " << "swap in" << " dim: " << batchX->rows << endl;
		batchX->rows = BATCH_SIZE;
		cout << "copy prepared for : " << nextbatchX->data << endl;

		CURRENT_BATCH += 1;
	}
	else if(CURRENT_BATCH == BATCHES-1)
	{
		boost::swap(batchX,nextoffbatchX);
		boost::swap(nextbatchX,nextoffbatchX);
		batchX->rows = OFFBATCH_SIZE;
		cout << CURRENT_BATCH << ": " << "swap out" << " dim: " << batchX->rows << endl;
		cout << "copy prepared for : " << nextbatchX->data << endl;

		CURRENT_BATCH = 0;
		EPOCH += 1;
	}

	cout << CURRENT_BATCH << ": " << "swap end" << " dim: " << batchX->rows << endl;

}


