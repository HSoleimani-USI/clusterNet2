/*
 * CPUtoCPUBatchAllocator.cpp
 *
 *  Created on: Mar 19, 2016
 *      Author: tim
 */

#include "CPUtoCPUBatchAllocator.h"

using std::cout;
using std::endl;

CPUtoCPUBatchAllocator::CPUtoCPUBatchAllocator(ClusterNet *gpu){GPU = gpu;}
CPUtoCPUBatchAllocator::CPUtoCPUBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size)
{
	GPU = gpu;
	BATCH_SIZE = batch_size;
	BATCHES = rows/batch_size;

	CURRENT_BATCH = 0;
	EPOCH = 0;

	batchX = GPU->OPS->empty(BATCH_SIZE, colsX);
	batchY = GPU->OPS->empty(BATCH_SIZE, colsY);

	nextbatchX = GPU->OPS->empty(BATCH_SIZE, colsX);
	nextbatchY = GPU->OPS->empty(BATCH_SIZE, colsY);

	batch_bufferX = GPU->OPS->empty(rows, colsX);
	batch_bufferY = GPU->OPS->empty(rows, colsY);

	batch_bufferX->data = X;
	batch_bufferY->data = y;

	//batch_bufferX = GPU->OPS->to_pinned(BATCHES*batch_size,colsX, X, sizeof(float)*BATCHES*batch_size*colsX);
	//batch_bufferY = GPU->OPS->to_pinned(BATCHES*batch_size,colsY, y,sizeof(float)*BATCHES*batch_size*colsY);

	//for(int i = 0; i < 10; i++)
	//	cout << batch_bufferY->data[i] << " ";
	//cout << endl;

	//GPU->OPS->to_gpu(batch_bufferX->data, batch_bufferX);
	//GPU->OPS->to_gpu(batch_bufferY->data, batch_bufferY);

	//cout << GPU->OPS->sum(batch_bufferX) << endl;
	//cout << GPU->OPS->sum(batch_bufferY) << endl;
	//GPU->OPS->printdim(batch_bufferX);
}



void CPUtoCPUBatchAllocator::allocate_next_batch_async(){}
void CPUtoCPUBatchAllocator::replace_current_with_next_batch()
{
	batchX = GPU->OPS->get_view(batch_bufferX, CURRENT_BATCH*BATCH_SIZE, (CURRENT_BATCH+1)*BATCH_SIZE);
	batchY = GPU->OPS->get_view(batch_bufferY, CURRENT_BATCH*BATCH_SIZE, (CURRENT_BATCH+1)*BATCH_SIZE);

	//GPU->OPS->to_gpu(GPU->OPS->get_view(batch_bufferX, CURRENT_BATCH*BATCH_SIZE, (CURRENT_BATCH+1)*BATCH_SIZE)->data, batchX);
	//GPU->OPS->to_gpu(GPU->OPS->get_view(batch_bufferY, CURRENT_BATCH*BATCH_SIZE, (CURRENT_BATCH+1)*BATCH_SIZE)->data, batchY);

	CURRENT_BATCH += 1;
	if(CURRENT_BATCH == BATCHES)
	{
		CURRENT_BATCH = 0;
		EPOCH += 1;
	}
}
