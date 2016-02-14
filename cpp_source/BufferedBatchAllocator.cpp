/*
 * BatchHandler.cpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include "BufferedBatchAllocator.h"
#include <Timer.cuh>
#include <tinydir.h>


BufferedBatchAllocator::BufferedBatchAllocator(){}
BufferedBatchAllocator::BufferedBatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size, std::string dir_path)
{
	tinydir_dir dir;
	tinydir_open(&dir, "/home/tim/git/clusterNet2/include/");

	while (dir.has_next)
	{
	    tinydir_file file;
	    tinydir_readfile(&dir, &file);

	    if (!file.is_dir)
	    	cout << file.name << endl;
	    tinydir_next(&dir);
	}

	tinydir_close(&dir);


	BATCH_SIZE = batch_size;
	BATCHES = rows/batch_size;

	batch_bufferX = to_pinned<float>(BATCHES*batch_size,colsX, X, sizeof(float)*BATCHES*batch_size*colsX);
	batch_bufferY = to_pinned<float>(BATCHES*batch_size,colsY, y,sizeof(float)*BATCHES*batch_size*colsY);
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

void BufferedBatchAllocator::allocate_next_batch_async()
{
		cudaMemcpyAsync(nextbatchX->data,&batch_bufferX->data[BATCH_SIZE*batch_bufferX->cols*CURRENT_BATCH], nextbatchX->bytes, cudaMemcpyHostToDevice,streamX);
		cudaMemcpyAsync(nextbatchY->data,&batch_bufferY->data[BATCH_SIZE*batch_bufferY->cols*CURRENT_BATCH], nextbatchY->bytes, cudaMemcpyHostToDevice,streamY);
}

void BufferedBatchAllocator::replace_current_with_next_batch()
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


