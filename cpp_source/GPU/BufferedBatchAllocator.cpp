/*
 * BatchHandler.cpp
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include "BufferedBatchAllocator.h"
#include <Timer.cuh>
#include <tinydir.h>
#include <stdio.h>
#include <iostream>

using std::endl;
using std::cout;

BufferedBatchAllocator::BufferedBatchAllocator(ClusterNet *gpu){ GPU = gpu;}
BufferedBatchAllocator::BufferedBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size, std::string dir_path)
{
	GPU = gpu;
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

void BufferedBatchAllocator::allocate_next_batch_async()
{
		cudaMemcpyAsync(nextbatchX->data,&batch_bufferX->data[BATCH_SIZE*batch_bufferX->cols*CURRENT_BATCH], nextbatchX->bytes, cudaMemcpyHostToDevice,streamX);
		cudaMemcpyAsync(nextbatchY->data,&batch_bufferY->data[BATCH_SIZE*batch_bufferY->cols*CURRENT_BATCH], nextbatchY->bytes, cudaMemcpyHostToDevice,streamY);
}

void BufferedBatchAllocator::replace_current_with_next_batch()
{

	cudaStreamSynchronize(streamX);
	cudaStreamSynchronize(streamY);
	std::swap(batchX,nextbatchX);
	std::swap(batchY,nextbatchY);

	CURRENT_BATCH += 1;

	if(CURRENT_BATCH == BATCHES)
	{
		CURRENT_BATCH = 0;
		EPOCH += 1;
	}
}


