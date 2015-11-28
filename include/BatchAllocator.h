/*
 * BatchHandler.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include <cuda_runtime_api.h>
#include <basicOps.cuh>
#include <boost/swap.hpp>


#ifndef BATCHALLOCATOR_H_
#define BATCHALLOCATOR_H_

class BatchAllocator {
public:
	BatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size);
	Matrix<float> *pinned_bufferX;
	Matrix<float> *pinned_bufferY;
	Matrix<float> *batchX;
	Matrix<float> *batchY;

	Matrix<float> *nextbatchX;
	Matrix<float> *nextoffbatchX;
	Matrix<float> *nextbatchY;
	Matrix<float> *nextoffbatchY;
	int BATCH_SIZE;
	int BATCHES;
	int CURRENT_BATCH;
	int EPOCH;


	cudaStream_t streamX;
	cudaStream_t streamY;

	int get_current_batch_size();
	void allocate_next_batch_async();
	void replace_current_with_next_batch();
private:
	int OFFBATCH_SIZE;
	size_t BYTES_X;
	size_t OFFBYTES_X;
	size_t BYTES_Y;
	size_t OFFBYTES_Y;
};

#endif /* BATCHALLOCATOR_H_ */
