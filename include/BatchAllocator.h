/*
 * BatchHandler.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */

#include <cuda_runtime_api.h>
#include <basicOps.cuh>
#include <boost/swap.hpp>


#ifndef BatchAllocator_H_
#define BatchAllocator_H_

class BatchAllocator {
public:
	BatchAllocator(){};
	virtual ~BatchAllocator(){};

	Matrix<float> *batch_bufferX;
	Matrix<float> *batch_bufferY;
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

	Matrix<float> *get_current_batchX();
	Matrix<float> *get_current_batchY();

	virtual void allocate_next_batch_async() = 0;
	virtual void replace_current_with_next_batch() = 0;
};

#endif /* BatchAllocator_H_ */
