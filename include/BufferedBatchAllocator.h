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
#include <string>


#ifndef BufferedBatchAllocator_H_
#define BufferedBatchAllocator_H_

class BufferedBatchAllocator : public BatchAllocator
{
public:
	BufferedBatchAllocator();
	BufferedBatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size, std::string dir_path);
	void allocate_next_batch_async();
	void replace_current_with_next_batch();
};

#endif /* BufferedBatchAllocator_H_ */
