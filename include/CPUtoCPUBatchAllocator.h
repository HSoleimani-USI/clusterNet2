/*
 * CPUtoCPUBatchAllocator.h
 *
 *  Created on: Mar 19, 2016
 *      Author: tim
 */

#ifndef CPUTOCPUBATCHALLOCATOR_H_
#define CPUTOCPUBATCHALLOCATOR_H_

#include <BatchAllocator.h>

class CPUtoCPUBatchAllocator : public BatchAllocator
{
public:
	CPUtoCPUBatchAllocator(ClusterNet *gpu);
	CPUtoCPUBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size);
	void allocate_next_batch_async();
	void replace_current_with_next_batch();
};

#endif /* CPUTOCPUBATCHALLOCATOR_H_ */
