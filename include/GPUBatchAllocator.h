/*
 * BatchHandler.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tim
 */


#ifndef GPUBatchAllocator_H_
#define GPUBatchAllocator_H_

#include <cuda_runtime_api.h>
#include <BasicOpsWrapper.h>
#include <BatchAllocator.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

class GPUBatchAllocator : public BatchAllocator
{
public:
	GPUBatchAllocator(ClusterNet *gpu);
	GPUBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size);
	void allocate_next_batch_async();
	void replace_current_with_next_batch();
	cudaStream_t streamX;
	cudaStream_t streamY;
};

#endif /* GPUBatchAllocator_H_ */
