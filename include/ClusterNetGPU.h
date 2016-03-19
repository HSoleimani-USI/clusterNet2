/*
 * ClusterNetGPU.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */


#ifndef CLUSTERNETGPU_H_
#define CLUSTERNETGPU_H_

#include <ClusterNet.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda.h>

class ClusterNetGPU : public ClusterNet
{
public:
		ClusterNetGPU();
 		void setRandomState(int seed);
 		Matrix<float> *rand(int rows, int cols);
 		Matrix<float> *randn(int rows, int cols);
 		Matrix<float> *normal(int rows, int cols, float mean, float std);

 	    Matrix<float> *get_uniformsqrt_weight(int input, int output);

 		void Tdot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
 		void dotT(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
 		void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
 		void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2);
 	    void dropout(Matrix<float> *A, Matrix <float> *out, const float dropout);


   private:
 		curandGenerator_t m_generator;
 		cublasHandle_t m_handle;
};

#endif /* CLUSTERNETGPU_H_ */
