/*
 * ClusterNetCPU.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#ifndef CLUSTERNETCPU_H_
#define CLUSTERNETCPU_H_

#include <Matrix.h>
#include <ClusterNet.h>

#ifdef PHI
	#include <mkl_vsl.h>
#endif

class ClusterNetCPU : public ClusterNet
{
public:
	ClusterNetCPU();
	~ClusterNetCPU(){};

	void setRandomState(int seed);
	Matrix<float> *rand(int rows, int cols);
	Matrix<float> *randn(int rows, int cols);
	Matrix<float> *normal(int rows, int cols, float mean, float std);

	//  initializing the weight of network
	Matrix<float> *get_uniformsqrt_weight(int input, int output);

	//  matrix multliplication A*B = out
	void Tdot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	void dotT(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	// nothing transpose
	void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
	//  define if the first and second are transpose
	void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2);
	//  for regularization , using random numbers
	//  getting the simplest model ...random between 0 and 1 , matrix
	void dropout(Matrix<float> *A, Matrix <float> *out, const float dropout);

#ifdef PHI
	VSLStreamStatePtr rdm_uniform;
	VSLStreamStatePtr rdm_standard_normal;
	VSLStreamStatePtr rdm_normal;
#endif
};

#endif /* CLUSTERNETCPU_H_ */
