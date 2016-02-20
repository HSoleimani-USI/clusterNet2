#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <iostream>
#include <basicOps.cuh>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda.h>
#include "nervana_c_api.h"
#include "leveldb/db.h"


#ifndef __ClusterNet_H__
#define __ClusterNet_H__

typedef enum TransformerType_t
{
	DropoutTransform = 0
} TransformerType_t;

typedef enum WeightInitType_t
{
	Gaussian = 0,
	UniformSqrt = 1
} WeightInitType_t;

typedef enum Unittype_t
{
	Logistic = 0,
	Rectified_Linear = 1,
	Softmax = 2,
	Linear = 3,
	Input = 4,
	Exponential_linear = 5,
	Output

} Unittype_t;

typedef enum DataPropagationType_t
{
	Training = 0,
	Trainerror = 1,
	CVerror = 2
} DataPropagationType_t;


typedef enum Costfunction_t
{
	Cross_Entropy = 0,
	Squared_Error = 1,
	Root_Squared_Error = 2,
	Misclassification = 4
} Costfunction_t;

typedef enum Layer_t
{
	FullyConnected = 0,
	LSTM = 1,
	Lookup = 2,
	InputLayer = 3,
	OutputLayer = 4
} Layer_t;

class ClusterNet
{
    public:
        ClusterNet();
		bool useNervanaGPU;

  		void setRandomState(int seed);
  		Matrix<float> *rand(int rows, int cols);
  		Matrix<float> *randn(int rows, int cols);
  		Matrix<float> *normal(int rows, int cols, float mean, float std);

  		void Tdot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
  		void dotT(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
  		void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
  		void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2);
  	    void dropout(Matrix<float> *A, Matrix <float> *out, const float dropout);

    private:
  		curandGenerator_t m_generator;
  		cublasHandle_t m_handle;

};



#endif //__ClusterNet_H__
