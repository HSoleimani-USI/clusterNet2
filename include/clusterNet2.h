#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <iostream>
#include <basicOps.cuh>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda.h>
#include "nervana_c_api.h"
#include "leveldb/db.h"

#ifndef __CLUSTERNET2_H__
#define __CLUSTERNET2_H__

typedef enum Unittype_t
{
	Logistic = 0,
	Rectified_Linear = 1,
	Softmax = 2,
	Linear = 4,
	Input = 8
} Unittype_t;

typedef enum DataPropagationType_t
{
	Training = 0,
	Trainerror = 1,
	CVerror = 2
} DataPropagationType_t;


typedef enum WeightUpdateType_t
{
	RMSProp = 0,
	Momentum = 1,
	PlainSGD = 2
} WeightUpdateType_t;

typedef enum Costfunction_t
{
	Cross_Entropy = 0,
	Squared_Error = 1,
	Root_Squared_Error = 2,
	Misclassification = 4
} Costfunction_t;


template<typename T> class ClusterNet2
{
    public:
        ClusterNet2();
  		void setRandomState(int seed);
  		Matrix<T> *rand(int rows, int cols);
  		Matrix<T> *randn(int rows, int cols);
  		Matrix<T> *normal(int rows, int cols, float mean, float std);
  		void dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out);
  		void dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out, bool T1, bool T2);
  		curandGenerator_t m_generator;
  	    void dropout(Matrix<T> *A, Matrix <T> *out, const float dropout);

};



#endif //__CLUSTERNET2_H__
