#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <iostream>
#include <basicOps.cuh>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda.h>
#include "nervana_c_api.h"

#ifndef __CLUSTERNET2_H__
#define __CLUSTERNET2_H__




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
  	    Matrix<unsigned int> *init_multiplier;
  	    Matrix<unsigned long long> *init_words;

};



#endif //__CLUSTERNET2_H__
