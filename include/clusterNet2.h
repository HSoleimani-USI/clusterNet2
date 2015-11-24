#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <iostream>
#include <basicOps.cuh>
#include <cublas_v2.h>

#ifndef __CLUSTERNET2_H__
#define __CLUSTERNET2_H__


template<typename T> class ClusterNet2
{
    public:
        ClusterNet2();
  		void runThreads();

  		void dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out);
  		void dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out, cublasOperation_t T1, cublasOperation_t T2);
  		cublasHandle_t m_handle;

};



#endif //__CLUSTERNET2_H__
