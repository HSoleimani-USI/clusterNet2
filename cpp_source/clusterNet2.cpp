#include "clusterNet2.h"
#include <stdlib.h>
#include "leveldb/db.h"
#include <iostream>     // std::cout
#include "json.hpp"

// for convenience
using json = nlohmann::json;

using std::cout;
using std::endl;

template ClusterNet2<float>::ClusterNet2();
template<typename T>
ClusterNet2<T>::ClusterNet2()
{
    cudaError_t res = cudaFree(0);
    if (res != cudaSuccess)
    {
        std::cout << "CUDA did not initialize correctly" << std::endl;
        exit(1);
    }

    if (!nervana_loadKernels("../cubin/"))
    {
        std::cerr << "Couldn't load all kernels" << std::endl;
        exit(1);
    }

	setRandomState(time(0));
	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, time(0));
	curandSetGeneratorOffset(m_generator, 100);


	/*
	leveldb::DB* db;
	  leveldb::Options options;
	  options.create_if_missing = true;
	  leveldb::Status status = leveldb::DB::Open(options, "/home/tim/wiki/raw_pages", &db);

	  std::string value;
	  leveldb::Status s = db->Get(leveldb::ReadOptions(), "raw_pages/npydata", &value);

	  cout << "print" << endl;
	  cout << value << endl;
	  auto j = json::parse(value);

	  cout << j.is_array() << endl;

	  float *data = (float*)j.get<double*>();

	  for(int i = 0; i < 10; i ++)
		  cout << data[i] << " ";
	  */
}



template void ClusterNet2<float>::setRandomState(int seed);
template<typename T> void ClusterNet2<T>::setRandomState(int seed)
{
	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, seed);
	curandSetGeneratorOffset(m_generator, 100);
}

template Matrix<float> *ClusterNet2<float>::rand(int rows, int cols);
template <typename T> Matrix<T> *ClusterNet2<T>::rand(int rows, int cols)
{
	Matrix<T> *out = empty<T>(rows, cols);
	curandGenerateUniform(m_generator, out->data, rows * cols);

	return out;
}

template Matrix<float> *ClusterNet2<float>::randn(int rows, int cols);
template <typename T> Matrix<T> *ClusterNet2<T>::randn(int rows, int cols){ return normal(rows, cols, 0.0f, 1.0f); }

template Matrix<float> *ClusterNet2<float>::normal(int rows, int cols, float mean, float std);
template <typename T> Matrix<T> *ClusterNet2<T>::normal(int rows, int cols, float mean, float std)
{
	Matrix<T> *out = empty<T>(rows, cols);
	curandGenerateNormal(m_generator, out->data, rows * cols, mean, std);

	return out;
}

template void ClusterNet2<float>::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template <typename T> void ClusterNet2<T>::dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out){ dot(A,B,out,false,false); }

template void ClusterNet2<float>::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2);
template <typename T> void ClusterNet2<T>::dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out, bool T1, bool T2)
{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		int A_rows = A->rows, A_cols = A->cols, B_cols = B->cols;
		if (T1){ A_rows = A->cols; A_cols = A->rows; }
		if (T2){ B_cols = B->rows; }

		bool success = nervana_sgemm(A->data, B->data, out->data, T1,T2,
									 A_rows, B_cols, A_cols,
									 A->cols,B->cols,out->cols,
									 alpha,beta,
									 NULL, false, false,0);


		if (!success){ throw "NERVANA ERROR"; }
}
