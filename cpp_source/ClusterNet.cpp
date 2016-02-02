#include "ClusterNet.h"
#include <stdlib.h>
#include "leveldb/db.h"
#include <iostream>     // std::cout
#include "json.hpp"


#include <unistd.h>


// for convenience
using json = nlohmann::json;

using std::cout;
using std::endl;

ClusterNet::ClusterNet()
{
    cudaError_t res = cudaFree(0);
    if (res != cudaSuccess)
    {
        std::cout << "CUDA did not initialize correctly" << std::endl;
        exit(1);
    }

    if (!nervana_loadKernels("/usr/local/cuda/cubin/"))
    {
        std::cerr << "Couldn't load all kernels" << std::endl;
        exit(1);
    }

	setRandomState(time(0));
	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, time(0));//random seed
	curandSetGeneratorOffset(m_generator, 100);//burn in the rdm generator


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



void ClusterNet::setRandomState(int seed)
{
	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, seed);
	curandSetGeneratorOffset(m_generator, 100);
}

Matrix<float> *ClusterNet::rand(int rows, int cols)
{
	Matrix<float> *out = empty<float>(rows, cols);
	curandGenerateUniform(m_generator, out->data, rows * cols);

	return out;
}

Matrix<float> *ClusterNet::randn(int rows, int cols){ return normal(rows, cols, 0.0f, 1.0f); }
Matrix<float> *ClusterNet::normal(int rows, int cols, float mean, float std)
{
	Matrix<float> *out = empty<float>(rows, cols);
	curandGenerateNormal(m_generator, out->data, out->size, mean, std);

	return out;
}


void ClusterNet::Tdot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ dot(A,B,out,true,false); }
void ClusterNet::dotT(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ dot(A,B,out,false,true); }
void ClusterNet::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out){ dot(A,B,out,false,false); }
void ClusterNet::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2)
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

void ClusterNet::dropout(Matrix<float> *A, Matrix <float> *out, const float dropout)
{
	curandGenerateUniform(m_generator, out->data, out->size);
	elementWise<kdropout>(A, out, out, dropout);
}


