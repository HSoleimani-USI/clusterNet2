#include "clusterNet2.h"
#include <stdlib.h>

template ClusterNet2<float>::ClusterNet2();
template<typename T>
ClusterNet2<T>::ClusterNet2()
{
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	m_handle = handle;
	setRandomState(time(0));

	curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(m_generator, time(0));
	curandSetGeneratorOffset(m_generator, 100);
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
template <typename T> void ClusterNet2<T>::dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out){ dot(A,B,out,CUBLAS_OP_N,CUBLAS_OP_N); }

template void ClusterNet2<float>::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, cublasOperation_t T1, cublasOperation_t T2);
template <typename T> void ClusterNet2<T>::dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out, cublasOperation_t T1, cublasOperation_t T2)
{
		cublasStatus_t status;
		const float alpha = 1.0f;
		const float beta = 0.0f;
		int A_rows = A->rows, A_cols = A->cols, B_cols = B->cols;
		if (T1 == CUBLAS_OP_T)
		{
			A_rows = A->cols;
			A_cols = A->rows;
		}
		if (T2 == CUBLAS_OP_T)
			B_cols = B->rows;


		status = cublasSgemm(m_handle, T1, T2, A_rows, B_cols,
				A_cols, &alpha, A->data, A->rows, B->data, B->rows, &beta,
				out->data, out->rows);

		if (status != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "CUBLAS ERROR: Status " << status << std::endl;
			throw "CUBLAS ERROR";
		}
}
