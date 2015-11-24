#include "clusterNet2.h"

template ClusterNet2<float>::ClusterNet2();
template<typename T>
ClusterNet2<T>::ClusterNet2()
{
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	m_handle = handle;
}

void workerFunc()
{
 	boost::posix_time::seconds workTime(3);

    std::cout << "Worker: running" << std::endl;

    // Pretend to do something useful...
    boost::this_thread::sleep(workTime);


    Matrix<int> *A = empty<int>(10,10);


    Matrix<int> *C =  fill_matrix<int>(10,10,13);


    Matrix<int> *B = C->to_host();

    for(int i =0; i < 100; i++)
    {
    	std::cout << B->data[i] << std::endl;
    }
    



    std::cout << "Worker: finished" << std::endl;
}
template<typename T>
void ClusterNet2<T>::runThreads()
{
	std::cout << "main: startup" << std::endl;

    boost::thread workerThread(workerFunc);

    std::cout << "main: waiting for thread" << std::endl;

    workerThread.join();

    std::cout << "main: done" << std::endl;
}

template void ClusterNet2<float>::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out);
template <typename T> void ClusterNet2<T>::dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out){ dot(A,B,out,CUBLAS_OP_T,CUBLAS_OP_T); }

template void ClusterNet2<float>::dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, cublasOperation_t T1, cublasOperation_t T2);
template <typename T> void ClusterNet2<T>::dot(Matrix<T> *A, Matrix<T> *B, Matrix<T> *out, cublasOperation_t T1, cublasOperation_t T2)
{
		//if(checkMatrixOperation(A, B, out, T1, T2, 1) == 1){ throw "Matrix *size error:\n"; }
		cublasStatus_t status;
		const float alpha = 1.0f;
		const float beta = 0.0f;
		int A_rows = A->cols, A_cols = A->rows, B_cols = B->rows, B_rows = B->cols;
		if (T1 == CUBLAS_OP_N)
		{
			A_rows = A->rows;
			A_cols = A->cols;
		}
		if (T2 == CUBLAS_OP_N)
		{
			B_cols = B->cols;
			B_rows = B->rows;
		}

		status = cublasSgemm(m_handle, T1, T2, A_rows, B_cols,
				A_cols, &alpha, B->data, A->rows, A->data, B->rows, &beta,
				out->data, out->rows);

		if (status != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "CUBLAS ERROR: Status " << status << std::endl;
			throw "CUBLAS ERROR";
		}
}
