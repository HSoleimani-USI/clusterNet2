#include <stdio.h>
	#include <mkl.h>
	#include <mkl_vsl.h>

template<typename T> class Matrix
{
  public:
    int rows;
    int cols;
    size_t bytes;
    long size;
    T *data;
    bool isRowMajor;
};

Matrix<float> *empty(int rows, int cols)
{
	Matrix<float> *ret = new Matrix<float>();
	{
		ret->data = (float*)_mm_malloc(sizeof(float)*rows*cols, 64);
	}
	ret->rows = rows;
	ret->cols = cols;
	ret->size = rows*cols;
	ret->bytes = rows*cols*sizeof(float);
	ret->isRowMajor = true;
	float *data = ret-> data;
	int size = rows*cols;

#ifdef PHI
	#pragma offload target(mic:0) in(size) inout(data: length(size) alloc_if(1) free_if(0))
	{
	}
#endif

	return ret;

}

Matrix<float> *rand(int rows, int cols)
{
	Matrix<float> *ret = empty(rows,cols);

	int size = ret->size;
	float *xret = ret->data;

	int seed = ::rand();

#ifdef PHI
	#pragma offload target(mic:0) \
	in(xret : length(0) alloc_if(0) free_if(0)) \
	in(size, seed) 
{ 

	vslNewStream(&rdm_uniform,	VSL_BRNG_MT19937,seed );
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,	rdm_uniform, size, xret ,0.0f, 1.0f);
}
#else
	#pragma omp parallel for
	for(int i = 0; i < size; i++)
		xret[i] =(float)((double) ::rand() / (RAND_MAX));
#endif



	return ret;
}


using std::endl;
using std::cout;


void dot(Matrix<float> *A, Matrix<float> *B, Matrix<float> *out, bool T1, bool T2)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	int A_rows = A->rows, A_cols = A->cols, B_rows = B->rows, B_cols = B->cols;
	int ldout = out->cols, ldA = A->cols, ldB = B->cols;
	float *xA = A->data;
	float *xB = B->data;
	float *xout = out->data;
	if (T1){ A_rows = A->cols; A_cols = A->rows; }
	if (T2){ B_cols = B->rows; B_rows = B->cols; }

        const char chrT1 = T1 ? 'N' : 'T';
        const char chrT2 = T2 ? 'N' : 'T';

	//OPS->check_matrix_multiplication(A, B, out, T1, T2);

#ifdef PHI
	__assume_aligned(xA,64);
	__assume_aligned(xB,64);
	__assume_aligned(xout,64);
	#pragma offload target(mic:0) \
	in(xA, xB, xout:length(0) alloc_if(0) free_if(0)) \
	in(T1, T2, A_rows, B_cols, A_cols, alpha, beta) \
	in(ldA,ldB, ldout)
#endif
	{

#ifdef PHI
		cblas_sgemm(CblasRowMajor,
				 T1 ? CblasTrans : CblasNoTrans,
				 T2 ? CblasTrans : CblasNoTrans,
				 A_rows, B_cols, A_cols, alpha,
				 xA, ldA, xB, ldB,
				 beta, xout, ldout);
#endif
	}
}



void test_nonvectorized()
{
	ClusterNet *acc = new ClusterNetCPU();

	int size = 1200;

	Matrix<float> *a = acc->rand(128,10);
	Matrix<float> *b = acc->rand(128,10);


	Matrix<float> *A = acc->OPS->zeros(size,size);
	Matrix<float> *B = acc->OPS->zeros(size,size);
	Matrix<float> *C = acc->OPS->zeros(size,size);



	//warm up
	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);


	double t0 = omp_get_wtime();
	acc->OPS->softmax(a,b);
	cout << "time softmax: " << omp_get_wtime()-t0 << endl;
	

	Matrix<float> *y = acc->OPS->read_csv("/home/dettmers/data/y.csv");

	Matrix<float> *X = acc->OPS->zeros(60000,10);

	//warm up
	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);


	t0 = omp_get_wtime();
	acc->OPS->get_t_matrix(y,X);
	cout << "time t matrix: " << omp_get_wtime()-t0 << endl;


	Matrix<float> *x = acc->rand(128,784);
	Matrix<float> *w1 = acc->rand(784,1200);
	Matrix<float> *a1 = acc->rand(128,1200);

	Matrix<float> *w2 = acc->rand(1200,1200);
	Matrix<float> *a2 = acc->rand(128,1200);
	Matrix<float> *w3 = acc->rand(1200,10);
	Matrix<float> *a3 = acc->rand(128,10);


	Matrix<float> *w4 = acc->rand(1200,1200);
	Matrix<float> *a4 = acc->rand(128,1200);


	Matrix<float> *w5 = acc->rand(1200,1200);
	Matrix<float> *a5 = acc->rand(128,1200);

	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);

	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
		acc->dot(x,w1,a1);
		acc->dot(a1,w2,a2);
		acc->dot(a2,w3,a3);
	}
	cout << "time forward dot full: " << omp_get_wtime()-t0 << endl;
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
		acc->dot(a1,w2,a2);
		acc->dot(a2,w4,a4);
		acc->dot(a4,w5,a5);
	}
	cout << "time forward dot 1200 only: " << omp_get_wtime()-t0 << endl;
	
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
		acc->dot(x,w1,a1);
		acc->dot(a1,w2,a2);
		acc->dot(a2,w4,a4);
		acc->dot(a4,w5,a5);
		acc->dot(a5,w3,a3);
	}
	cout << "time forward dot full + 3x1200: " << omp_get_wtime()-t0 << endl;
	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
	acc->dot(x,w1,a1);
	}
	cout << "time forward dot 784: " << omp_get_wtime()-t0 << endl;
	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
	acc->dot(a1,w2,a2);
	}
	cout << "time forward dot 1200: " << omp_get_wtime()-t0 << endl;
	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
	acc->dot(a2,w3,a3);
	}
	cout << "time forward dot 10: " << omp_get_wtime()-t0 << endl;

	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
	acc->dot(x,w1,a1);
	acc->dot(a1,w2,a2);
	}
	cout << "time forward dot 784 + 1200: " << omp_get_wtime()-t0 << endl;

	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
	acc->dot(a1,w2,a2);
	acc->dot(a2,w3,a3);
}
	cout << "time forward dot 1200 + 10: " << omp_get_wtime()-t0 << endl;


	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
	acc->dot(x,w1,a1);
	acc->dot(a2,w3,a3);
	}
	cout << "time forward dot 784 + 10: " << omp_get_wtime()-t0 << endl;


	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
		acc->rand(128,1200);
		acc->rand(128,1200);
	}
	cout << "random same size: " << omp_get_wtime()-t0 << endl;

	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);
	t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
	{
		acc->rand(128,1200);
		acc->rand(128,600);
		acc->rand(128,300);
	}
	cout << "random same size: " << omp_get_wtime()-t0 << endl;
} 

int main(int argc, char *argv[])
 {

	test_nonvectorized();


	return 0;
}
