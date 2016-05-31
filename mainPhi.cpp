#include "ClusterNet.h"
#include <stdio.h>
#include <ClusterNetCPU.h>
//#include <ClusterNetGPU.h>
//#include "Timer.cuh"
#include "Network.h"
#include <Optimizer.h>
#include <FCLayer.h>
#include <BatchAllocator.h>
//#include <CPUtoCPUBatchAllocator.h>
//#include <GPUBatchAllocator.h>
//#include <BufferedBatchAllocator.h>
#include <ErrorHandler.h>
#include <Configurator.h>
#include <string>
#include <map>
#include <BasicOpsWrapperCPU.h>
#include <CPUtoCPUBatchAllocator.h>
#include <gradientAccumulator.h>

#include <mpi.h>
#include <omp.h>
#include <freader.h>

#include <vector>

using std::endl;
using std::cout;

void test_timer()
{
	ClusterNet *gpu = new ClusterNetCPU();
	Matrix<float> *A = gpu->rand(100,100);
	Matrix<float> *B = gpu->rand(100,100);
	Matrix<float> *C = gpu->rand(100,100);

	/*
	Timer t = Timer();

	t.tick();
	for(int i = 0; i < 10000; i++)
		gpu->dot(A,B,C);
	t.tock();
	*/
}

void test_transfer_time()
{
	ClusterNet *gpu = new ClusterNetCPU();
	int batch_size = 32;
	int time_steps = 100;

	Matrix<float> *data = gpu->rand(128,batch_size*time_steps);
	Matrix<float> *host = gpu->OPS->to_host(data);
	Matrix<float> *pinned = gpu->OPS->to_pinned(128,batch_size*time_steps,data->data);


	Matrix<float> *error = gpu->rand(batch_size*time_steps,256);
	Matrix<float> *w = gpu->rand(128,256);

	/*
	Timer t = Timer();

	t.tick();
	for(int i = 0; i < 100; i++)
		gpu->OPS->to_gpu(host->data,data);
	t.tock();

	t.tick();
	for(int i = 0; i < 100; i++)
		gpu->OPS->to_gpu(pinned->data,data);
	t.tock();


	t.tick();
	for(int i = 0; i < 100; i++)
		gpu->dot(data,error,w);
	t.tock();
	*/

}

void test_LSTM_swapping()
{

	ClusterNet *gpu = new ClusterNetCPU();

	int hidden = 256;
	int timesteps = 1000;
	int batch_size = 32;
	int stack_size = 4;

	Matrix<float> *R = gpu->rand(hidden*stack_size,hidden);
	Matrix<float> *inputR = gpu->rand(batch_size*stack_size,hidden);
	Matrix<float> *errorsR = gpu->rand(batch_size*stack_size,hidden);

	/*
	Timer t = Timer();


	for(int i = 0; i < 1000; i++)
		gpu->dot(inputR,errorsR,R);

	t.tick();

	for(int i = 0; i < 10000; i++)
		gpu->dot(inputR,errorsR,R);

	t.tock();
	*/
}

void test_lookup_time()
{

	ClusterNet *gpu = new ClusterNetCPU();

	int embedding_rows = 10000;
	int embedding_cols = 256;
	int batch_size = 128;
	int batch_cols = 1024;

	Matrix<float> *embedding = gpu->rand(embedding_rows,embedding_cols);
	Matrix<float> *batch = gpu->rand(batch_size*batch_cols,embedding_cols);
	Matrix<float> *batch1 = gpu->rand(batch_size*batch_cols,embedding_cols);

	/*
	Timer t = Timer();
	t.tick();
	for(int i = 0; i < 1000; i++)
		cudaMemcpy(batch1->data, batch->data,batch->bytes,cudaMemcpyDeviceToDevice);
	t.tock();
	*/

	/*

	Matrix<float> *inputR = gpu->rand(hidden,batch_size);
	Matrix<float> *errorsR = gpu->rand(batch_size,hidden*stack_size);

	Timer t = Timer();


	for(int i = 0; i < 1000; i++)
		gpu->dot(inputR,errorsR,R);

	t.tick();

	for(int i = 0; i < 10000; i++)
		gpu->dot(inputR,errorsR,R);

	t.tock();
	*/
}

void test_phi()
{

	float test[4] = {1.7,1.3,1.2,1.1};
	float test2[4] = {1.7,1.3,1.2,1.1};
	ClusterNet *gpu = new ClusterNetCPU();

	cout << "before offload" << endl;
	Matrix<float>* a = gpu->OPS->zeros(2,2);
	cout << "after offload" << endl;
	Matrix<float>* b = gpu->OPS->ones(2,2);
	Matrix<float>* c = gpu->OPS->empty(2,2);


	gpu->OPS->to_host(a, test);
	gpu->OPS->to_host(b, test);
	gpu->OPS->to_host(c, test);
	gpu->OPS->to_gpu(test2, a);
	gpu->OPS->to_host(a, test);

	gpu->OPS->add(a,b,c);

	gpu->OPS->to_host(c, test);

	for(int i =0; i < 4; i++)
		cout << test[i] << " ";
	cout << endl;

	cout << gpu->OPS->sum(a) << endl;
	cout << gpu->OPS->sum(b) << endl;
	cout << gpu->OPS->sum(c) << endl;
	cout << "Sum should be: " << 4 + 1.7+1.3+1.2+1.1 << endl;


}

void test_neural_network_MPI()
{

//	test_transfer_time();

	//Timer t = Timer();
	printf("init clusternet cpu\n");
	ClusterNet *gpu = new ClusterNetCPU();



//	gpu->useNervanaGPU = true;

	  int matrix_rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &matrix_rank);

	printf("loading data\n");
	Matrix<float> *X = gpu->OPS->read_csv("/home/dettmers/data/X.csv");
	Matrix<float> *y = gpu->OPS->read_csv("/home/dettmers/data/y.csv");


	for(int i = 0; i < 20; i++)
	cout << y->data[i] << " ";
	cout << endl;

	int samples = X->rows;
	int cv = 10000;
	int dim = X->cols;
	int classes = 10;

	Matrix<float> *trainX;
	Matrix<float> *trainy;

	Matrix<float> *cvX;  
	Matrix<float> *cvy;  

	if(matrix_rank==0)
	{
		trainX = gpu->OPS->zeros(30000,dim);
		trainy = gpu->OPS->zeros(30000,1);
		cvX = gpu->OPS->zeros(10000,dim);
		cvy = gpu->OPS->zeros(10000,1);

		gpu->OPS->slice(X,trainX,0,30000,0,dim);
		gpu->OPS->slice(y,trainy,0,30000,0,1);
		gpu->OPS->slice(X,cvX,samples-cv,samples,0,dim);
		gpu->OPS->slice(y,cvy,samples-cv,samples,0,1);
	}
	else
	{
		trainX = gpu->OPS->zeros(30000,dim);
		trainy = gpu->OPS->zeros(30000,1);
		cvX = gpu->OPS->zeros(10000,dim);
		cvy = gpu->OPS->zeros(10000,1);
		gpu->OPS->slice(X,trainX,30000,60000,0,dim);
		gpu->OPS->slice(y,trainy,30000,60000,0,1);
		gpu->OPS->slice(X,cvX,samples-cv,samples,0,dim);
		gpu->OPS->slice(y,cvy,samples-cv,samples,0,1);
	}
	gpu->OPS->mul(trainX,trainX,1.0f/255.0f);

	gpu->OPS->mul(cvX,cvX,1.0f/255.0f);

	//gpu->OPS->to_host(trainX,trainX->data);
	//gpu->OPS->to_host(cvX,cvX->data);

	cout << gpu->OPS->max(cvX) << endl;

	BatchAllocator *b_train = new CPUtoCPUBatchAllocator(gpu, trainX->data, trainy->data, trainX->rows, trainX->cols,trainy->cols,128);
	BatchAllocator *b_cv = new CPUtoCPUBatchAllocator(gpu, cvX->data, cvy->data, cvX->rows, cvX->cols,cvy->cols,128);

	Network net = Network(gpu);

	net._conf->LEARNING_RATE = 0.003f;
	net._conf->RMSPROP_MOMENTUM = 0.99f;

	net.add(new FCLayer(784,Input));
	net.add(new FCLayer(1200,Exponential_linear));
	net.add(new FCLayer(1200,Exponential_linear));
	net.add(new FCLayer(10,Softmax));

	for(int i = 0; i < net._layers.size(); i++)
		net._layers[i]->_transformer.clear();

	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.2f;

	net._opt = new Optimizer(gpu, RMSProp);
	net.init_weights(UniformSqrt);





	//t.tick();
	double t0 = omp_get_wtime();
	cout << "pre train" << endl;
	net.train(b_train, b_cv, 2);

	net._conf->DROPOUT = 0.25f;
	net._conf->LEARNING_RATE *= 0.2f;
	net._conf->LEARNING_RATE_DECAY = 0.95f;
	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.1f;

	
	//net.train(b_train, b_cv, 11);
	cout << "time: " << omp_get_wtime()-t0 << endl;

	//t.tock();
}

void test_neural_network()
{

//	test_transfer_time();

	//Timer t = Timer();
	printf("init clusternet cpu\n");
	ClusterNet *gpu = new ClusterNetCPU();



//	gpu->useNervanaGPU = true;


	printf("loading data\n");
	Matrix<float> *X = gpu->OPS->read_csv("/home/dettmers/data/X.csv");
	Matrix<float> *y = gpu->OPS->read_csv("/home/dettmers/data/y.csv");


	for(int i = 0; i < 20; i++)
	cout << y->data[i] << " ";
	cout << endl;

	int samples = X->rows;
	int cv = 10000;
	int dim = X->cols;
	int classes = 10;

	Matrix<float> *trainX = gpu->OPS->zeros(samples-cv,dim);
	Matrix<float> *trainy = gpu->OPS->zeros(samples-cv,1);

	Matrix<float> *cvX = gpu->OPS->zeros(cv,dim);
	Matrix<float> *cvy = gpu->OPS->zeros(cv,1);
	gpu->OPS->slice(X,trainX,0,samples-cv,0,dim);
	gpu->OPS->slice(y,trainy,0,samples-cv,0,1);

	gpu->OPS->mul(trainX,trainX,1.0f/255.0f);

	gpu->OPS->slice(X,cvX,samples-cv,samples,0,dim);
	gpu->OPS->slice(y,cvy,samples-cv,samples,0,1);
	gpu->OPS->mul(cvX,cvX,1.0f/255.0f);

	//gpu->OPS->to_host(trainX,trainX->data);
	//gpu->OPS->to_host(cvX,cvX->data);

	cout << gpu->OPS->max(cvX) << endl;

	BatchAllocator *b_train = new CPUtoCPUBatchAllocator(gpu, trainX->data, trainy->data, trainX->rows, trainX->cols,trainy->cols,128);
	BatchAllocator *b_cv = new CPUtoCPUBatchAllocator(gpu, cvX->data, cvy->data, cvX->rows, cvX->cols,cvy->cols,128);

	Network net = Network(gpu);

	net._conf->LEARNING_RATE = 0.003f;
	net._conf->RMSPROP_MOMENTUM = 0.99f;

	net.add(new FCLayer(784,Input));
	net.add(new FCLayer(1200,Exponential_linear));
	net.add(new FCLayer(1200,Exponential_linear));
	net.add(new FCLayer(10,Softmax));

	//for(int i = 0; i < net._layers.size(); i++)
		//net._layers[i]->_transformer.clear();

	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.2f;

	net._opt = new Optimizer(gpu, RMSProp);
	net.init_weights(UniformSqrt);

	//t.tick();
	double t0 = omp_get_wtime();
	cout << "pre train" << endl;
	net.train(b_train, b_cv, 2);

	net._conf->DROPOUT = 0.25f;
	net._conf->LEARNING_RATE *= 0.2f;
	net._conf->LEARNING_RATE_DECAY = 0.95f;
	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.1f;

	cout << "time: " << omp_get_wtime()-t0 << endl;

	//t.tock();
}



void test_MPI(int argc, char *argv[]){

	ClusterNet *gpu = new ClusterNetCPU();
	Matrix<float> *A = gpu->rand(100,100);
	Matrix<float> *B = gpu->rand(2,2);

	cout << "pre gradient init" << endl;
	GradientAccumulator *ga = new GradientAccumulator(gpu);
	cout << "pre init mpi" << endl;
	ga->init_MPI();

	float a[4] = {0,0,0,0};
	if(ga->my_rank == 0)
	{
		for(int i = 0; i < 4; i++)
			a[i] = (float)i;
	}
	else
	{
		for(int i = 0; i < 4; i++)
			a[i] = (float)i*2;
	}

	cout << "pre gpu" << endl;
	//gpu->OPS->to_gpu(a, B);


	Matrix<float> *M = new Matrix<float>();
	M->rows = 2;
	M->cols = 2;
	M->size = 4;
	M->bytes = 4*4;
	M->data = a;
        M->isRowMajor = true;

	cout << "pre init matrix" << endl;
	ga->init_Matrix(M);
	cout << "pre send matrix" << endl;
	ga->send_MPI();
	cout << "pre recv matrix" << endl;
	ga->recv_MPI();
	cout <<"testingg"<< endl;
	//gpu->OPS->to_host(ga->buffer,a);

	MPI_Barrier(MPI_COMM_WORLD);
	if(ga->my_rank == 0)
	{
		cout << "Myrank " << ga->my_rank << endl;
		for(int i = 0; i < 4; i++)
	cout <<		ga->matrix->data[i] << " ";
	cout << endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(ga->my_rank == 1)
	{
		cout << "Myrank " << ga->my_rank << endl;
		for(int i = 0; i < 4; i++)
	cout <<		ga->matrix->data[i] << " ";
	cout << endl;
	}


}



void test_gem()
{
	ClusterNet *acc = new ClusterNetCPU();

	int size = 1200;

	Matrix<float> *a = acc->rand(size,size);
	Matrix<float> *b = acc->rand(size,size);
	acc->OPS->to_host(a,a->data);
	acc->OPS->to_host(b,b->data);

	Matrix<float> *A = acc->OPS->zeros(size,size);
	Matrix<float> *B = acc->OPS->zeros(size,size);
	Matrix<float> *C = acc->OPS->zeros(size,size);

	acc->OPS->to_gpu(a->data, A);
	acc->OPS->to_gpu(b->data, B);

	cout << C->data[0] << endl;

	acc->OPS->to_host(C,C->data);
	cout << C->data[0] << endl;

	//warm up
	acc->dot(A,B,C);
	double t0 = omp_get_wtime();
	for(int i = 0; i < 100; i++)
		acc->dot(A,B,C);


	cout << "time: " << omp_get_wtime()-t0 << endl;
	cout << "GFLOP: " << ((double)size*size*size*100) /((omp_get_wtime() - t0) *10e9)<< endl;
	for(int i = 0; i < 4; i++)
		cout << a->data[i] << " ";
	cout << endl;
	for(int i = 0; i < 4; i++)
		cout << b->data[i] << " ";
	cout << endl;

	float *aa = a->data;
	float *bb = b->data;
	cout << aa[0]*bb[0] + (aa[1]*bb[2]) << endl;

//	acc->OPS->printmat(A);
//	acc->OPS->printmat(B);

	acc->OPS->to_host(C,aa);

	cout << aa[0] << endl;
	cout << C->data[0] << endl;
}

void test_rdm()
{
	ClusterNet *acc = new ClusterNetCPU();

	int size = 4;

	Matrix<float> *a = acc->rand(size,size);
	acc->OPS->to_host(a,a->data);

	for(int i = 0; i < size; i++)
		cout << a->data[i] << " ";
	cout << endl;

	Matrix<float> *b = acc->randn(size,size);
	acc->OPS->to_host(b,b->data);

	for(int i = 0; i < size; i++)
		cout << b->data[i] << " ";
	cout << endl;

	Matrix<float> *c = acc->normal(size,size,10.0f,0.2f);
	acc->OPS->to_host(c,c->data);

	for(int i = 0; i < size; i++)
		cout << c->data[i] << " ";
	cout << endl;
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
	cout << "different size: " << omp_get_wtime()-t0 << endl;
} 

void filereader_test()
{
	ClusterNet *acc = new ClusterNetCPU();
	freader *r = new freader("/data/NLP/out.txt", acc);

	r->printMap();
	std::string ret = r->read_chunk("/data/NLP/out.txt", 0, 1000);
	//r->getMatrix(128, 3);
	cout << ret << endl;


}


int test_MPI_simple(int argc, char *argv[]) 
{
	ClusterNet *gpu = new ClusterNetCPU();
	  MPI_Init(&argc , &argv);
	  
	  int my_rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	  int size;
	  MPI_Comm_size(MPI_COMM_WORLD, &size);

	int print_size = 4;

	if(my_rank == 0) gpu->setRandomState(234234);


	Matrix<float> *A = gpu->rand(2,2);
	Matrix<float> *recv = gpu->OPS->zeros(2,2);

	std::vector<Matrix<float>*> b;
	std::vector<Matrix<float>*> v;

	v.push_back(gpu->OPS->get_view(A,0, 1));
	b.push_back(gpu->OPS->get_view(recv,0, 1));
	v.push_back(gpu->OPS->get_view(A,1, 2));
	b.push_back(gpu->OPS->get_view(recv,1, 2));

	gpu->OPS->to_host(A, A->data);
	gpu->OPS->to_host(recv, recv->data);

	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank ==0)
	{
		for(int i = 0; i < print_size; i++)
			cout << A->data[i] << " ";
		cout << endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank ==1)
	{
		for(int i = 0; i < print_size; i++)
			cout << A->data[i] << " ";
		cout << endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	/*
	if(my_rank == 0)
	{
	  MPI_Send(v[1]->data,v[1]->size, MPI_FLOAT,1,999,MPI_COMM_WORLD);
	}
	else
	{
	  MPI_Recv(b[1]->data, b[1]->size, MPI_FLOAT,0,999,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

*/


for(int i =0; i < size; i++)
	MPI_Scatter(A->data, A->size/size, MPI_FLOAT, b[i]->data, recv->size/size, MPI_FLOAT, i, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank ==0)
	{
		for(int i = 0; i < print_size; i++)
			cout << recv->data[i] << " ";
		cout << endl;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank ==1)
	{
		for(int i = 0; i < print_size; i++)
			cout << recv->data[i] << " ";
		cout << endl;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);


}

void test_MPI_MIC(int argc, char *argv[]){

	ClusterNet *gpu = new ClusterNetCPU();

	GradientAccumulator *ga = new GradientAccumulator(gpu);
	ga->init_MPI();

	if(ga->my_rank ==0)
		gpu->setRandomState(23523);

	Matrix<float> *B = gpu->rand(2,2);
	
	gpu->OPS->to_host(B,B->data);

	MPI_Barrier(MPI_COMM_WORLD);
	if(ga->my_rank==0)
		for(int i = 0; i < 4; i++)
	cout <<		B->data[i] << " ";
	cout << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	if(ga->my_rank==1)
		for(int i = 0; i < 4; i++)
	cout <<		B->data[i] << " ";
	cout << endl;
	MPI_Barrier(MPI_COMM_WORLD);


	ga->init_Matrix(B);
	ga->send_MPI();
	ga->recv_MPI();

	gpu->OPS->to_host(B,B->data);

	MPI_Barrier(MPI_COMM_WORLD);
	if(ga->my_rank == 0)
	{
		for(int i = 0; i < 4; i++)
	//cout <<		ga->matrix->data[i] << " ";
	cout <<		B->data[i] << " ";
	cout << endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(ga->my_rank == 1)
	{
		cout << "Myrank " << ga->my_rank << endl;
		for(int i = 0; i < 4; i++)
	//cout <<		ga->matrix->data[i] << " ";
	cout <<		B->data[i] << " ";
	cout << endl;
	}


}
int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);
	printf("abc2\n");
	cout << "a" << endl;
	//test_LSTM_swapping();
	//deeplearningdb_test();
	test_phi();
	//test_rdm();
	//filereader_test();
	//test_MPI_simple(argc,argv);
	test_nonvectorized();
	//test_gem();
	//test_MPI(argc, argv);
	//test_MPI_MIC(argc, argv);
	//test_lookup_time();
	test_neural_network();
	//test_neural_network_MPI();
	MPI_Finalize();


	return 0;
}
