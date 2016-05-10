#include "ClusterNet.h"
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

using namespace std;


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

void test_neural_network()
{

	test_transfer_time();

	//Timer t = Timer();
	ClusterNet *gpu = new ClusterNetCPU();



	gpu->useNervanaGPU = true;


	Matrix<float> *X = gpu->OPS->read_csv("/home/tim/data/mnist/X.csv");
	Matrix<float> *y = gpu->OPS->read_csv("/home/tim/data/mnist/y.csv");
	//Matrix<float> *X = read_hdf5<float>("/home/tim/data/astro/X.hdf5");
	//Matrix<float> *y = read_hdf5<float>("/home/tim/data/astro/y.hdf5");

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


	gpu->OPS->slice(X,cvX,samples-cv,samples,0,dim);
	gpu->OPS->slice(y,cvy,samples-cv,samples,0,1);



	BatchAllocator *b_train = new CPUtoCPUBatchAllocator(gpu, trainX->data, trainy->data, trainX->rows, trainX->cols,trainy->cols,128);
	BatchAllocator *b_cv = new CPUtoCPUBatchAllocator(gpu, cvX->data, cvy->data, cvX->rows, cvX->cols,cvy->cols,100);

	Network net = Network(gpu);

	net._conf->LEARNING_RATE = 0.003f;
	net._conf->RMSPROP_MOMENTUM = 0.99f;

	net.add(new FCLayer(784,Input));
	net.add(new FCLayer(1024,Exponential_linear));
	net.add(new FCLayer(1024,Exponential_linear));
	net.add(new FCLayer(10,Softmax));

	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.2f;

	net._opt = new Optimizer(gpu, RMSProp);
	net.init_weights(UniformSqrt);


	/*
	net._opt->_updatetype = RMSPropInit;
	net._conf->RMSPROP_MOMENTUM = 0.0f;
	net.fit_partial(b_train,10);
	net._conf->RMSPROP_MOMENTUM = 0.99f;
	net._opt->_updatetype = RMSProp;
	*/


	//t.tick();
	net.train(b_train, b_cv, 20);

	/*
	b_train->BATCH_SIZE = 256;
	b_train->BATCHES *= 0.5;
	b_train->CURRENT_BATCH = 0;
	*/

	net._conf->DROPOUT = 0.25f;
	net._conf->LEARNING_RATE *= 0.2f;
	net._conf->LEARNING_RATE_DECAY = 0.95f;
	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.1f;

	net.train(b_train, b_cv, 11);

	//t.tock();

}

/*
void test_MPI(int argc, char const *argv[]){

	ClusterNet *gpu = new ClusterNetCPU();
	Matrix<float> *A = gpu->rand(100,100);
	Matrix<float> *B = gpu->rand(2,2);


	gradientAccumulator *ga = new gradientAccumulator();
	ga.init_MPI(argc, argv);

	float a[4] = {0,0,0,0};
	if(ga->my_rank == 0)
	{
		for(int i = 0; i < 4; i++)
			a[i] = 1.7;
	}
	else
	{
		for(int i = 0; i < 4; i++)
			a[i] = 1.2;
	}

	to_gpu(B,a);




	ga.init_Matrix(B);
	ga.send_MPI();
	ga.recv_MPI();

	if(ga->my_rank == 0)
	{
		cout << "Myrank " << ga->my_rank << endl;
		for(int = 0; i < 4; i++)
			ga->buffer->data[i];
	}


}
*/





int main(int argc, char const *argv[]) {

	//test_LSTM_swapping();
	//deeplearningdb_test();
	test_neural_network();
	//test_lookup_time();


	return 0;
}
