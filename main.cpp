#include "ClusterNet.h"
#include <ClusterNetGPU.h>
#include <ClusterNetCPU.h>
#include "Timer.cuh"
#include "Network.h"
#include <Optimizer.h>
#include <FCLayer.h>
#include <BatchAllocator.h>
#include <CPUBatchAllocator.h>
#include <GPUBatchAllocator.h>
#include <BufferedBatchAllocator.h>
#include <ErrorHandler.h>
#include <Configurator.h>
#include <string>
#include <map>
#include <BasicOpsWrapperCPU.h>

using namespace std;


void test_timer()
{
	ClusterNet *gpu = new ClusterNetGPU();
	Matrix<float> *A = gpu->rand(100,100);
	Matrix<float> *B = gpu->rand(100,100);
	Matrix<float> *C = gpu->rand(100,100);

	Timer t = Timer();

	t.tick();
	for(int i = 0; i < 10000; i++)
		gpu->dot(A,B,C);
	t.tock();
}

void get_csv_mnist_test()
{

	ClusterNet *gpu = new ClusterNetGPU();
	Matrix<float> *Xy = gpu->OPS->empty(70000,785);
	Matrix<float> *train = gpu->OPS->read_csv("/home/tim/data/mnist/mnist_train.csv");
	Matrix<float> *test = gpu->OPS->read_csv("/home/tim/data/mnist/mnist_test.csv");
	cudaMemcpy(&(Xy->data[0]), train->data, train->size, cudaMemcpyDefault);
	cudaMemcpy(&(Xy->data[60000*785]), test->data, test->size, cudaMemcpyDefault);

	Matrix<float> *y = gpu->OPS->empty(70000,1);
	gpu->OPS->slice(Xy, y, 0,70000,0,1);
	gpu->OPS->printmat(y,59900, 60010,0,1);



}

void test_transfer_time()
{
	ClusterNet *gpu = new ClusterNetGPU();
	int batch_size = 32;
	int time_steps = 100;

	Matrix<float> *data = gpu->rand(128,batch_size*time_steps);
	Matrix<float> *host = gpu->OPS->to_host(data);
	Matrix<float> *pinned = gpu->OPS->to_pinned(128,batch_size*time_steps,data->data);


	Matrix<float> *error = gpu->rand(batch_size*time_steps,256);
	Matrix<float> *w = gpu->rand(128,256);

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

}

void test_LSTM_swapping()
{

	ClusterNet *gpu = new ClusterNetGPU();

	int hidden = 256;
	int timesteps = 1000;
	int batch_size = 32;
	int stack_size = 4;

	Matrix<float> *R = gpu->rand(hidden*stack_size,hidden);
	Matrix<float> *inputR = gpu->rand(batch_size*stack_size,hidden);
	Matrix<float> *errorsR = gpu->rand(batch_size*stack_size,hidden);

	Timer t = Timer();


	for(int i = 0; i < 1000; i++)
		gpu->dot(inputR,errorsR,R);

	t.tick();

	for(int i = 0; i < 10000; i++)
		gpu->dot(inputR,errorsR,R);

	t.tock();
}

void test_LSTM_optimizations()
{

	ClusterNet *gpu = new ClusterNetGPU();

	int hidden = 1024;
	int timesteps = 1000;
	int input_size = 256;
	int batch_size = 32;
	int stack_size = 4;

	Matrix<float> *W = gpu->rand(input_size,hidden);
	Matrix<float> *W_T = gpu->rand(hidden*stack_size,input_size);

	Matrix<float> *inputs = gpu->rand(batch_size*timesteps,input_size);
	Matrix<float> *inputsT = gpu->rand(input_size,batch_size*timesteps);

	Matrix<float> *outputs = gpu->rand(batch_size*timesteps,hidden);
	Matrix<float> *outputsT = gpu->rand(hidden*stack_size,batch_size*timesteps);

	Matrix<float> *batch = gpu->rand(batch_size,hidden);

	Matrix<float> *errorsR = gpu->rand(batch_size*timesteps,hidden);
	Matrix<float> *errorsRT = gpu->rand(hidden*stack_size,batch_size*timesteps);

	Timer t = Timer();


	for(int i = 0; i < 100; i++)
		gpu->dot(inputs,W,outputs);

	t.tick("transposed");
	for(int i = 0; i < 100; i++)
	{
		gpu->dot(W_T,inputsT,outputsT);
		gpu->OPS->transpose(outputsT, outputs,outputsT->rows, outputsT->cols);
		//for(int r = 0; r < 4; r++)
			//for(int j = 0; j < timesteps; j++)
				//gpu->OPS->slice(outputsT, batch, hidden*r, hidden*(r+1), j*batch_size, (j+1)*batch_size);
	}
	t.tock("transposed");

	t.tick("normal");
	for(int i = 0; i < 400; i++)
		gpu->dot(inputs,W,outputs);
	t.tock("normal");


}

void test_lookup_time()
{

	ClusterNet *gpu = new ClusterNetGPU();

	int embedding_rows = 10000;
	int embedding_cols = 256;
	int batch_size = 128;
	int batch_cols = 1024;

	Matrix<float> *embedding = gpu->rand(embedding_rows,embedding_cols);
	Matrix<float> *batch = gpu->rand(batch_size*batch_cols,embedding_cols);
	Matrix<float> *batch1 = gpu->rand(batch_size*batch_cols,embedding_cols);

	Timer t = Timer();
	t.tick();
	for(int i = 0; i < 1000; i++)
		cudaMemcpy(batch1->data, batch->data,batch->bytes,cudaMemcpyDeviceToDevice);
	t.tock();

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

	Timer t = Timer();
	ClusterNet *gpu = new ClusterNetGPU();



	gpu->useNervanaGPU = true;

	//Matrix<float> *X = gpu->OPS->read_hdf5("/home/tim/data/mnist/distributed_X.hdf5");
	//Matrix<float> *y = gpu->OPS->read_hdf5("/home/tim/data/mnist/distributed_y.hdf5");


	Matrix<float> *XCPU = gpu->OPS->read_csv("/home/tim/data/mnist/X.csv");
	Matrix<float> *yCPU = gpu->OPS->read_csv("/home/tim/data/mnist/y.csv");
	Matrix<float> *X = gpu->OPS->empty(70000,784);
	Matrix<float> *y = gpu->OPS->empty(70000,1);


	gpu->OPS->to_gpu(XCPU->data,X);
	gpu->OPS->to_gpu(yCPU->data,y);


	/*

	gpu->OPS->mul(X,X,1.0f/255.0f);
	gpu->OPS->printmat(y,59990,60010,0,1);

	cout << gpu->OPS->max(X) << endl;
	gpu->OPS->printdim(X);
	gpu->OPS->printdim(y);
	*/

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

	//gpu->OPS->mul(trainX, trainX, 1.0f/255.0f);
	//gpu->OPS->mul(cvX, cvX, 1.0f/255.0f);



	BatchAllocator *b_train = new GPUBatchAllocator(gpu, trainX->data, trainy->data, trainX->rows, trainX->cols,trainy->cols,128);
	BatchAllocator *b_cv = new GPUBatchAllocator(gpu, cvX->data, cvy->data, cvX->rows, cvX->cols,cvy->cols,100);

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


	t.tick();
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

	t.tock();

}

void run_astro()
{

	test_transfer_time();

	Timer t = Timer();
	ClusterNet *gpu = new ClusterNetGPU();



	gpu->useNervanaGPU = true;

	//Matrix<float> *X = gpu->OPS->read_hdf5("/home/tim/data/astro/X.hdf5");
	//Matrix<float> *y = gpu->OPS->read_hdf5("/home/tim/data/astro/y.hdf5");

	Matrix<float> *X = gpu->OPS->read_hdf5("/home/tim/data/astro/X_processed.hdf5");
	Matrix<float> *y = gpu->OPS->read_hdf5("/home/tim/data/astro/y_processed.hdf5");


	int samples = X->rows;
	int cv = 80000;
	int dim = X->cols;
	int classes = 2;

	Matrix<float> *trainX = gpu->OPS->get_view(X,0,samples-cv);
	Matrix<float> *trainy = gpu->OPS->get_view(y,0,samples-cv);

	Matrix<float> *cvX = gpu->OPS->get_view(X,samples-cv,samples);
	Matrix<float> *cvy = gpu->OPS->get_view(X,samples-cv,samples);




	BatchAllocator *b_train = new CPUBatchAllocator(gpu, trainX->data, trainy->data, trainX->rows, trainX->cols,trainy->cols,128);
	BatchAllocator *b_cv = new CPUBatchAllocator(gpu, cvX->data, cvy->data, cvX->rows, cvX->cols,cvy->cols,128);

	Network net = Network(gpu);

	net._conf->LEARNING_RATE = 0.00001f;
	net._conf->RMSPROP_MOMENTUM = 0.90f;

	net.add(new FCLayer(dim,Input));
	net.add(new FCLayer(256,Exponential_linear));
	net.add(new FCLayer(256,Exponential_linear));
	net.add(new FCLayer(classes,Softmax));

	net._conf->DROPOUT = 0.0f;
	net._layers.front()->_conf->DROPOUT = 0.0f;
	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.2f;

	cout << net._layers.front()->_conf->DROPOUT << endl;

	net._opt = new Optimizer(gpu, RMSProp);
	net.init_weights(UniformSqrt);



	t.tick();
	net.train(b_train, b_cv, 500);

	t.tock();


}

int main(int argc, char const *argv[]) {

	//get_csv_mnist_test();
	//test_LSTM_optimizations();
	//test_LSTM_swapping();
	//deeplearningdb_test();
	//run_astro();
	test_neural_network();
	//test_lookup_time();


	return 0;
}
