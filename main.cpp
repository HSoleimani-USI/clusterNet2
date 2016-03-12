#include "ClusterNet.h"
#include <ClusterNetGPU.h>
#include "Timer.cuh"
#include "Network.h"
#include <Optimizer.h>
#include <FCLayer.h>
#include <BatchAllocator.h>
#include <GPUBatchAllocator.h>
#include <BufferedBatchAllocator.h>
#include <ErrorHandler.h>
#include <Configurator.h>
#include <DeepLearningDB.h>
#include <Table.h>
#include <string>
#include <map>

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

void test_transfer_time()
{
	ClusterNet *gpu = new ClusterNetGPU();
	int batch_size = 32;
	int time_steps = 100;

	Matrix<float> *data = gpu->rand(128,batch_size*time_steps);
	Matrix<float> *host = to_host(data);
	Matrix<float> *pinned = to_pinned(128,batch_size*time_steps,data->data);


	Matrix<float> *error = gpu->rand(batch_size*time_steps,256);
	Matrix<float> *w = gpu->rand(128,256);

	Timer t = Timer();

	t.tick();
	for(int i = 0; i < 100; i++)
		to_gpu(host->data,data);
	t.tock();

	t.tick();
	for(int i = 0; i < 100; i++)
		to_gpu(pinned->data,data);
	t.tock();


	t.tick();
	for(int i = 0; i < 100; i++)
		gpu->dot(data,error,w);
	t.tock();

}

void deeplearningdb_test()
{
	DeepLearningDB *db = new DeepLearningDB();

	Table *tbl = db->get_table("vocab");

	std::map<std::string, int> dict = tbl->get_dictionary<std::string,int>("brown/vocab2idx");

	cout << dict["car"] << endl;


	Matrix<int> *arr = tbl->get_data<int>("brown/idx");
	cout << "post" << endl;
	for(int i = 0; i < 100; i++)
		cout << arr->data[i] << endl;


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

	Matrix<float> *X = read_hdf5<float>("/home/tim/data/mnist/distributed_X.hdf5");
	Matrix<float> *y = read_hdf5<float>("/home/tim/data/mnist/distributed_y.hdf5");


	//Matrix<float> *X = read_hdf5<float>("/home/tim/data/astro/X.hdf5");
	//Matrix<float> *y = read_hdf5<float>("/home/tim/data/astro/y.hdf5");

	int samples = X->rows;
	int cv = 10000;
	int dim = X->cols;
	int classes = 10;

	Matrix<float> *trainX = zeros<float>(samples-cv,dim);
	Matrix<float> *trainy = zeros<float>(samples-cv,1);

	Matrix<float> *cvX = zeros<float>(cv,dim);
	Matrix<float> *cvy = zeros<float>(cv,1);

	slice(X,trainX,0,samples-cv,0,dim);
	slice(y,trainy,0,samples-cv,0,1);


	slice(X,cvX,samples-cv,samples,0,dim);
	slice(y,cvy,samples-cv,samples,0,1);



	BatchAllocator *b_train = new GPUBatchAllocator(trainX->data, trainy->data, trainX->rows, trainX->cols,trainy->cols,128);
	BatchAllocator *b_cv = new GPUBatchAllocator(cvX->data, cvy->data, cvX->rows, cvX->cols,cvy->cols,100);

	Network net = Network(gpu);

	net._conf->LEARNING_RATE = 0.003f;
	net._conf->RMSPROP_MOMENTUM = 0.99f;

	net.add(new FCLayer(784,Input));
	net.add(new FCLayer(1024,Exponential_linear));
	net.add(new FCLayer(1024,Exponential_linear));
	net.add(new FCLayer(10,Softmax));

	net.copy_global_params_to_layers();
	net._layers.front()->_conf->DROPOUT = 0.2f;

	net._opt = new Optimizer(RMSProp);
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

int main(int argc, char const *argv[]) {

	//test_LSTM_swapping();
	//deeplearningdb_test();
	test_neural_network();
	//test_lookup_time();


	return 0;
}
