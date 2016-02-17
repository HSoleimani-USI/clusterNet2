#include "ClusterNet.h"
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
	ClusterNet gpu = ClusterNet();
	Matrix<float> *A = gpu.rand(100,100);
	Matrix<float> *B = gpu.rand(100,100);
	Matrix<float> *C = gpu.rand(100,100);

	Timer t = Timer();

	t.tick();
	for(int i = 0; i < 10000; i++)
		gpu.dot(A,B,C);
	t.tock();
}

void test_transfer_time()
{
	ClusterNet gpu = ClusterNet();
	int batch_size = 32;
	int time_steps = 100;

	Matrix<float> *data = gpu.rand(128,batch_size*time_steps);
	Matrix<float> *host = data->to_host();
	Matrix<float> *pinned = to_pinned(128,batch_size*time_steps,data->data);


	Matrix<float> *error = gpu.rand(batch_size*time_steps,256);
	Matrix<float> *w = gpu.rand(128,256);

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
		gpu.dot(data,error,w);
	t.tock();

}

void deeplearningdb_test()
{
	DeepLearningDB *db = new DeepLearningDB();

	Table *tbl = db->get_table("Test");

	std::map<std::string, int> dict = tbl->get_dictionary<std::string,int>("dict");

	cout << dict["uden"] << endl;


}

void test_neural_network()
{

	test_transfer_time();

	Timer t = Timer();
	ClusterNet *gpu = new ClusterNet();

	gpu->useNervanaGPU = true;

	Matrix<float> *X = read_hdf5("/home/tim/data/mnist/distributed_X.hdf5");
	Matrix<float> *y = read_hdf5("/home/tim/data/mnist/distributed_y.hdf5");

	Matrix<float> *trainX = zeros<float>(60000,784);
	Matrix<float> *trainy = zeros<float>(60000,1);
	Matrix<float> *test_slice = zeros<float>(128,784);

	Matrix<float> *cvX = zeros<float>(10000,784);
	Matrix<float> *cvy = zeros<float>(10000,1);

	slice(X,trainX,0,60000,0,784);
	slice(y,trainy,0,60000,0,1);


	slice(X,cvX,60000,70000,0,784);
	slice(y,cvy,60000,70000,0,1);



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

	deeplearningdb_test();
//	test_neural_network();


	return 0;
}
