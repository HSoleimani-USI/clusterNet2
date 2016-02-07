#include "ClusterNet.h"
#include "Timer.cuh"
#include "Network.h"
#include <Optimizer.h>
#include <FCLayer.h>
#include <BatchAllocator.h>
#include <ErrorHandler.h>
#include <Configurator.h>

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

void test_neural_network()
{

	ClusterNet *gpu = new ClusterNet();

	Matrix<float> *X = read_hdf5("/home/tim/data/mnist/distributed_X.hdf5");
	Matrix<float> *y = read_hdf5("/home/tim/data/mnist/distributed_y.hdf5");

	Matrix<float> *trainX = zeros<float>(60000,784);
	Matrix<float> *trainy = zeros<float>(60000,1);

	Matrix<float> *cvX = zeros<float>(10000,784);
	Matrix<float> *cvy = zeros<float>(10000,1);

	slice(X,trainX,0,60000,0,784);
	slice(y,trainy,0,60000,0,1);


	slice(X,cvX,60000,70000,0,784);
	slice(y,cvy,60000,70000,0,1);


	BatchAllocator *b_train = new BatchAllocator(trainX->data, trainy->data, trainX->rows, trainX->cols,trainy->cols,128);
	BatchAllocator *b_cv= new BatchAllocator(cvX->data, cvy->data, cvX->rows, cvX->cols,cvy->cols,128);

	Network net = Network(gpu);

	net._conf->LEARNING_RATE = 0.003f;
	net._conf->RMSPROP_MOMENTUM = 0.9f;

	net.add(new FCLayer(784,Input));
	net.add(new FCLayer(1024,Rectified_Linear));
	net.add(new FCLayer(1024,Rectified_Linear));
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

	Timer t = Timer();

	t.tick();
	net.train(b_train, b_cv, 100);
	t.tock();

	//net.fit(b_train,200);
}

int main(int argc, char const *argv[]) {

	test_neural_network();

	Matrix<float> *A = empty<float>(10, 10);

	Matrix<float> *C = fill_matrix<float>(10, 10, 17);

	Matrix<float> *B = C->to_host();

	Timer *t = new Timer();


	test_timer();



	return 0;
}
