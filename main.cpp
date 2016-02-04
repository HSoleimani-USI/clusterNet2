#include "ClusterNet.h"
#include "Timer.cuh"
#include "NeuralNetwork.h"

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

	std::vector<int> FCLayers = std::vector<int>();
	FCLayers.push_back(1024);
	FCLayers.push_back(1024);
	NeuralNetwork net = NeuralNetwork(gpu, b_train, b_cv, FCLayers, Rectified_Linear, 10);

	net.fit();
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
