#include "clusterNet2.h"
#include "Timer.cuh"
#include "NeuralNetwork.h"

using namespace std;


void test_timer()
{
	ClusterNet2<float> gpu = ClusterNet2<float>();
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

	ClusterNet2<float> *gpu = new ClusterNet2<float>();

	Matrix<float> *X_train = gpu->rand(1000,10);
	Matrix<float> *y_train = gpu->rand(1000,1);
	elementWiseUnary<ksgt>(y_train, y_train, 0.5f);

	Matrix<float> *X_cv = gpu->rand(200,10);
	Matrix<float> *y_cv = gpu->rand(200,1);
	elementWiseUnary<ksgt>(y_cv, y_cv, 0.5f);

	BatchAllocator *b_train = new BatchAllocator(X_train->to_host()->data, y_train->to_host()->data, X_train->rows, X_train->cols,y_train->cols,128);
	BatchAllocator *b_cv = new BatchAllocator(X_cv->to_host()->data, y_cv->to_host()->data, X_cv->rows, X_cv->cols,y_cv->cols,128);

	std::vector<int> layers = std::vector<int>();
	layers.push_back(1024);
	layers.push_back(1024);
	NeuralNetwork net = NeuralNetwork(gpu, b_train, b_cv, layers, Rectified_Linear, 2);

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
