#include "clusterNet2.h"
#include "Timer.cuh"

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

int main(int argc, char const *argv[]) {


	Matrix<float> *A = empty<float>(10, 10);

	Matrix<float> *C = fill_matrix<float>(10, 10, 17);

	Matrix<float> *B = C->to_host();

	Timer *t = new Timer();


	test_timer();



	return 0;
}
