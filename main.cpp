#include "clusterNet2.h"
#include "Timer.cuh"

using namespace std;

int main(int argc, char const *argv[]) {
	ClusterNet2<float>* fooClass = new ClusterNet2<float>();

	Matrix<float> *A = empty<float>(10, 10);

	Matrix<float> *C = fill_matrix<float>(10, 10, 17);

	Matrix<float> *B = C->to_host();

	Timer *t = new Timer();


	t->tick();
	elementWise<kadd>(A,A,A,0.0f);
	t->tock();


	return 0;
}
