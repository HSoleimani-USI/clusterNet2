#include "clusterNet2.h"

using namespace std;

int main(int argc, char const *argv[])
{
	ClusterNet2<float>* gpu = new ClusterNet2<float>();


    Matrix<float> *A = gpu->rand(10,10);
    Matrix<float> *B = gpu->rand(10,10);
    Matrix<float> *C = empty<float>(10,10);

    gpu->dot(A,B,C);

    float *cpu = (float*)malloc(C->bytes);
    to_host<float>(C, cpu);

    for(int i = 0; i < 10; i++)
      std::cout << cpu[i] << std::endl;


    return 0;
}
