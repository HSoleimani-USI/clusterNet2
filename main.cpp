#include "clusterNet2.h"

using namespace std;

int main(int argc, char const *argv[])
{
	ClusterNet2 * fooClass = new ClusterNet2();

    fooClass->runThreads();

    Matrix<int> *A = empty<int>(10,10);




    Matrix<int> *C =  fill_matrix<int>(10,10,17);

    
    Matrix<int> *B = C->to_host();

    for(int i =0; i < 100; i++)
    {
    	std::cout << B->data[i] << std::endl;
    }
    
    



    return 0;
}
