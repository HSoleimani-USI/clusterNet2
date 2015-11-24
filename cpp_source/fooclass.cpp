#include "fooclass.h"

char* FooClass::SayHello() 
{
    return "Hello Im a Linux Shared Library";
}



FooClass::FooClass(){}

void workerFunc()
{
 	boost::posix_time::seconds workTime(3);

    std::cout << "Worker: running" << std::endl;

    // Pretend to do something useful...
    boost::this_thread::sleep(workTime);


    Matrix<int> *A = empty<int>(10,10);


    Matrix<int> *C =  fill_matrix<int>(10,10,13);


    Matrix<int> *B = C->to_host();

    for(int i =0; i < 100; i++)
    {
    	std::cout << B->data[i] << std::endl;
    }
    



    std::cout << "Worker: finished" << std::endl;
}
void FooClass::runThreads()
{
	std::cout << "main: startup" << std::endl;

    boost::thread workerThread(workerFunc);

    std::cout << "main: waiting for thread" << std::endl;

    workerThread.join();

    std::cout << "main: done" << std::endl;
}