#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <iostream>
#include <basicOps.cuh>

#ifndef __FOOCLASS_H__
#define __FOOCLASS_H__

class FooClass 
{
    public:
        char* SayHello(); 
  		FooClass();
  		void runThreads();
};

#endif //__FOOCLASS_H__