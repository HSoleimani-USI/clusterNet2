#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <iostream>
#include <basicOps.cuh>

#ifndef __CLUSTERNET2_H__
#define __CLUSTERNET2_H__

class ClusterNet2
{
    public:
        ClusterNet2();
  		void runThreads();
};

#endif //__CLUSTERNET2_H__
