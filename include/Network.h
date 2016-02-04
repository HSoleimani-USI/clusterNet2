#ifndef Network_H
#define Network_H

#include <ClusterNet.h>
#include <basicOps.cuh>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

class Layer;

class Network
{
public:
	Network();
	~Network(){};
	bool _isTrainTime;


protected:
	vector<Layer*> _layers;






};

#endif
