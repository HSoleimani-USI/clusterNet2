#ifndef Network_H
#define Network_H

#include <ClusterNet.h>
#include <basicOps.cuh>
#include <vector>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::vector;

class Layer;
class Optimizer;
class Configurator;
class ErrorHandler;
class ClusterNet;
class BatchAllocator;

class Network
{
public:
	Network(ClusterNet *gpu);
	~Network(){};
	bool _isTrainTime;
	void add(Layer *layer);

	void init_weights(WeightInitType_t wtype);
	void copy_global_params_to_layers();

	void fit_partial(BatchAllocator *b, int batches);
	void fit(BatchAllocator *b, int epochs);
	void train(BatchAllocator *train, BatchAllocator *CV, int epochs);
	Matrix<float> *predict(Matrix<float> *X);

	void get_errors(BatchAllocator *b, std::string message);


	Optimizer *_opt;
	Configurator *_conf;
	ErrorHandler *_errorhandler;
	vector<Layer*> _layers;

protected:
	void init_activations(int batchsize);
	ClusterNet *_gpu;






};

#endif
