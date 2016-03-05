/*
 * RecurrentNetwork.h
 *
 *  Created on: Mar 5, 2016
 *      Author: tim
 */

#include <ClusterNet.h>
#include <vector>

using std::vector;

#ifndef RECURRENTNETWORK_H_
#define RECURRENTNETWORK_H_

class LSTMLayer;
class BatchAllocator;
class Optimizer;
class ErrorHandler;
class Configurator;

class RecurrentNetwork
{
public:
	RecurrentNetwork(ClusterNet *gpu);
	~RecurrentNetwork(){};
	bool _isTrainTime;
	void add(LSTMLayer *layer);

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
	vector<LSTMLayer*> _layers;

protected:
	void init_activations(int batchsize);
	ClusterNet *_gpu;

};

#endif /* RECURRENTNETWORK_H_ */
