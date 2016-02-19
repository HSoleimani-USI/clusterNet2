#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <ClusterNet.h>

#ifndef Layer_H
#define Layer_H

class Network;
class ActivationFunction;
class Transformer;
class Configurator;

class Layer
{
public:
	Layer *next;
	Layer *prev;

	std::string _LayerType;

	Matrix<float> *b_grad_next;
	Matrix<float>* w_grad_next;
	Matrix<float> *w_next;
	Matrix<float> *b_next;

	Matrix<float> *w_rms_next;
	Matrix<float> *b_rms_next;

	Matrix<float> *error;
	Matrix<float> *activation;
	Matrix<float> *activation_grad;

	Matrix<float> *target;
	Matrix<float> *target_matrix;

	Matrix<float> *result;
	Matrix<float> *eq;

	ClusterNet *GPU;


	int UNITCOUNT;
	int BATCH_SIZE;
	int Layer_ID;
	Network *_network;

	Layer(){};
	~Layer(){};

	virtual void forward() = 0;
	virtual void backward_errors() = 0;
	virtual void backward_grads() = 0;

	void init(int unitcount, Unittype_t unitType);
	Matrix<float> *get_forward_activation();
	void apply_transformations();
	void init_transformers(ClusterNet *gpu, Network *net);
	void init_transformer_activations(int batch_size);

	ActivationFunction *_func;

	std::vector< Transformer*> _transformer;

	Configurator *_conf;




protected:



};

#endif
