#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <ClusterNet.h>

#ifndef Layer_H
#define Layer_H

class Layer
{
public:
	Layer *next;
	Layer *prev;

	Matrix<float> *b_grad_next;
	Matrix<float>* w_grad_next;
	Matrix<float> *w_next;
	Matrix<float> *b_next;

	Matrix<float> *w_rms_next;
	Matrix<float> *b_rms_next;

	Matrix<float> *bias_activations;
	Matrix<float> *out;
	Matrix<float> *error;
	Matrix<float> *activation;

	Matrix<float> *target;
	Matrix<float> *target_matrix;

	Matrix<float> *result;
	Matrix<float> *eq;

	ClusterNet *GPU;

	float LEARNING_RATE;
	float RMSPROP_MOMENTUM;
	float RUNNING_ERROR;
	float RUNNING_SAMPLE_SIZE;
	Unittype_t UNIT_TYPE;
	Costfunction_t COST;
	float DROPOUT;
	int UNITCOUNT;
	int BATCH_SIZE;
	int Layer_ID;



	WeightUpdateType_t UPDATE_TYPE;
	virtual void forward() = 0;
	virtual void forward(bool useDropout) = 0;
	virtual void running_error() = 0;
	virtual void backward_errors() = 0;
	virtual void backward_grads() = 0;
	virtual void print_error(std::string message) = 0;
	virtual void weight_update() = 0;

	virtual void link_with_next_Layer(Layer *next_Layer) = 0;
	virtual void init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet *gpu) = 0;

	virtual void set_hidden_dropout(float dropout) = 0;

	virtual void dropout_decay() = 0;
	virtual void learning_rate_decay(float decay_rate) = 0;

	virtual Layer *get_root() = 0;

protected:
	virtual void unit_activation() = 0;
	virtual void unit_activation(bool useDropout) = 0;
	virtual void activation_gradient() = 0;
	virtual void apply_dropout() = 0;


};

#endif
