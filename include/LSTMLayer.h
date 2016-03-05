/*
 * LSTMLayer.h
 *
 *  Created on: Feb 20, 2016
 *      Author: tim
 */


#include <ClusterNet.h>
#include <basicOps.cuh>
#include <vector>

using std::vector;

#ifndef LSTMLAYER_H_
#define LSTMLAYER_H_

class RecurrentNetwork;
class ActivationFunction;
class Configurator;
class Transformer;


class LSTMLayer
{
public:
	LSTMLayer(){};
	~LSTMLayer(){};


	LSTMLayer *input;
	LSTMLayer *next;
	LSTMLayer *prev;
	LSTMLayer *output;


	vector<Matrix<float> * > skip_weights;
	vector<Matrix<float> * > skip_weights_rms;
	vector<Matrix<float> * > skip_weights_grad;

	vector<Matrix<float> * > skip_weights_input_gate;
	vector<Matrix<float> * > skip_weights_input_gate_rms;
	vector<Matrix<float> * > skip_weights_input_gate_grad;

	vector<Matrix<float> * > skip_weights_forget_gate;
	vector<Matrix<float> * > skip_weights_forget_gate_rms;
	vector<Matrix<float> * > skip_weights_forget_gate_grad;

	vector<Matrix<float> * > skip_weights_output_gate;
	vector<Matrix<float> * > skip_weights_output_gate_rms;
	vector<Matrix<float> * > skip_weights_output_gate_grad;

	vector<Matrix<float> * > output_weights;
	vector<Matrix<float> * > output_weights_rms;
	vector<Matrix<float> * > output_weights_grad;

	vector<Matrix<float> * > output_biases;
	vector<Matrix<float> * > output_biases_rms;
	vector<Matrix<float> * > output_biases_grad;

	Matrix<float>* w_grad_next_input;
	Matrix<float> *w_next_input;
	Matrix<float> *w_rms_next_input;

	Matrix<float>* w_grad_next_input_gate;
	Matrix<float> *w_next_input_gate;
	Matrix<float> *w_rms_next_input_gate;

	Matrix<float>* w_grad_next_forget_gate;
	Matrix<float> *w_next_forget_gate;
	Matrix<float> *w_rms_next_forget_gate;

	Matrix<float>* w_grad_next_output_gate;
	Matrix<float> *w_next_output_gate;
	Matrix<float> *w_rms_next_output_gate;

	Matrix<float>* r_grad_next_input;
	Matrix<float> *r_next_input;
	Matrix<float> *r_rms_next_input;

	Matrix<float>* r_grad_next_input_gate;
	Matrix<float> *r_next_input_gate;
	Matrix<float> *r_rms_next_input_gate;

	Matrix<float>* r_grad_next_forget_gate;
	Matrix<float> *r_next_forget_gate;
	Matrix<float> *r_rms_next_forget_gate;

	Matrix<float>* r_grad_next_output_gate;
	Matrix<float> *r_next_output_gate;
	Matrix<float> *r_rms_next_output_gate;


	Matrix<float> *bw_rms_next_input;
	Matrix<float> *bw_grad_next_input;
	Matrix<float> *bw_next_input;

	Matrix<float> *bw_rms_next_input_gate;
	Matrix<float> *bw_grad_next_input_gate;
	Matrix<float> *bw_next_input_gate;

	Matrix<float> *bw_rms_next_forget_gate;
	Matrix<float> *bw_grad_next_forget_gate;
	Matrix<float> *bw_next_forget_gate;

	Matrix<float> *bw_rms_next_output_gate;
	Matrix<float> *bw_grad_next_output_gate;
	Matrix<float> *bw_next_output_gate;


	Matrix <float> * activation_R_input_batch;
	Matrix <float> * activation_R_input_gate_batch;
	Matrix <float> * activation_R_forget_gate_batch;
	Matrix <float> * activation_R_output_gate_batch;

	Matrix <float> * skip_activations_full;
	Matrix <float> * skip_activations_input_gate_full;
	Matrix <float> * skip_activations_forget_gate_full;
	Matrix <float> * skip_activations_output_gate_full;

	Matrix <float> * output_full;
	Matrix <float> * activations_input_full;
	Matrix <float> * activations_input_gate_full;
	Matrix <float> * activations_forget_gate_full;
	Matrix <float> * activations_output_gate_full;
	Matrix <float> * activations_cell_full;
	Matrix <float> * activations_post_cell_full;

	Matrix <float> * grad_input_full;
	Matrix <float> * grad_input_gate_full;
	Matrix <float> * grad_forget_gate_full;
	Matrix <float> * grad_output_gate_full;
	Matrix <float> * grad_cell_full;

	Matrix <float> * error_input_full;
	Matrix <float> * error_input_gate_full;
	Matrix <float> * error_forget_gate_full;
	Matrix <float> * error_output_gate_full;
	Matrix <float> * error_output_full;

	Matrix <float> * activation_cell_buffer;
	Matrix <float> * error_cell_prev;
	Matrix <float> * error_cell_current;
	Matrix <float> * error_cell_buffer;

	vector<Matrix <float> *> skip_activations_batch;

	vector<Matrix <float> * > output_batch;
	vector<Matrix <float> * > activations_input_batch;
	vector<Matrix <float> * > activations_input_gate_batch;
	vector<Matrix <float> * > activations_forget_gate_batch;
	vector<Matrix <float> * > activations_output_gate_batch;
	vector<Matrix <float> * > activations_cell_batch;
	vector<Matrix <float> * > activations_cell_tanh_batch;

	vector<Matrix <float> * > grad_input_batch;
	vector<Matrix <float> * > grad_input_gate_batch;
	vector<Matrix <float> * > grad_forget_gate_batch;
	vector<Matrix <float> * > grad_output_gate_batch;
	vector<Matrix <float> * > grad_cell_tanh_batch;

	vector<Matrix <float> * > error_input_batch;
	vector<Matrix <float> * > error_input_gate_batch;
	vector<Matrix <float> * > error_forget_gate_batch;
	vector<Matrix <float> * > error_output_gate_batch;
	vector<Matrix <float> * > error_output_batch;

	Layer_t layer_type;

	int CURRENT_TIME_STEP;
	int MAX_TIME_STEP_BATCH;
	int MAX_TIME_STEP;




	Matrix<float> *target;
	Matrix<float> *target_matrix;

	Matrix<float> *result;
	Matrix<float> *eq;

	ClusterNet *GPU;


	int UNITCOUNT;
	int BATCH_SIZE;
	int Layer_ID;
	RecurrentNetwork *_network;

	Matrix<float> *get_forward_activation();

	void forward();
	void forward_to_output();
	void forward_to_skip_connections();

	void backward_errors();
	void backward_grads();
	/*
	virtual void forward() = 0;
	virtual void backward_errors() = 0;
	virtual void backward_grads() = 0;

	void init(int unitcount, Unittype_t unitType, Layer_t layerType);
	void apply_transformations();
	void init_transformers(ClusterNet *gpu, Network *net);
	void init_transformer_activations(int batch_size);
	*/

	ActivationFunction *_func;

	std::vector< Transformer*> _transformer;

	Configurator *_conf;

};

#endif /* LSTMLAYER_H_ */
