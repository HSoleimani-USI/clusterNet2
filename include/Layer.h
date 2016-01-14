#ifndef Layer_H
#define Layer_H
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <clusterNet2.h>

class Layer
{
public:
	Matrix<float> *b_grad_next;
	Layer *next;
	Layer *prev;
	Matrix<float> *w_next;
	Matrix<float> *b_next;

	Matrix<float>* w_grad_next;

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

	ClusterNet2<float> *GPU;

	int count;
	int count2;

	float LEARNING_RATE;
	float MOMENTUM;
	float RMSPROP_MOMENTUM;
	float RUNNING_ERROR;
	float RUNNING_SAMPLE_SIZE;
	float L2;
    Unittype_t UNIT_TYPE;
	Costfunction_t COST;
	float DROPOUT;
	int UNITCOUNT;
	int BATCH_SIZE;
	int LAYER_ID;



	WeightUpdateType_t UPDATE_TYPE;

	virtual ~Layer();
	Layer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<float> *gpu);
	Layer(int unitcount, Unittype_t unit);
	Layer(int unitcount);

	Layer(int unitcount, int start_batch_size, Unittype_t unit, Layer *prev, ClusterNet2<float> *gpu);
	Layer(int unitcount, Unittype_t unit, Layer *prev);
	Layer(int unitcount, Layer *prev);

	virtual void forward();
	virtual void forward(bool useDropout);
	virtual void running_error();
	virtual void backward_errors();
	virtual void backward_grads();
	virtual void print_error(std::string message);
	virtual void weight_update();

	virtual void link_with_next_layer(Layer *next_layer);
	virtual void init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<float> *gpu);

	virtual void set_hidden_dropout(float dropout);

	virtual void dropout_decay();
	virtual void learning_rate_decay(float decay_rate);

	virtual Layer *get_root();



private:
	virtual void unit_activation();
	virtual void unit_activation(bool useDropout);
	virtual void activation_gradient();
	virtual void apply_dropout();
	void handle_offsize();


};

#endif
