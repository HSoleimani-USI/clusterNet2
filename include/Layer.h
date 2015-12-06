#ifndef Layer_H
#define Layer_H
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <basicOps.cuh>
#include <clusterNet2.h>

template <typename T> class Layer
{
public:
	Matrix<T> *b_grad_next;
	Layer *next;
	Layer *prev;
	Matrix<T> *w_next;
	Matrix<T> *b_next;

	std::vector<Matrix<T>* > vec_w_grad_next;
	std::vector<Matrix<T>* > vec_w_grad_next_8bit;

	Matrix<T> *w_rms_next;
	Matrix<T> *b_rms_next;

	Matrix<T> *bias_activations;
	Matrix<T> *out;
	Matrix<T> *error;
	Matrix<T> *activation;

	Matrix<T> *out_offsize;
	Matrix<T> *activation_offsize;
	Matrix<T> *error_offsize;
	Matrix<T> *bias_activations_offsize;
	Matrix<T> *target_matrix_offsize;

	Matrix<T> *target;
	Matrix<T> *target_matrix;

	ClusterNet2<T> *GPU;

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
	Layer(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<T> *gpu);
	Layer(int unitcount, Unittype_t unit);
	Layer(int unitcount);

	Layer(int unitcount, int start_batch_size, Unittype_t unit, Layer<T> *prev, ClusterNet2<T> *gpu);
	Layer(int unitcount, Unittype_t unit, Layer<T> *prev);
	Layer(int unitcount, Layer<T> *prev);

	virtual void forward();
	virtual void forward(bool useDropout);
	virtual void running_error();
	virtual void backward_errors();
	virtual void backward_grads();
	virtual void print_error(std::string message);
	virtual void weight_update();

	virtual void MPI_synchronization_async();
	virtual void wait_for_synchronization();

	virtual void limit_magnitude();

	virtual void link_with_next_layer(Layer<T> *next_layer);
	virtual void init(int unitcount, int start_batch_size, Unittype_t unit, ClusterNet2<T> *gpu);

	virtual void set_hidden_dropout(float dropout);

	virtual void dropout_decay();
	virtual void learning_rate_decay(float decay_rate);

	virtual Layer<T> *get_root();



private:
	virtual void unit_activation();
	virtual void unit_activation(bool useDropout);
	virtual void activation_gradient();
	virtual void apply_dropout();
	void handle_offsize();


};

#endif
