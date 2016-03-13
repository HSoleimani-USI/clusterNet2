/*
 * LSTMLayer.cpp
 *
 *  Created on: Feb 20, 2016
 *      Author: tim
 */

#include "LSTMLayer.h"
#include <boost/swap.hpp>
#include <RecurrentNetwork.h>
#include <ActivationFunction.h>
#include <Configurator.h>
#include <Transformer.h>


void LSTMLayer::forward()
{
	if(!prev){  next->forward(); return; }
	GPU->dot(prev->output_full, prev->w_next_input, activations_input_full);
	GPU->dot(prev->output_full, prev->w_next_input_gate, activations_input_gate_full);
	GPU->dot(prev->output_full, prev->w_next_forget_gate, activations_forget_gate_full);
	GPU->dot(prev->output_full, prev->w_next_output_gate, activations_output_gate_full);


	if(prev != input)
	{
		//use skip connections to forward input
		GPU->dot(input->output_full, input->skip_weights[Layer_ID-1], skip_activations_full);
		GPU->dot(input->output_full, input->skip_weights_input_gate[Layer_ID-1], skip_activations_input_gate_full);
		GPU->dot(input->output_full, input->skip_weights_forget_gate[Layer_ID-1], skip_activations_forget_gate_full);
		GPU->dot(input->output_full, input->skip_weights_output_gate[Layer_ID-1], skip_activations_output_gate_full);

		//pool
		GPU->OPS->add(activations_input_full,skip_activations_full,activations_input_full);
		GPU->OPS->add(activations_input_gate_full,skip_activations_input_gate_full,activations_input_gate_full);
		GPU->OPS->add(activations_forget_gate_full,skip_activations_forget_gate_full,activations_forget_gate_full);
		GPU->OPS->add(activations_output_gate_full,skip_activations_output_gate_full,activations_output_gate_full);
	}

	GPU->OPS->vadd(activations_input_full, prev->bw_next_input, activations_input_full);
	GPU->OPS->vadd(activations_input_gate_full, prev->bw_next_input_gate, activations_input_gate_full);
	GPU->OPS->vadd(activations_forget_gate_full, prev->bw_next_forget_gate, activations_forget_gate_full);
	GPU->OPS->vadd(activations_output_gate_full, prev->bw_next_output_gate, activations_output_gate_full);

	GPU->OPS->mul(activation_R_input_batch,activation_R_input_batch,0.0f);
	GPU->OPS->mul(activation_R_input_gate_batch,activation_R_input_gate_batch,0.0f);
	GPU->OPS->mul(activation_R_forget_gate_batch,activation_R_forget_gate_batch,0.0f);
	GPU->OPS->mul(activation_R_output_gate_batch,activation_R_output_gate_batch,0.0f);

	for(int i = 0; i < MAX_TIME_STEP_BATCH; i++)
	{
		GPU->OPS->add(activations_input_batch[i], activation_R_input_batch,activations_input_batch[i]);
		GPU->OPS->tanh(activations_input_batch[i], activations_input_batch[i]);

		GPU->OPS->add(activations_input_gate_batch[i], activation_R_input_gate_batch,activations_input_gate_batch[i]);
		GPU->OPS->logistic(activations_input_gate_batch[i], activations_input_gate_batch[i]);

		GPU->OPS->add(activations_forget_gate_batch[i], activation_R_forget_gate_batch,activations_forget_gate_batch[i]);
		GPU->OPS->logistic(activations_forget_gate_batch[i], activations_forget_gate_batch[i]);

		GPU->OPS->add(activations_output_gate_batch[i], activation_R_output_gate_batch,activations_output_gate_batch[i]);
		GPU->OPS->logistic(activations_output_gate_batch[i], activations_output_gate_batch[i]);

		GPU->OPS->mul(activations_input_batch[i],activations_input_gate_batch[i], activation_cell_buffer);
		GPU->OPS->mul(activations_cell_batch[i-1],activations_forget_gate_batch[i], activations_cell_batch[i]);
		GPU->OPS->add(activations_cell_batch[i],activation_cell_buffer, activations_cell_batch[i]);

		//activation_output = get_view(activation_output_full, i,i+1);
		GPU->OPS->tanh(output_batch[i], output_batch[i]);
		GPU->OPS->mul(output_batch[i],activations_output_gate_batch[i], output_batch[i]);


		GPU->dot(output_batch[i],prev->r_next_input, activation_R_input_batch);
		GPU->dot(output_batch[i],prev->r_next_input_gate, activation_R_input_gate_batch);
		GPU->dot(output_batch[i],prev->r_next_forget_gate, activation_R_forget_gate_batch);
		GPU->dot(output_batch[i],prev->r_next_output_gate, activation_R_output_gate_batch);
	}

	if(next){ next->forward(); }
}

void LSTMLayer::backward_errors()
{

	if(layer_type != OutputLayer){ next->backward_errors(); }



	if(layer_type == OutputLayer)
	{
		if(output_full->cols != target->cols && !target_matrix){ target_matrix = GPU->OPS->zeros(BATCH_SIZE*MAX_TIME_STEP,output_full->cols); }
		if(output_full->cols != target->cols)
		{
			GPU->OPS->get_t_matrix(target, target_matrix);
			GPU->OPS->sub(output_full,target_matrix,error_output_full);
		}
		else{ GPU->OPS->sub(output_full,target,error_output_full); }


		GPU->OPS->mul(error_output_full,error_output_full,1.0f/error_output_batch[0]->rows);

		return;

	}



	if(layer_type == InputLayer){ backward_grads(); return; }

	for(int i = MAX_TIME_STEP_BATCH; i > 0; i--)
	{
		//output gate error
		GPU->OPS->logistic_grad(activations_output_gate_batch[i], grad_output_gate_batch[i]);
		GPU->OPS->mul(error_output_batch[i], activations_cell_tanh_batch[i], error_output_gate_batch[i]);
		GPU->OPS->mul(error_output_gate_batch[i], grad_output_gate_batch[i], error_output_gate_batch[i]);

		//cell error
		GPU->OPS->mul(error_cell_current,error_cell_current,0.0f);
		GPU->OPS->tanh_grad(activations_cell_tanh_batch[i], grad_cell_tanh_batch[i]);
		GPU->OPS->mul(error_output_batch[i], activations_output_gate_batch[i], error_cell_current);
		GPU->OPS->mul(error_cell_current, grad_cell_tanh_batch[i], error_cell_current);
		if(i<MAX_TIME_STEP_BATCH)
		{
			GPU->OPS->mul(error_cell_prev, activations_forget_gate_batch[i+1], error_cell_prev);
			GPU->OPS->add(error_cell_current, error_cell_prev, error_cell_current);
		}

		if(i > 0)
		{
			//forget gate error
			GPU->OPS->logistic_grad(activations_forget_gate_batch[i], grad_forget_gate_batch[i]);
			GPU->OPS->mul(error_cell_current, activations_cell_batch[i-1], error_forget_gate_batch[i]);
			GPU->OPS->mul(error_forget_gate_batch[i], grad_forget_gate_batch[i] , error_forget_gate_batch[i]);
		}

		//input gate error
		GPU->OPS->logistic_grad(activations_input_gate_batch[i], grad_input_gate_batch[i]);
		GPU->OPS->mul(error_cell_current, activations_input_batch[i], error_input_gate_batch[i]);
		GPU->OPS->mul(error_input_gate_batch[i], grad_input_gate_batch[i] , error_input_gate_batch[i]);

		if(prev->layer_type != InputLayer)
		{
			//input error
			GPU->OPS->tanh_grad(activations_input_batch[i], grad_input_batch[i]);
			GPU->OPS->mul(error_cell_current, activations_input_gate_batch[i], error_input_batch[i]);
			GPU->OPS->mul(error_input_batch[i], grad_input_batch[i] , error_input_batch[i]);
		}

		boost::swap(error_cell_prev,error_cell_current);

	}

}

void LSTMLayer::backward_grads()
{
	GPU->Tdot(get_forward_activation(), next->error_input_full, w_grad_next_input);
	GPU->Tdot(get_forward_activation(), next->error_input_gate_full, w_grad_next_input_gate);
	GPU->Tdot(get_forward_activation(), next->error_forget_gate_full, w_grad_next_forget_gate);
	GPU->Tdot(get_forward_activation(), next->error_output_gate_full, w_grad_next_output_gate);

	//TODO: fix time overlap
	GPU->Tdot(output_full, next->error_input_full, r_grad_next_input);
	GPU->Tdot(output_full, next->error_input_gate_full, r_grad_next_input_gate);
	GPU->Tdot(output_full, next->error_forget_gate_full, r_grad_next_forget_gate);
	GPU->Tdot(output_full, next->error_output_gate_full, r_grad_next_output_gate);

	GPU->OPS->mean_of_rows(next->error_input_full,bw_grad_next_input);
	GPU->OPS->mean_of_rows(next->error_input_gate_full,bw_grad_next_input_gate);
	GPU->OPS->mean_of_rows(next->error_forget_gate_full,bw_grad_next_forget_gate);
	GPU->OPS->mean_of_rows(next->error_output_gate_full,bw_grad_next_output_gate);

	if(!next->target){ next->backward_grads(); }
}



void LSTMLayer::forward_to_output()
{
	//GPU->dot(activation_output_full, w_output, output->activation_full);
	//GPU->OPS->vadd(activation_output_full, b_output, activation_output_full);
}


void LSTMLayer::forward_to_skip_connections()
{
	//GPU->dot(activation_output_full, w_output, output->activation_full);
	//GPU->OPS->vadd(activation_output_full, b_output, activation_output_full);
}

Matrix<float> *LSTMLayer::get_forward_activation()
{
	/*
	if(Layer_ID == 0)
	{
		//just the inputs
		return input->output_full;
	}
	else
	{
		//output + skip connection input
		output_full
	}
	*/
	return NULL;
}
