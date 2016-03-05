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
		elementWise<kadd>(activations_input_full,skip_activations_full,activations_input_full);
		elementWise<kadd>(activations_input_gate_full,skip_activations_input_gate_full,activations_input_gate_full);
		elementWise<kadd>(activations_forget_gate_full,skip_activations_forget_gate_full,activations_forget_gate_full);
		elementWise<kadd>(activations_output_gate_full,skip_activations_output_gate_full,activations_output_gate_full);
	}

	vectorWise<kvadd>(activations_input_full, prev->bw_next_input, activations_input_full);
	vectorWise<kvadd>(activations_input_gate_full, prev->bw_next_input_gate, activations_input_gate_full);
	vectorWise<kvadd>(activations_forget_gate_full, prev->bw_next_forget_gate, activations_forget_gate_full);
	vectorWise<kvadd>(activations_output_gate_full, prev->bw_next_output_gate, activations_output_gate_full);

	elementWise<ksmul>(activation_R_input_batch,activation_R_input_batch,0.0f);
	elementWise<ksmul>(activation_R_input_gate_batch,activation_R_input_gate_batch,0.0f);
	elementWise<ksmul>(activation_R_forget_gate_batch,activation_R_forget_gate_batch,0.0f);
	elementWise<ksmul>(activation_R_output_gate_batch,activation_R_output_gate_batch,0.0f);

	for(int i = 0; i < MAX_TIME_STEP_BATCH; i++)
	{
		elementWise<kadd>(activations_input_batch[i], activation_R_input_batch,activations_input_batch[i]);
		elementWise<ktanh>(activations_input_batch[i], activations_input_batch[i]);

		elementWise<kadd>(activations_input_gate_batch[i], activation_R_input_gate_batch,activations_input_gate_batch[i]);
		elementWise<klogistic>(activations_input_gate_batch[i], activations_input_gate_batch[i]);

		elementWise<kadd>(activations_forget_gate_batch[i], activation_R_forget_gate_batch,activations_forget_gate_batch[i]);
		elementWise<klogistic>(activations_forget_gate_batch[i], activations_forget_gate_batch[i]);

		elementWise<kadd>(activations_output_gate_batch[i], activation_R_output_gate_batch,activations_output_gate_batch[i]);
		elementWise<klogistic>(activations_output_gate_batch[i], activations_output_gate_batch[i]);

		elementWise<kmul>(activations_input_batch[i],activations_input_gate_batch[i], activation_cell_buffer);
		elementWise<kmul>(activations_cell_batch[i-1],activations_forget_gate_batch[i], activations_cell_batch[i]);
		elementWise<kadd>(activations_cell_batch[i],activation_cell_buffer, activations_cell_batch[i]);

		//activation_output = get_view(activation_output_full, i,i+1);
		elementWise<ktanh>(output_batch[i], output_batch[i]);
		elementWise<kmul>(output_batch[i],activations_output_gate_batch[i], output_batch[i]);


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
		if(output_full->cols != target->cols && !target_matrix){ target_matrix = zeros<float>(BATCH_SIZE*MAX_TIME_STEP,output_full->cols); }
		if(output_full->cols != target->cols)
		{
			vectorWise<ktmatrix>(target, target_matrix);
			elementWise<ksub>(output_full,target_matrix,error_output_full);
		}
		else{ elementWise<ksub>(output_full,target,error_output_full); }


		elementWise<ksmul>(error_output_full,error_output_full,1.0f/error_output_batch[0]->rows);

		return;

	}



	if(layer_type == InputLayer){ backward_grads(); return; }

	for(int i = MAX_TIME_STEP_BATCH; i > 0; i--)
	{
		//output gate error
		elementWise<klogistic_grad>(activations_output_gate_batch[i], grad_output_gate_batch[i]);
		elementWise<kmul>(error_output_batch[i], activations_cell_tanh_batch[i], error_output_gate_batch[i]);
		elementWise<kmul>(error_output_gate_batch[i], grad_output_gate_batch[i], error_output_gate_batch[i]);

		//cell error
		elementWise<ksmul>(error_cell_current,error_cell_current,0.0f);
		elementWise<ktanh_grad>(activations_cell_tanh_batch[i], grad_cell_tanh_batch[i]);
		elementWise<kmul>(error_output_batch[i], activations_output_gate_batch[i], error_cell_current);
		elementWise<kmul>(error_cell_current, grad_cell_tanh_batch[i], error_cell_current);
		if(i<MAX_TIME_STEP_BATCH)
		{
			elementWise<kmul>(error_cell_prev, activations_forget_gate_batch[i+1], error_cell_prev);
			elementWise<kadd>(error_cell_current, error_cell_prev, error_cell_current);
		}

		if(i > 0)
		{
			//forget gate error
			elementWise<klogistic_grad>(activations_forget_gate_batch[i], grad_forget_gate_batch[i]);
			elementWise<kmul>(error_cell_current, activations_cell_batch[i-1], error_forget_gate_batch[i]);
			elementWise<kmul>(error_forget_gate_batch[i], grad_forget_gate_batch[i] , error_forget_gate_batch[i]);
		}

		//input gate error
		elementWise<klogistic_grad>(activations_input_gate_batch[i], grad_input_gate_batch[i]);
		elementWise<kmul>(error_cell_current, activations_input_batch[i], error_input_gate_batch[i]);
		elementWise<kmul>(error_input_gate_batch[i], grad_input_gate_batch[i] , error_input_gate_batch[i]);

		if(prev->layer_type != InputLayer)
		{
			//input error
			elementWise<ktanh_grad>(activations_input_batch[i], grad_input_batch[i]);
			elementWise<kmul>(error_cell_current, activations_input_gate_batch[i], error_input_batch[i]);
			elementWise<kmul>(error_input_batch[i], grad_input_batch[i] , error_input_batch[i]);
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

	reduceToCols<rmean>(next->error_input_full,bw_grad_next_input);
	reduceToCols<rmean>(next->error_input_gate_full,bw_grad_next_input_gate);
	reduceToCols<rmean>(next->error_forget_gate_full,bw_grad_next_forget_gate);
	reduceToCols<rmean>(next->error_output_gate_full,bw_grad_next_output_gate);

	if(!next->target){ next->backward_grads(); }
}



void LSTMLayer::forward_to_output()
{
	//GPU->dot(activation_output_full, w_output, output->activation_full);
	//vectorWise<kvadd>(activation_output_full, b_output, activation_output_full);
}


void LSTMLayer::forward_to_skip_connections()
{
	//GPU->dot(activation_output_full, w_output, output->activation_full);
	//vectorWise<kvadd>(activation_output_full, b_output, activation_output_full);
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
