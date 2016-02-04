/*
 * NeuralNetwork.cpp
 *
 *  Created on: Jan 13, 2016
 *      Author: tim
 */

#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(ClusterNet *gpu, BatchAllocator *b_train, BatchAllocator *b_cv, std::vector<int> FCLayers, Unittype_t unit, int classes)
{
	_gpu = gpu;

	_b_train = b_train;
	_b_cv = b_cv;

	_b_train->allocate_next_batch_async();
	_b_cv->allocate_next_batch_async();

	cout << "neural net init" << endl;
	FCLayer *l0 = new FCLayer(_b_train->get_current_batchX()->cols,128,Input,_gpu);
	cout << "neural net init2" << endl;
	l0->DROPOUT = 0.2f;
	FCLayer *prev = l0;
	for(int i = 0; i < FCLayers.size(); i++)
	{
		cout << "neural net init3" << endl;
		FCLayer *next = new FCLayer(FCLayers[i], unit, prev);
		prev = next;
	}



	cout << "neural net init4" << endl;
	FCLayer *next = new FCLayer(classes, Softmax, prev);

	cout << "post softmax" << endl;

	start = l0;
	end = next;
	l0->set_hidden_dropout(0.3f);
	cout << "post hidden dropout set" << endl;
}

void NeuralNetwork::fit()
{

	float decay = 0.99f;
	for(int epoch = 0; epoch < 100; epoch++)
	{
		cout << "EPOCH: " << epoch + 1 << endl;


		for(int i = 0; i < _b_train->BATCHES; i++)
		{
			cout << "batch init" << endl;
			_b_train->replace_current_with_next_batch();
			_b_train->allocate_next_batch_async();
			start->activation = _b_train->get_current_batchX();
			end->target = _b_train->get_current_batchY();

			cout << "forward" << endl;
			start->forward(true);
			cout << "backwards errors" << endl;
			start->backward_errors();
			cout << "backwards grads" << endl;
			start->backward_grads();
			cout << "weight update" << endl;
			start->weight_update();

		}
		_b_train->replace_current_with_next_batch();
		_b_train->allocate_next_batch_async();

		for(int i = 0; i < _b_train->BATCHES; i++)
		{
			_b_train->replace_current_with_next_batch();
			_b_train->allocate_next_batch_async();
			start->activation = _b_train->get_current_batchX();
			end->target = _b_train->get_current_batchY();

			start->forward(false);
			start->running_error();
		}
		_b_train->replace_current_with_next_batch();
		_b_train->allocate_next_batch_async();
		start->print_error("Train error: ");


		for(int i = 0; i < _b_cv->BATCHES; i++)
		{
			_b_cv->replace_current_with_next_batch();
			_b_cv->allocate_next_batch_async();
			start->activation = _b_cv->get_current_batchX();
			end->target = _b_cv->get_current_batchY();

			start->forward(false);
			start->running_error();
		}
		_b_cv->replace_current_with_next_batch();
		_b_cv->allocate_next_batch_async();
		start->print_error("CV error: ");

		start->learning_rate_decay(decay);

		if(epoch == 60)
		{
			//l0->dropout_decay();
			//decay = 0.85f;
		}
	}

}
