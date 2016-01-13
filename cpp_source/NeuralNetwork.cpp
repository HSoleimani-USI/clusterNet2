/*
 * NeuralNetwork.cpp
 *
 *  Created on: Jan 13, 2016
 *      Author: tim
 */

#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(ClusterNet2<float> *gpu, BatchAllocator b_train, BatchAllocator b_cv)
{
	_gpu = gpu;

	_b_train = b_train;
	_b_cv = b_cv;

	_b_train.allocate_next_batch_async();
	_b_cv.allocate_next_batch_async();



}

void NeuralNetwork::run()
{
	Layer *l0 = new Layer(_b_train.get_current_batchX()->cols,128,Input,_gpu);
	Layer *l1 = new Layer(1024, Rectified_Linear, l0);
	Layer *l2 = new Layer(1024, Rectified_Linear, l1);
	Layer *l3 = new Layer(2, Softmax, l2);
	l0->DROPOUT = 0.2f;
	l0->set_hidden_dropout(0.3f);

	Matrix<float> *X;
	Matrix<float> *y;
	float decay = 0.99f;
	for(int epoch = 0; epoch < 10000; epoch++)
	{
		cout << "EPOCH: " << epoch + 1 << endl;


		for(int i = 0; i < _b_train.BATCHES; i++)
		{
			_b_train.replace_current_with_next_batch();
			_b_train.allocate_next_batch_async();
			l0->activation = _b_train.get_current_batchX();
			l3->target = _b_train.get_current_batchY();

			l0->forward(true);
			l0->backward_errors();
			l0->backward_grads();
			l0->weight_update();
		}

		for(int i = 0; i < _b_train.BATCHES; i++)
		{
			_b_train.replace_current_with_next_batch();
			_b_train.allocate_next_batch_async();
			l0->activation = _b_train.get_current_batchX();
			l3->target = _b_train.get_current_batchY();

			l0->forward(false);
			l0->running_error();
		}
		l0->print_error("Train error: ");


		for(int i = 0; i < _b_cv.BATCHES; i++)
		{
			_b_cv.replace_current_with_next_batch();
			_b_cv.allocate_next_batch_async();
			l0->activation = _b_cv.get_current_batchX();
			l3->target = _b_cv.get_current_batchY();

			l0->forward(false);
			l0->running_error();
		}
		l0->print_error("CV error: ");

		l0->learning_rate_decay(decay);

		if(epoch == 60)
		{
			//l0->dropout_decay();
			//decay = 0.85f;
		}
	}

}
