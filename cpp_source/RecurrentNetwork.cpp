/*
 * RecurrentNetwork.cpp
 *
 *  Created on: Mar 5, 2016
 *      Author: tim
 */

#include "RecurrentNetwork.h"
#include <LSTMLayer.h>
#include <BatchAllocator.h>
#include <Optimizer.h>
#include <ErrorHandler.h>
#include <Configurator.h>


RecurrentNetwork::RecurrentNetwork(ClusterNet *gpu)
{
	_isTrainTime = true;
	_gpu = gpu;
	_errorhandler = new ErrorHandler(gpu);
	_conf = new Configurator();
	_opt = new Optimizer(gpu, RMSProp);
}

void RecurrentNetwork::add(LSTMLayer *layer)
{
	LSTMLayer *prev;
	if(_layers.size() > 0)
	{
		prev = _layers.back();
		layer->prev = prev;
		prev->next = layer;
	}

	layer->Layer_ID = _layers.size();
	_layers.push_back(layer);
	layer->GPU = _gpu;
	layer->_network = this;
	layer->input = _layers.front();
	//layer->init_transformers(_gpu, this);

}

void RecurrentNetwork::init_weights(WeightInitType_t wtype)
{


	LSTMLayer *input = _layers.front();
	for(int i = 1; i < _layers.size(); i++)
	{
		if(wtype == UniformSqrt)
		{

		}
	}

	/*
	LSTMLayer *prev = NULL;
	for(int i = 0; i < _layers.size(); i++)
	{
		if(!prev){ prev = _layers[i]; continue; }

		if(wtype == Gaussian)
		{
			prev->w_next = _gpu->normal(prev->UNITCOUNT,_layers[i]->UNITCOUNT,0.0f,0.0001f);
		}
		else if(wtype == UniformSqrt)
		{
			prev->w_next = _gpu->rand(prev->UNITCOUNT,_layers[i]->UNITCOUNT);
			float range = 8.0f*sqrtf(6.0f/((float)prev->UNITCOUNT + _layers[i]->UNITCOUNT));
			elementWise<ksmul>(prev->w_next,prev->w_next,range);
			elementWise<kssub>(prev->w_next,prev->w_next,range/2.0f);
		}

		prev->w_rms_next = zeros<float>(prev->UNITCOUNT,_layers[i]->UNITCOUNT);
		prev->w_grad_next = zeros<float>(prev->UNITCOUNT,_layers[i]->UNITCOUNT);

		prev->b_next = zeros<float>(1,_layers[i]->UNITCOUNT);
		prev->b_grad_next = zeros<float>(1,_layers[i]->UNITCOUNT);
		prev->b_rms_next = zeros<float>(1,_layers[i]->UNITCOUNT);


		prev = _layers[i];
	}
	*/

}

void RecurrentNetwork::init_activations(int batchsize)
{
	/*
	_layers.front()->init_transformer_activations(batchsize);

	for(int i = 1; i < _layers.size(); i++)
	{
		_layers[i]->init_transformer_activations(batchsize);
		if(_layers[i]->activation != NULL && _layers[i]->activation->rows == batchsize){ return; }

		if(_layers[i]->activation != NULL)
		{
			_layers[i]->activation->free_matrix();
			_layers[i]->activation_grad->free_matrix();
			_layers[i]->error->free_matrix();
			if(i == _layers.size()-1){ _layers[i]->target_matrix->free_matrix(); }
		}

		_layers[i]->activation = zeros<float>(batchsize, _layers[i]->UNITCOUNT);
		_layers[i]->activation_grad = zeros<float>(batchsize, _layers[i]->UNITCOUNT);
		_layers[i]->error = zeros<float>(batchsize, _layers[i]->UNITCOUNT);
		if(i == _layers.size()-1){ _layers[i]->target_matrix = zeros<float>(batchsize, _layers[i]->UNITCOUNT);}
	}
	*/

}

void RecurrentNetwork::copy_global_params_to_layers()
{
	for(int i = 0; i < _layers.size(); i++)
	{
		_layers[i]->_conf->LEARNING_RATE = _conf->LEARNING_RATE;
		_layers[i]->_conf->RMSPROP_MOMENTUM = _conf->RMSPROP_MOMENTUM;
		_layers[i]->_conf->DROPOUT = _conf->DROPOUT;
		_layers[i]->_conf->LEARNING_RATE_DECAY = _conf->LEARNING_RATE_DECAY;
	}
}
