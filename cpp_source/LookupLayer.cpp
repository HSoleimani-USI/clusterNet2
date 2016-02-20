/*
 * LookupLayer.cpp
 *
 *  Created on: Feb 19, 2016
 *      Author: tim
 */

#include "LookupLayer.h"
#include <Configurator.h>



LookupLayer::LookupLayer(int embedding_columns, std::map<std::string,int> vocab2idx)
{
	init(embedding_columns, Input, Lookup);
	_vocab2idx = vocab2idx;
	_embeddings = NULL;
	_rms_embedding = NULL;
}
void LookupLayer::init_embeddings(Matrix<float> *embeddings)
{
	if(!_embeddings){ _embeddings->free_matrix(); }
	_embeddings = embeddings;
}



void LookupLayer::forward()
{

	if(!prev){ apply_transformations(); next->forward(); return; }

	lookup(_embeddings, prev->get_forward_activation(), activation);
	apply_transformations();

    if(next){ next->forward(); }
}

void LookupLayer::backward_errors()
{
	if(!target){ next->backward_errors(); }
	GPU->dotT(next->error, w_next,error);
}

void LookupLayer::backward_grads()
{
	GPU->Tdot(activation, next->error, w_grad_next);
	if(!next->target){ next->backward_grads(); }
	reduceToCols<rmean>(next->error,b_grad_next);
}


void LookupLayer::update_embeddings()
{
	embeddingUpdate(_embeddings, prev->get_forward_activation(), error, _rms_embedding, _conf->RMSPROP_MOMENTUM, _conf->LEARNING_RATE);
}

