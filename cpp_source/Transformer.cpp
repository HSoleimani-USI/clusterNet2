/*
 * Transformer.cpp
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#include "Transformer.h"
#include <Network.h>
#include <Layer.h>
#include <Configurator.h>


Transformer::Transformer(TransformerType_t ttype, Layer *layer)
{
	_ttype = ttype;
	_gpu = NULL;
	_layer = layer;
	output = NULL;
	_net = NULL;
}


Matrix<float> *Transformer::transform(Matrix<float> *input)
{
	if(_net->_isTrainTime)
	{
		_gpu->dropout(input,output,_layer->_conf->DROPOUT);
	}
	else
	{
		elementWiseUnary<ksmul>(input,output,(1.0f-_layer->_conf->DROPOUT));
	}

	return output;
}
