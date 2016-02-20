/*
 * LookupLayer.cpp
 *
 *  Created on: Feb 19, 2016
 *      Author: tim
 */

#include "LookupLayer.h"


LookupLayer::LookupLayer(int unitcount, std::map<std::string,int> vocab2idx, Matrix<float> *embeddings)
{
	init(unitcount, Input);
	_vocab2idx = vocab2idx;
	_embeddings = embeddings;

}

LookupLayer::LookupLayer(int unitcount, std::map<std::string,int> vocab2idx)
{
	init(unitcount, Input);
	_vocab2idx = vocab2idx;
}



void LookupLayer::forward()
{

	lookup(_embeddings, prev->get_forward_activation(), activation);

}



