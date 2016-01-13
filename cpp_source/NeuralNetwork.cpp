/*
 * NeuralNetwork.cpp
 *
 *  Created on: Jan 13, 2016
 *      Author: tim
 */

#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(BatchAllocator b)
{

}

void NeuralNetwork::add(Layer layer)
{
	_layers.push_back(layer);
}
