/*
 * NeuralNetwork.h
 *
 *  Created on: Jan 13, 2016
 *      Author: tim
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <Layer.h>
#include <BatchAllocator.h>

class NeuralNetwork
{
public:
	NeuralNetwork(BatchAllocator b);
	void add(Layer layer);
private:
	std::vector<Layer> _layers;
};

#endif /* NEURALNETWORK_H_ */
