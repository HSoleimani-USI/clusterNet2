/*
 * NeuralNetwork.h
 *
 *  Created on: Jan 13, 2016
 *      Author: tim
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <FCLayer.h>
#include <BatchAllocator.h>

class NeuralNetwork
{
public:
	NeuralNetwork(ClusterNet *gpu, BatchAllocator *b_train, BatchAllocator *b_cv, std::vector<int> FCLayers, Unittype_t unit, int classes);
	void fit();
private:
	BatchAllocator *_b_train;
	BatchAllocator *_b_cv;
	ClusterNet *_gpu;
	FCLayer *start;
	FCLayer *end;
};

#endif /* NEURALNETWORK_H_ */
