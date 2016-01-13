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
	NeuralNetwork(ClusterNet2<float> *gpu, BatchAllocator b_train, BatchAllocator b_cv);
	void run();
private:
	BatchAllocator _b_train;
	BatchAllocator _b_cv;
	ClusterNet2<float> *_gpu;
};

#endif /* NEURALNETWORK_H_ */
