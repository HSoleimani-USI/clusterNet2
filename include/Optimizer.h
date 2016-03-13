/*
 * Optimizer.h
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <ClusterNet.h>

class Optimizer
{
public:
	Optimizer(ClusterNet *gpu, WeightUpdateType_t updatetype);
	~Optimizer(){};

	void weight_update(Matrix<float> *accelerator, Matrix<float> *weight, Matrix<float> *grad,
				       float accelerator_value, float learning_rate);

	WeightUpdateType_t _updatetype;
	ClusterNet *GPU;

};

#endif /* OPTIMIZER_H_ */
