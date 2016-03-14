/*
 * Optimizer.cpp
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#include "Optimizer.h"
#include <stdio.h>
#include <iostream>

using std::endl;
using std::cout;

Optimizer::Optimizer(ClusterNet *gpu, WeightUpdateType_t updatetype)
{
	_updatetype = updatetype;
	GPU = gpu;
}

void Optimizer::weight_update(Matrix<float> *accelerator, Matrix<float> *weight, Matrix<float> *grad, float accelerator_value, float learning_rate)
{
	//cout << "pre: "<< reduceToValue<rsum>(grad) << endl;
	//cout << "pre w: "<< reduceToValue<rsum>(weight) << endl;
	switch(_updatetype)
	{
		case RMSProp:
			GPU->OPS->WeightUpdate_RMSProp(accelerator,grad,weight,accelerator_value,learning_rate);
			break;
		case RMSPropInit:
			cout << "Not implemented yet!" << endl;
			throw "Not implemented yet!";
			break;
		case PlainSGD:
			cout << "Not implemented yet!" << endl;
			throw "Not implemented yet!";
			break;
		default:
			cout << "Unknown update type!" << endl;
			throw "Unknown update type!";
			break;
	}
	//cout << "post: "<< reduceToValue<rsum>(grad) << endl;
	//cout << "post w: "<< reduceToValue<rsum>(weight) << endl;
}
