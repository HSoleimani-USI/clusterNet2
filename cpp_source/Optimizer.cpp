/*
 * Optimizer.cpp
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#include "Optimizer.h"

Optimizer::Optimizer(WeightUpdateType_t updatetype)
{
	_updatetype = updatetype;
}

void Optimizer::weight_update(Matrix<float> *accelerator, Matrix<float> *weight, Matrix<float> *grad, float accelerator_value, float learning_rate)
{
	//cout << "pre: "<< reduceToValue<rsum>(grad) << endl;
	//cout << "pre w: "<< reduceToValue<rsum>(weight) << endl;
	switch(_updatetype)
	{
		case RMSProp:
			WeightUpdate<RMSProp>(accelerator,grad,weight,accelerator_value,learning_rate);
			break;
		case RMSPropInit:
			WeightUpdate<RMSPropInit>(accelerator,grad,weight,accelerator_value,learning_rate);
			break;
		case PlainSGD:
			elementWise<ksmul>(grad,grad,learning_rate);
			elementWise<ksub>(weight,grad,weight);
			break;
		default:
			cout << "Unknown update type!" << endl;
			throw "Unknown update type!";
			break;
	}
	//cout << "post: "<< reduceToValue<rsum>(grad) << endl;
	//cout << "post w: "<< reduceToValue<rsum>(weight) << endl;
}
