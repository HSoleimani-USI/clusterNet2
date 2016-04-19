/*
 * ErrorHandler.cpp
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#include "ErrorHandler.h"
#include <BasicOpsWrapper.h>
#include <cmath>
#include <iomanip>


using std::cout;
using std::endl;

ErrorHandler::ErrorHandler(ClusterNet *gpu)
{
	RUNNING_MEAN = 0.0f;
	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
	RUNNING_MEAN_DIFFERENCE = 0.0f;
	RUNNING_VARIANCE = 0.0f;
	RUNNING_BATCHES = 0.0f;
	result = NULL;
	eq = NULL;
	GPU = gpu;

	_errors = std::vector<float>();
}

void ErrorHandler::init_buffers(Matrix<float> *predictions, Matrix<float> *labels)
{

	if(!eq)
	{
		result = GPU->OPS->zeros(labels->rows,labels->cols);
		eq = GPU->OPS->zeros(labels->rows, labels->cols);
	}
	else if(labels->rows != result->rows || labels->cols != result->cols)
	{
		result = GPU->OPS->zeros(labels->rows,labels->cols);
		eq = GPU->OPS->zeros(labels->rows, labels->cols);
	}


}


void ErrorHandler::reset()
{
	RUNNING_MEAN = 0.0f;
	RUNNING_ERROR = 0.0f;
	RUNNING_SAMPLE_SIZE = 0.0f;
	RUNNING_VARIANCE= 0.0f;
	RUNNING_MEAN_DIFFERENCE = 0.0f;
	RUNNING_BATCHES = 0.0f;
	_errors.clear();
}

void ErrorHandler::add_error(Matrix<float> *predictions, Matrix<float> *labels)
{
	if(_errors.size() == 0){ init_buffers(predictions, labels); }
	GPU->OPS->argmax(predictions, result);
	GPU->OPS->equal(result,labels,eq);
	float sum_value = GPU->OPS->sum(eq);
	_errors.push_back(((predictions->rows  - sum_value)/(float)predictions->rows));
	RUNNING_ERROR += (predictions->rows  - sum_value);
	RUNNING_SAMPLE_SIZE += predictions->rows;
	RUNNING_BATCHES += 1;
	float new_mean = RUNNING_MEAN + ((_errors.back()-RUNNING_MEAN)/(float)_errors.size());
	RUNNING_MEAN_DIFFERENCE = RUNNING_MEAN_DIFFERENCE + ((_errors.back() - RUNNING_MEAN)*(_errors.back() - new_mean));
	RUNNING_MEAN = new_mean;
	if(_errors.size() > 1)
		RUNNING_VARIANCE = RUNNING_MEAN_DIFFERENCE/(float)_errors.size();
}

void ErrorHandler::print_error(std::string message)
{
	std::string value = message;
	std::string strEmpty(25-message.length(),' ');
	float standard_error = (1.96*sqrtf(RUNNING_VARIANCE)/sqrtf(RUNNING_BATCHES));


	printf("%s %s %1.4f (%1.4f,%1.4f)\n", message.c_str(), strEmpty.c_str(), (RUNNING_ERROR/RUNNING_SAMPLE_SIZE), (RUNNING_MEAN - standard_error),(RUNNING_MEAN + standard_error));

	reset();
}

