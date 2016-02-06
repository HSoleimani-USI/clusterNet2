/*
 * ErrorHandler.h
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#ifndef ERRORHANDLER_H_
#define ERRORHANDLER_H_
#include <string>
#include <vector>

template <typename T> class Matrix;

class ErrorHandler
{
public:
	ErrorHandler();
	~ErrorHandler(){};


	float RUNNING_MEAN;
	float RUNNING_MEAN_DIFFERENCE;
	float RUNNING_VARIANCE;
	float RUNNING_ERROR;
	float RUNNING_SAMPLE_SIZE;
	float RUNNING_BATCHES;
	std::vector<float> _errors;

	void reset();
	void add_error(Matrix<float> *predictions, Matrix<float> *labels);
	void print_error(std::string message);
	void init_buffers(Matrix<float> *predictions, Matrix<float> *labels);

protected:
	Matrix <float> *result;
	Matrix <float> *eq;
};

#endif /* ERRORHANDLER_H_ */
