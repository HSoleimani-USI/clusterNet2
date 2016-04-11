
/*
 * BasicOpsWrapper.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#ifndef GRADIENTACCUMULATOR_H_
#define GRADIENTACCUMULATOR_H_

#include <Matrix.h>
#include <mpi.h>
#include <vector>
#include "ClusterNet.h"



class GradientAccumulator{



public:
	int my_rank;
	int node_count;
	ClusterNet *cn;
	Matrix<float> *buffer;
	Matrix<float> *matrix;
	std :: vector<Matrix <float> *> v;
	std :: vector<Matrix <float> *> b;

	void init_MPI(int argc, char** argv);
	void init_Matrix (Matrix<float> * m);
	void send_MPI();
	void recv_MPI();


    GradientAccumulator();


};










#endif /* GRADIENTACCUMULATOR_H_ */
