
/*
 * BasicOpsWrapper.h
 *
i *  Created on: Mar 12, 2016
 *      Author: tim
 */

#ifndef GRADIENTACCUMULATOR_H_
#define GRADIENTACCUMULATOR_H_

#include <Matrix.h>
#include <vector>
#include <ClusterNet.h>



class GradientAccumulator{



public:
	int my_rank;
	int node_count;
	ClusterNet *cn;
	Matrix<float> *buffer;
	Matrix<float> *matrix;
	std :: vector<Matrix <float> *> v;
	std :: vector<Matrix <float> *> b;

#ifdef PHI
	void init_MPI();
	void init_Matrix (Matrix<float> * m);
	void send_MPI();
	void recv_MPI();

        GradientAccumulator(ClusterNet *clusterNet);
#endif


};










#endif /* GRADIENTACCUMULATOR_H_ */
