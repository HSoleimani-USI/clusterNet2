
/*
 * BasicOpsWrapper.h
 *
 *  Created on: Mar 12, 2016
 *      Author: tim
 */

#ifndef GRADIENTACCUMULATOR_H_
#define GRADIENTACCUMULATOR_H_

#include <Matrix.h>
#include <mpif.h>
#include <vector>



class GradientAccumulator{

public:

	void init_MPI(int argc, char** argv);
	void init_Matrix (Matrix<float> * m);
	void send_MPI();
	     int my_rank;
	     int node_count;
	     Matrix<float> *buffer;
	     Matrix<float> *matrix;
	     std : vector<Matrix <float> *> v;
	     std : vector<Matrix <float> *> b;

	void recv_MPI();

		 

};










#endif /* GRADIENTACCUMULATOR_H_ */
