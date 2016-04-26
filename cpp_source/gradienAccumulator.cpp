/*
 * gradientAccumulator.cpp
 *
 *  Created on: Apr 4, 2016
 *      Author: Hanieh
 */

#include "gradientAccumulator.h"



#ifdef HDF5
	#include <hdf5.h>
#endif

using std::cout;
using std::endl;


#ifdef PHI
	#include <mpi.h>


	GradientAccumulator::GradientAccumulator(ClusterNet *clusterNet){

	  cn = clusterNet;
	}



	void GradientAccumulator::init_MPI(int argc, char** argv) {

	  MPI_Init(&argc , &argv);
	  
	  int matrix_rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &matrix_rank);
	  int matrix_size;
	  my_rank = matrix_rank;
	  MPI_Comm_size(MPI_COMM_WORLD, &matrix_size);
	  node_count = matrix_size;


	  

	}


	void GradientAccumulator::init_Matrix(Matrix<float> * m){

	  matrix = m;
	  buffer = cn->OPS->zeros(m->rows, m->cols);
	  int slice_size = m->rows/node_count;
	  for(int i=0; i< m->rows; i = i+slice_size){
		v.push_back (cn->OPS->get_view( m, i, i+slice_size));
	    b.push_back (cn->OPS->get_view( buffer, i, i+slice_size));

	  }

	 
	}


	void GradientAccumulator::send_MPI(){


	  for(int i=0; i<=node_count; i++){ 
	  
	      MPI_Scatter(
		matrix,
		v[1]->size,
		MPI_FLOAT,
		b[i],
		v[1]->size,
		MPI_FLOAT,
		i,
		MPI_COMM_WORLD);
	  }

	}




	void GradientAccumulator::recv_MPI(){

	  for(int i=0; i<= node_count; i++){

	      cn->OPS->add(b[i], b[my_rank], b[my_rank]);

	  }

	  MPI_Allgather(
		b[my_rank],
		b[1] ->size,
		MPI_FLOAT,
		matrix,
		v[1]-> size,
		MPI_FLOAT,
		MPI_COMM_WORLD);


	}





	    



#endif
