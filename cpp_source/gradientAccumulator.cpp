/*
 * gradientAccumulator.cpp
 *
 *  Created on: Apr 4, 2016
 *      Author: Hanieh
 */

#include "BasicOpsWrapper.h"
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include <vector>
#include <string>
#include <string.h> // memcpy
#include <iostream>
#include <sstream>
#include <fstream>

#ifdef HDF5
	#include <hdf5.h>
#endif

using std::cout;
using std::endl;


void gradientAccumulator::init_MPI(int argc, char** argv) {

  MPI_Init(&argc , &argv);
  
  int matrix_rank;
  MPI_Comm_rank(MPI_COMM_MATRIX, &matrix_rank);
  int matrix_size;
  my_rank = matrix_rank;
  MPI_Comm_size(MPI_COMM_MATRIX, &matrix_size);
  node_count = matrix_size;


  

}


void init_Matrix(Matrix<float> * m){

  matrix = m;
  buffer = zeros(m->rows, m->cols);
  int slice_size = m->rows/node_count;
  for(int i=0; i< m->rows; i = i+slice_size){
  	v.push_back (get_view( &m, i, i+slice_size));
    b.push_back (get_view( &buffer, i, i+slice_size));

  }

 
}


void gradientAccumulator::send_MPI(){


  for(int i=0; i<=node_count; i++){ 
  
      MPI_Scatter(
        matrix,
        v[1]->size,
        MPI_Float,
        b[i],
        v[1]->size,
        MPI_Float,
        i,
        MPI_COMM_WORLD);
  }

}




void gradientAccumulator::recv_MPI(){

  for(int i=0; i<= node_count; i++){

      add(b[i], b[my_rank], b[my_rank]);

  }

  MPI_Allgather(
        b[my_rank],
        b[1] ->size,
        MPI_Float,
        matrix,
        v[1]-> size,
        MPI_Float,
        MPI_COMM_WORLD);


}





    





