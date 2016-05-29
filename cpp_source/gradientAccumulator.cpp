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
	  //cn->OPS->to_host(m, matrix->data);
	  buffer = cn->OPS->zeros(m->rows, m->cols);
	  cn->OPS->to_host(buffer, buffer->data);
	  //buffer-> data = (float*)malloc(m->bytes); 
	  //cn->OPS->to_host(buffer,buffer->data);
	  int slice_size = m->rows/node_count;
	  for(int i=0; i< m->rows; i = i+slice_size){
		v.push_back (cn->OPS->get_view( matrix, i, i+slice_size));
	        b.push_back (cn->OPS->get_view( buffer, i, i+slice_size));

	  }


}
	void GradientAccumulator::send_MPI(){

		
	    cn->OPS->to_host(matrix,matrix->data);
        
	cout << "The root  is ************* " << node_count;
	//cout <<"the size of v is" <<  endl << v.size();//sizeof(v)/sizeof(v[0]);
//	cout<<endl<<b.size();//sizeof(b)/sizeof(b[0]); 
	//cn->OPS->to_host(matrix, matrix->data);
	//cn->OPS->to_host(buffer,buffer->data);
	 for(int i=0; i<node_count; i++){ 
	  
	//cn->OPS->to_host(b[i],b[i]->data);
	      MPI_Scatter(
		matrix->data,
		v[0]->size,
		MPI_FLOAT,
		b[i]->data,
		v[0]->size,
		MPI_FLOAT,
		i,
		MPI_COMM_WORLD);
	  }

 }



	void GradientAccumulator::recv_MPI(){

	cout << "pre loop" << endl;
cout << "my rank is" << my_rank << endl;
cout << "my b length`is" << b.size() << endl;
	  for(int i=0; i< node_count; i++){
		
		if(my_rank == i) continue;
	      //cn->OPS->add(b[i], b[my_rank], b[my_rank]);

			#pragma omp parallel for
			for(int j=0; j < b[my_rank]->size ;j++)
				b[my_rank]->data[j] = b[my_rank]->data[j] + b[i]->data[j];
           }


	cout << "pre MPI" << endl;
MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allgather(
        b[my_rank]->data,
        b[0] ->size,
        MPI_FLOAT,
        matrix->data,
        v[0]->size,
        MPI_FLOAT,
        MPI_COMM_WORLD);
   
	cout << "pre to gpu" << endl;
     cn->OPS->to_gpu(matrix->data,matrix);
}
 


#endif
