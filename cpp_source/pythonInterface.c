/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <pythonWrapper.h>


extern "C"
{
	FloatMatrix *fempty(int rows, int cols){ return empty(rows, cols);}
	FloatMatrix *ffill_matrix(int rows, int cols, float fill_value){ return fill_matrix(rows, cols, fill_value);}
	void fto_host(FloatMatrix *gpu, float *cpu){ to_host(gpu, cpu);}
	void fto_gpu(float *cpu, FloatMatrix *gpu){ to_gpu(cpu, gpu); }
	ClusterNet *fget_clusterNet(){ return get_clusterNet(); }
	void *fdot(ClusterNet *gpu, FloatMatrix*A, FloatMatrix *B, FloatMatrix*C){ gpu->dot(A,B,C); }
}
