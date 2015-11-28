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
	void fdot(ClusterNet *gpu, FloatMatrix*A, FloatMatrix *B, FloatMatrix*C){ gpu->dot(A,B,C); }
	void frand(ClusterNet *gpu, int rows, int cols){ gpu->rand(rows, cols); }
	void frandn(ClusterNet *gpu, int rows, int cols){ gpu->randn(rows, cols); }
	void fsetRandomState(ClusterNet *gpu, int seed){ gpu->setRandomState(seed); }
	FloatMatrix *fT(FloatMatrix * A){ return transpose(A); }
	void ffabs(FloatMatrix * A, FloatMatrix *out){ return abs(A,out); }
	void flog(FloatMatrix * A, FloatMatrix *out){ return log(A,out); }
	void fsqrt(FloatMatrix * A, FloatMatrix *out){ return sqrt(A,out); }
	void fpow(FloatMatrix * A, FloatMatrix *out, float scalar){ return pow(A,out, scalar); }
	void flogistic(FloatMatrix * A, FloatMatrix *out, float scalar){ return logistic(A,out, scalar); }
	void flogistic_grad(FloatMatrix * A, FloatMatrix *out, float scalar){ return logistic_grad(A,out, scalar); }
	void frectified(FloatMatrix * A, FloatMatrix *out, float scalar){ return rectified(A,out, scalar); }
	void frectified_grad(FloatMatrix * A, FloatMatrix *out, float scalar){ return rectified_grad(A,out, scalar); }

	void fadd(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return add(A,B, out, scalar); }
	void fsub(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return sub(A,B, out, scalar); }
	void fmul(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return mul(A,B, out, scalar); }
	void fdiv(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return div(A,B, out, scalar); }
	void feq(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return eq(A,B, out, scalar); }
	void flt(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return lt(A,B, out, scalar); }
	void fgt(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return gt(A,B, out, scalar); }
	void fge(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return ge(A,B, out, scalar); }
	void fle(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return le(A,B, out, scalar); }
	void fne(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return ne(A,B, out, scalar); }
	void fsquared_diff(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out, float scalar){ return squared_diff(A,B, out, scalar); }

	void fvadd(FloatMatrix * A, FloatMatrix *v, FloatMatrix *out){ return vadd(A,v, out, 0.0f); }

	void fslice(FloatMatrix *A, FloatMatrix *out, int rstart, int rend, int cstart, int cend){ slice(A, out, rstart, rend, cstart, cend); }
	void fsoftmax(FloatMatrix *A, FloatMatrix *out){ softmax(A, out);}

}
