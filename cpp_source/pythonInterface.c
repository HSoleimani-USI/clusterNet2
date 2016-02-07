/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <pythonWrapper.h>
#include <Timer.cuh>

//the extern C statement requires that we have no templates and overloaded methods, because
//it cannot infer the signature of the method otherwise in our python library
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
	void *ftranspose(FloatMatrix *A, FloatMatrix *out){ transpose(A, out, out->cols, out->rows); }
	void ffabs(FloatMatrix * A, FloatMatrix *out){ return abs(A,out); }
	void flog(FloatMatrix * A, FloatMatrix *out){ return log(A,out); }
	void fsqrt(FloatMatrix * A, FloatMatrix *out){ return sqrt(A,out); }
	void fpow(FloatMatrix * A, FloatMatrix *out, float scalar){ return pow(A,out, scalar); }
	void flogistic(FloatMatrix * A, FloatMatrix *out){ return logistic(A,out); }
	void flogistic_grad(FloatMatrix * A, FloatMatrix *out){ return logistic_grad(A,out); }
	void frectified(FloatMatrix * A, FloatMatrix *out){ return rectified(A,out); }
	void frectified_grad(FloatMatrix * A, FloatMatrix *out){ return rectified_grad(A,out); }
	void fcopy(FloatMatrix * A, FloatMatrix *out){ return wcopy(A,out); }

	void fadd(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return add(A,B, out); }
	void fsub(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return sub(A,B, out); }
	void fmul(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return mul(A,B, out); }
	void fdiv(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return div(A,B, out); }
	void feq(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return eq(A,B, out); }
	void flt(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return lt(A,B, out); }
	void fgt(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return gt(A,B, out); }
	void fge(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return ge(A,B, out); }
	void fle(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return le(A,B, out); }
	void fne(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return ne(A,B, out); }
	void fsquared_diff(FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return squared_diff(A,B, out); }

	void fscalar_mul(FloatMatrix * A, FloatMatrix *out, float scalar){ scalar_mul(A,out, scalar); }

	void fvadd(FloatMatrix * A, FloatMatrix *v, FloatMatrix *out){ return vadd(A,v, out); }
	void fvsub(FloatMatrix * A, FloatMatrix *v, FloatMatrix *out){ return vsub(A,v, out); }
	void ftmatrix(FloatMatrix *v, FloatMatrix *out){ return tmatrix(v, out); }

	void fslice(FloatMatrix *A, FloatMatrix *out, int rstart, int rend, int cstart, int cend){ slice(A, out, rstart, rend, cstart, cend); }
	void fsoftmax(FloatMatrix *A, FloatMatrix *out){ softmax(A, out);}
	void fargmax(FloatMatrix *A, FloatMatrix *out){ argmax(A, out);}

	float *fto_pinned(int rows, int cols, float *cpu){ return to_pinned<float>(rows, cols, cpu)->data; }


	FloatBatchAllocator *fget_BatchAllocator(float *X, float *y, int rows, int colsX, int colsY, int batch_size)
	{ return new FloatBatchAllocator(X, y, rows, colsX, colsY, batch_size); }
	void falloc_next_batch(FloatBatchAllocator *alloc){ alloc->allocate_next_batch_async(); }
	void freplace_current_with_next_batch(FloatBatchAllocator *alloc){ alloc->replace_current_with_next_batch(); }
	FloatMatrix *fgetOffBatchX(FloatBatchAllocator *alloc){ return alloc->nextoffbatchX; }
	FloatMatrix *fgetOffBatchY(FloatBatchAllocator *alloc){ return alloc->nextoffbatchY; }

	FloatMatrix *fgetBatchX(FloatBatchAllocator *alloc){ return alloc->get_current_batchX(); }
	FloatMatrix *fgetBatchY(FloatBatchAllocator *alloc){ return alloc->get_current_batchY(); }


	void frowMax(FloatMatrix *A, FloatMatrix *vout){ rowMax(A, vout); }
	void frowSum(FloatMatrix *A, FloatMatrix *vout){ rowSum(A, vout); }

	float ffmax(FloatMatrix *A){ return wMax(A); }
	float ffsum(FloatMatrix *A){ return wSum(A); }


	Timer *fget_Timer(){ return new Timer(); }
	void ftick(Timer *t, char *name){ t->tick(std::string(name));}
	float ftock(Timer *t, char *name){ return t->tock(std::string(name));}

	void ffree(FloatMatrix *A){ freemat(A); }

	void fsortbykey(FloatMatrix *keys, FloatMatrix *values){ wsortbykey(keys, values); }

	void fprintmat(FloatMatrix *A, int rstart, int rend, int cstart, int cend){ wprintmat(A,rstart,rend,cstart,cend); }

}
