/*
 * export.c
 *
 *  Created on: Nov 24, 2015
 *      Author: tim
 */

#include <pythonWrapper.h>

//the extern C statement requires that we have no templates and overloaded methods, because
//it cannot infer the signature of the method otherwise in our python library
extern "C"
{
	FloatMatrix *fempty(ClusterNet *gpu, int rows, int cols){ return empty(gpu,rows, cols);}
	FloatMatrix *ffill_matrix(ClusterNet *gpu, int rows, int cols, float fill_value){ return fill_matrix(gpu,rows, cols, fill_value);}
	void fto_host(ClusterNet *gpu, FloatMatrix *gpumat, float *cpu){ to_host(gpu,gpumat, cpu);}
	void fto_gpu(ClusterNet *gpu, float *cpu, FloatMatrix *gpumat){ to_gpu(gpu,cpu, gpumat); }
	ClusterNet *fget_clusterNetCPU(){ return get_clusterNetCPU(); }
	void fdot(ClusterNet *gpu, FloatMatrix*A, FloatMatrix *B, FloatMatrix*C){ gpu->dot(A,B,C); }
	void frand(ClusterNet *gpu, int rows, int cols){ gpu->rand(rows, cols); }
	void frandn(ClusterNet *gpu, int rows, int cols){ gpu->randn(rows, cols); }
	void fsetRandomState(ClusterNet *gpu, int seed){ gpu->setRandomState(seed); }
	FloatMatrix *fT(ClusterNet *gpu, FloatMatrix * A){ return wT(gpu,A); }
	void ftranspose(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ wtranspose(gpu,A, out); }
	void ffabs(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return abs(gpu,A,out); }
	void flog(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return log(gpu,A,out); }
	void fsqrt(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return sqrt(gpu,A,out); }
	void fpow(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out, float scalar){ return pow(gpu,A,out, scalar); }
	void flogistic(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return logistic(gpu,A,out); }
	void flogistic_grad(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return logistic_grad(gpu,A,out); }
	void frectified(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return rectified(gpu,A,out); }
	void frectified_grad(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return rectified_grad(gpu,A,out); }
	void fcopy(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out){ return wcopy(gpu,A,out); }

	void fadd(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return add(gpu,A,B, out); }
	void fsub(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return sub(gpu,A,B, out); }
	void fmul(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return mul(gpu,A,B, out); }
	void fdiv(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return div(gpu,A,B, out); }
	void feq(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return eq(gpu,A,B, out); }
	void flt(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return lt(gpu,A,B, out); }
	void fgt(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return gt(gpu,A,B, out); }
	void fge(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return ge(gpu,A,B, out); }
	void fle(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return le(gpu,A,B, out); }
	void fne(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return ne(gpu,A,B, out); }
	void fsquared_diff(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *B, FloatMatrix *out){ return squared_diff(gpu,A,B, out); }

	void fscalar_mul(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *out, float scalar){ scalar_mul(gpu,A,out, scalar); }

	void fvadd(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *v, FloatMatrix *out){ return vadd(gpu,A,v, out); }
	void fvsub(ClusterNet *gpu, FloatMatrix * A, FloatMatrix *v, FloatMatrix *out){ return vsub(gpu,A,v, out); }
	void ftmatrix(ClusterNet *gpu, FloatMatrix *v, FloatMatrix *out){ return tmatrix(gpu,v, out); }

	void fslice(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out, int rstart, int rend, int cstart, int cend){ wslice(gpu,A, out, rstart, rend, cstart, cend); }
	void fsoftmax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ wsoftmax(gpu,A, out);}
	void fargmax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ wargmax(gpu,A, out);}
	FloatCPUtoCPUBatchAllocator *fget_CPUtoCPUBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size)
	{ return new FloatCPUtoCPUBatchAllocator(gpu, X, y, rows, colsX, colsY, batch_size); }


	void falloc_next_batch(FloatBatchAllocator *alloc){ alloc->allocate_next_batch_async(); }
	void freplace_current_with_next_batch(FloatBatchAllocator *alloc){ alloc->replace_current_with_next_batch(); }
	FloatMatrix *fgetOffBatchX(FloatBatchAllocator *alloc){ return alloc->nextoffbatchX; }
	FloatMatrix *fgetOffBatchY(FloatBatchAllocator *alloc){ return alloc->nextoffbatchY; }
	FloatMatrix *fgetBatchX(FloatBatchAllocator *alloc){ return alloc->get_current_batchX(); }
	FloatMatrix *fgetBatchY(FloatBatchAllocator *alloc){ return alloc->get_current_batchY(); }


	FloatMatrix *fget_view(ClusterNet *gpu, FloatMatrix *A, int rstart, int rend){ return wget_view(gpu,A, rstart, rend); }


	void frowMax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ rowMax(gpu,A, vout); }
	void frowSum(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ rowSum(gpu,A, vout); }
	void frowMean(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ rowMean(gpu,A, vout); }
	void fcolMax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ colMax(gpu,A, vout); }
	void fcolSum(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ colSum(gpu,A, vout); }
	void fcolMean(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ colMean(gpu,A, vout); }

	float ffmax(ClusterNet *gpu, FloatMatrix *A){ return wMax(gpu,A); }
	float ffsum(ClusterNet *gpu, FloatMatrix *A){ return wSum(gpu,A); }
	float ffmean(ClusterNet *gpu, FloatMatrix *A){ return wMean(gpu,A); }

	FloatMatrix *fto_pinned(ClusterNet *gpu, int rows, int cols, float *cpu){ return wto_pinned(gpu, rows, cols, cpu); }


#ifdef PHI
	ClusterNet *fget_clusterNet(){ return 0; }
	void ffree(FloatMatrix *A){}
#else
	void ffree(FloatMatrix *A){ freemat(A); }
	ClusterNet *fget_clusterNet(){ return get_clusterNet(); }
	Timer *fget_Timer(){ return new Timer(); }
	void ftick(Timer *t, char *name){ t->tick(std::string(name));}
	float ftock(Timer *t, char *name){ return t->tock(std::string(name));}

	FloatCPUBatchAllocator *fget_CPUBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size)
	{ return new FloatCPUBatchAllocator(gpu, X, y, rows, colsX, colsY, batch_size); }
	FloatGPUBatchAllocator *fget_GPUBatchAllocator(ClusterNet *gpu, float *X, float *y, int rows, int colsX, int colsY, int batch_size)
	{ return new FloatGPUBatchAllocator(gpu, X, y, rows, colsX, colsY, batch_size); }
#endif
	void fprintmat(ClusterNet *gpu, FloatMatrix *A, int rstart, int rend, int cstart, int cend){ wprintmat(gpu,A,rstart,rend,cstart,cend); }


	void flookup(ClusterNet *gpu, FloatMatrix *embedding, FloatMatrix *idx_batch, FloatMatrix *out){ wlookup(gpu,embedding,idx_batch,out); }

}
