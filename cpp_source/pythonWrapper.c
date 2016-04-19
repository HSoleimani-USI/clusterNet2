#include <pythonWrapper.h>

//simple wrapper to remove templates for the python interface
FloatMatrix *fill_matrix(ClusterNet *gpu, int rows, int cols, float fill_value){	return gpu->OPS->fill_matrix(rows, cols, fill_value); }
FloatMatrix *empty(ClusterNet *gpu, int rows, int cols){	return gpu->OPS->empty(rows, cols); }
void to_host(ClusterNet *gpu, FloatMatrix *gpumat, float *cpu){ gpu->OPS->to_host(gpumat, cpu); }
void to_gpu(ClusterNet *gpu, float *cpu, FloatMatrix *gpumat){ gpu->OPS->to_gpu(cpu, gpumat); }

ClusterNet *get_clusterNetCPU(){ return new ClusterNetCPU(); }

void pow(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out, float scalar){ gpu->OPS->pow(A, out, scalar); }
void abs(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->abs(A, out); }
void log(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->log(A, out); }
void sqrt(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->sqrt(A, out); }
void logistic(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->logistic(A, out); }
void logistic_grad(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->logistic_grad(A, out); }
void rectified(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->rectified(A, out); }
void rectified_grad(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->rectified_grad(A, out); }
void wcopy(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->copy(A, out); }
void wtranspose(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->transpose(A, out, out->cols, out->rows); }
FloatMatrix *wT(ClusterNet *gpu, FloatMatrix * A){  return gpu->OPS->transpose(A); }

void add(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->add(A, B, out); }
void sub(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->sub(A, B, out); }
void div(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->div(A, B, out); }
void mul(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->mul(A, B, out); }
void scalar_mul(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out, float scalar){ gpu->OPS->mul(A, out, scalar); }
void eq(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->equal(A, B, out); }
void lt(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->less_than(A, B, out); }
void gt(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->greater_than(A, B, out); }
void le(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->less_equal(A, B, out); }
void ge(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->greater_equal(A, B, out); }
void ne(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->not_equal(A, B, out); }
void squared_diff(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ gpu->OPS->squared_diff(A, B, out); }
void vadd(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *v, FloatMatrix *out){ gpu->OPS->vadd(A, v, out); }
void vsub(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *v, FloatMatrix *out){ gpu->OPS->vsub(A, v, out); }
void tmatrix(ClusterNet *gpu, FloatMatrix *v, FloatMatrix *out){ gpu->OPS->get_t_matrix(v, out); }

void wlookup(ClusterNet *gpu, FloatMatrix *embedding, FloatMatrix *idx_batch, FloatMatrix *out){ gpu->OPS->lookup(embedding,idx_batch,out); }

void rowMax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ gpu->OPS->reduceToRowsMax(A, vout); }
void rowSum(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ gpu->OPS->reduceToRowsSum(A, vout); }
void rowMean(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ gpu->OPS->reduceToRowsMean(A, vout); }

void colMax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ gpu->OPS->reduceToColsMax(A, vout); }
void colSum(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ gpu->OPS->reduceToColsSum(A, vout); }
void colMean(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout){ gpu->OPS->reduceToColsMean(A, vout); }

float wMax(ClusterNet *gpu, FloatMatrix *A){ return gpu->OPS->max(A); }
float wSum(ClusterNet *gpu, FloatMatrix *A){ return gpu->OPS->sum(A); }
float wMean(ClusterNet *gpu, FloatMatrix *A){ return gpu->OPS->mean(A); }
#ifdef PHI
#else
	void freemat(FloatMatrix *A){ cudaFree(A->data); free(A); }
	ClusterNet *get_clusterNet(){ return new ClusterNetGPU(); }
#endif

FloatMatrix *wto_pinned(ClusterNet *gpu, int rows, int cols, float *cpu){ return gpu->OPS->to_pinned(rows, cols, cpu); }



void wprintmat(ClusterNet *gpu, FloatMatrix *A, int rstart, int rend, int cstart, int cend){ gpu->OPS->printmat(A,rstart,rend,cstart,cend); }
FloatMatrix *wget_view(ClusterNet *gpu, FloatMatrix *A, int rstart, int rend){ return gpu->OPS->get_view(A, rstart, rend); }

void wslice(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out, int rstart, int rend, int cstart, int cend){ gpu->OPS->slice(A, out, rstart, rend, cstart, cend); }

void wsoftmax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->softmax(A, out);}
void wargmax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out){ gpu->OPS->argmax(A, out);}

