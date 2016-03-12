#include <pythonWrapper.h>

//simple wrapper to remove templates for the python interface
FloatMatrix *fill_matrix(int rows, int cols, float fill_value){	return fill_matrix<float>(rows, cols, fill_value); }
FloatMatrix *empty(int rows, int cols){	return empty<float>(rows, cols); }
void to_host(FloatMatrix *gpu, float *cpu){ to_host<float>(gpu, cpu); }
void to_gpu(float *cpu, FloatMatrix *gpu){ to_gpu<float>(cpu, gpu); }
ClusterNet *get_clusterNet(){ return new ClusterNetGPU(); }

void pow(FloatMatrix *A, FloatMatrix *out, float scalar){ elementWise<kpow>(A, out, scalar); }
void abs(FloatMatrix *A, FloatMatrix *out){ elementWise<kabs>(A, out); }
void log(FloatMatrix *A, FloatMatrix *out){ elementWise<klog>(A, out); }
void sqrt(FloatMatrix *A, FloatMatrix *out){ elementWise<ksqrt>(A, out); }
void logistic(FloatMatrix *A, FloatMatrix *out){ elementWise<klogistic>(A, out); }
void logistic_grad(FloatMatrix *A, FloatMatrix *out){ elementWise<klogistic_grad>(A, out); }
void rectified(FloatMatrix *A, FloatMatrix *out){ elementWise<krectified>(A, out); }
void rectified_grad(FloatMatrix *A, FloatMatrix *out){ elementWise<krectified_grad>(A, out); }
void wcopy(FloatMatrix *A, FloatMatrix *out){ elementWise<kcopy>(A, out); }


void add(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<kadd>(A, B, out); }
void sub(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<ksub>(A, B, out); }
void div(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<kdiv>(A, B, out); }
void mul(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<kmul>(A, B, out); }
void scalar_mul(FloatMatrix *A, FloatMatrix *out, float scalar){ elementWise<ksmul>(A, out, scalar); }
void eq(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<keq>(A, B, out); }
void lt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<klt>(A, B, out); }
void gt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<kgt>(A, B, out); }
void le(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<kle>(A, B, out); }
void ge(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<kge>(A, B, out); }
void ne(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<kne>(A, B, out); }
void squared_diff(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out){ elementWise<ksquared_diff>(A, B, out); }
void vadd(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out){ vectorWise<kvadd>(A, v, out); }
void vsub(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out){ vectorWise<kvsub>(A, v, out); }
void tmatrix(FloatMatrix *v, FloatMatrix *out){ vectorWise<ktmatrix>(v, out); }

void wlookup(FloatMatrix *embedding, FloatMatrix *idx_batch, FloatMatrix *out){ lookup(embedding,idx_batch,out); }

void rowMax(FloatMatrix *A, FloatMatrix *vout){ reduceToRows<rmax>(A, vout); }
void rowSum(FloatMatrix *A, FloatMatrix *vout){ reduceToRows<rsum>(A, vout); }
void rowMean(FloatMatrix *A, FloatMatrix *vout){ reduceToRows<rmean>(A, vout); }

void colMax(FloatMatrix *A, FloatMatrix *vout){ reduceToCols<rmax>(A, vout); }
void colSum(FloatMatrix *A, FloatMatrix *vout){ reduceToCols<rsum>(A, vout); }
void colMean(FloatMatrix *A, FloatMatrix *vout){ reduceToCols<rmean>(A, vout); }

float wMax(FloatMatrix *A){ return reduceToValue<rmax>(A); }
float wSum(FloatMatrix *A){ return reduceToValue<rsum>(A); }
float wMean(FloatMatrix *A){ return reduceToValue<rmean>(A); }
void freemat(FloatMatrix *A){ cudaFree(A->data); free(A); }

void wsortbykey(FloatMatrix *keys, FloatMatrix *values){ sortbykey<float>(keys, values); }

void wprintmat(FloatMatrix *A, int rstart, int rend, int cstart, int cend){ printmat(A,rstart,rend,cstart,cend); }
FloatMatrix *wget_view(FloatMatrix *A, int rstart, int rend){ return get_view(A, rstart, rend); }


