#include <pythonWrapper.h>

//simple wrapper to remove templates for the python interface
FloatMatrix *fill_matrix(int rows, int cols, float fill_value){	return fill_matrix<float>(rows, cols, fill_value); }
FloatMatrix *empty(int rows, int cols){	return empty<float>(rows, cols); }
void to_host(FloatMatrix *gpu, float *cpu){ to_host<float>(gpu, cpu); }
void to_gpu(float *cpu, FloatMatrix *gpu){ to_gpu<float>(cpu, gpu); }
ClusterNet *get_clusterNet(){ return new ClusterNet2<float>(); }
void abs(FloatMatrix *A, FloatMatrix *out){ elementWiseUnary<kabs>(A, out, 0.0f); }
void log(FloatMatrix *A, FloatMatrix *out){ elementWiseUnary<klog>(A, out, 0.0f); }
void sqrt(FloatMatrix *A, FloatMatrix *out){ elementWiseUnary<ksqrt>(A, out, 0.0f); }
void pow(FloatMatrix *A, FloatMatrix *out, float scalar){ elementWiseUnary<kpow>(A, out, scalar); }
void logistic(FloatMatrix *A, FloatMatrix *out, float scalar){ elementWiseUnary<klogistic>(A, out, scalar); }
void logistic_grad(FloatMatrix *A, FloatMatrix *out, float scalar){ elementWiseUnary<klogistic_grad>(A, out, scalar); }
void rectified(FloatMatrix *A, FloatMatrix *out, float scalar){ elementWiseUnary<krectified>(A, out, scalar); }
void rectified_grad(FloatMatrix *A, FloatMatrix *out, float scalar){ elementWiseUnary<krectified_grad>(A, out, scalar); }


void add(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<kadd>(A, B, out, scalar); }
void sub(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<ksub>(A, B, out, scalar); }
void div(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<kdiv>(A, B, out, scalar); }
void mul(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<kmul>(A, B, out, scalar); }
void eq(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<keq>(A, B, out, scalar); }
void lt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<klt>(A, B, out, scalar); }
void gt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<kgt>(A, B, out, scalar); }
void le(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<kle>(A, B, out, scalar); }
void ge(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<kge>(A, B, out, scalar); }
void ne(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<kne>(A, B, out, scalar); }
void squared_diff(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar){ elementWise<ksquared_diff>(A, B, out, scalar); }
void vadd(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out, float scalar){ vectorWise<kvadd>(A, v, out, scalar); }
void tmatrix(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out, float scalar){ vectorWise<ktmatrix>(A, v, out, scalar); }


void rowMax(FloatMatrix *A, FloatMatrix *vout){ reduceToRows<rmax>(A, vout); }
void rowSum(FloatMatrix *A, FloatMatrix *vout){ reduceToRows<rsum>(A, vout); }

