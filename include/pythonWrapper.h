#include <BasicOpsCUDA.cuh>
#include <ClusterNet.h>
#include <ClusterNetGPU.h>
#include <BatchAllocator.h>
#include <CPUBatchAllocator.h>
#include <GPUBatchAllocator.h>

typedef Matrix<float> FloatMatrix;
typedef BatchAllocator FloatBatchAllocator;
typedef CPUBatchAllocator FloatCPUBatchAllocator;
typedef GPUBatchAllocator FloatGPUBatchAllocator;


ClusterNet *get_clusterNet();

FloatMatrix *fill_matrix(int rows, int cols, float fill_value);
FloatMatrix *empty(int rows, int cols);
void to_host(FloatMatrix *gpu, float *cpu);
void to_gpu(float *cpu, FloatMatrix *gpu);
void abs(FloatMatrix *A, FloatMatrix *out);
void log(FloatMatrix *A, FloatMatrix *out);
void sqrt(FloatMatrix *A, FloatMatrix *out);
void pow(FloatMatrix *A, FloatMatrix *out, float scalar);
void logistic(FloatMatrix *A, FloatMatrix *out);
void logistic_grad(FloatMatrix *A, FloatMatrix *out);
void rectified(FloatMatrix *A, FloatMatrix *out);
void rectified_grad(FloatMatrix *A, FloatMatrix *out);
void wcopy(FloatMatrix *A, FloatMatrix *out);

void add(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void sub(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void div(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void mul(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void scalar_mul(FloatMatrix *A, FloatMatrix *out, float scalar);
void eq(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void lt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void gt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void ge(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void le(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void ne(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void squared_diff(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);


void vadd(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out);
void vsub(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out);
void tmatrix(FloatMatrix *v, FloatMatrix *out);

void rowMax(FloatMatrix *A, FloatMatrix *vout);
void rowSum(FloatMatrix *A, FloatMatrix *vout);
void rowMean(FloatMatrix *A, FloatMatrix *vout);
void colMax(FloatMatrix *A, FloatMatrix *vout);
void colSum(FloatMatrix *A, FloatMatrix *vout);
void colMean(FloatMatrix *A, FloatMatrix *vout);

float wMax(FloatMatrix *A);
float wSum(FloatMatrix *A);
float wMean(FloatMatrix *A);
void freemat(FloatMatrix *A);

void wsortbykey(FloatMatrix *keys, FloatMatrix *values);

void wprintmat(FloatMatrix *A, int rstart, int rend, int cstart, int cend);
FloatMatrix *wget_view(FloatMatrix *A, int rstart, int rend);

void wlookup(FloatMatrix *embedding, FloatMatrix *idx_batch, FloatMatrix *out);


