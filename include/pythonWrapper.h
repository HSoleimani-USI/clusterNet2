#include <basicOps.cuh>
#include <clusterNet2.h>
#include <BatchAllocator.h>

typedef ClusterNet2<float> ClusterNet;
typedef Matrix<float> FloatMatrix;
typedef BatchAllocator FloatBatchAllocator;

ClusterNet *get_clusterNet();

FloatMatrix *fill_matrix(int rows, int cols, float fill_value);
FloatMatrix *empty(int rows, int cols);
void to_host(FloatMatrix *gpu, float *cpu);
void to_gpu(float *cpu, FloatMatrix *gpu);
void abs(FloatMatrix *A, FloatMatrix *out);
void log(FloatMatrix *A, FloatMatrix *out);
void sqrt(FloatMatrix *A, FloatMatrix *out);
void pow(FloatMatrix *A, FloatMatrix *out, float scalar);
void logistic(FloatMatrix *A, FloatMatrix *out, float scalar);
void logistic_grad(FloatMatrix *A, FloatMatrix *out, float scalar);
void rectified(FloatMatrix *A, FloatMatrix *out, float scalar);
void rectified_grad(FloatMatrix *A, FloatMatrix *out, float scalar);
void wcopy(FloatMatrix *A, FloatMatrix *out, float scalar);

void add(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void sub(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void div(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void mul(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void eq(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void lt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void gt(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void ge(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void le(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void ne(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);
void squared_diff(FloatMatrix *A, FloatMatrix *B, FloatMatrix *out, float scalar);


void vadd(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out, float scalar);
void vsub(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out, float scalar);
void tmatrix(FloatMatrix *A, FloatMatrix *v, FloatMatrix *out, float scalar);

void rowMax(FloatMatrix *A, FloatMatrix *vout);
void rowSum(FloatMatrix *A, FloatMatrix *vout);

float wMax(FloatMatrix *A);
float wSum(FloatMatrix *A);
void freemat(FloatMatrix *A);

void wsortbykey(FloatMatrix *keys, FloatMatrix *values);

