#include <BasicOpsWrapper.h>
#include <ClusterNet.h>
#include <ClusterNetCPU.h>
#include <BatchAllocator.h>
#include <CPUtoCPUBatchAllocator.h>


typedef BatchAllocator FloatBatchAllocator;
typedef Matrix<float> FloatMatrix;
typedef CPUtoCPUBatchAllocator FloatCPUtoCPUBatchAllocator;

#ifdef PHI
#else
	#include <ClusterNetGPU.h>
	#include <CPUBatchAllocator.h>
	#include <GPUBatchAllocator.h>
	#include <Timer.cuh>
	typedef GPUBatchAllocator FloatGPUBatchAllocator;
	typedef CPUBatchAllocator FloatCPUBatchAllocator;
	ClusterNet *get_clusterNet();
	void freemat(FloatMatrix *A);
#endif

ClusterNet *get_clusterNetCPU();

FloatMatrix *fill_matrix(ClusterNet *gpu, int rows, int cols, float fill_value);
FloatMatrix *empty(ClusterNet *gpu, int rows, int cols);
void to_host(ClusterNet *gpu, FloatMatrix *gpumat, float *cpu);
void to_gpu(ClusterNet *gpu, float *cpu, FloatMatrix *gpumat);
void abs(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void log(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void sqrt(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void pow(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out, float scalar);
void logistic(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void logistic_grad(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void rectified(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void rectified_grad(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void wcopy(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void wtranspose(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
FloatMatrix *wT(ClusterNet *gpu, FloatMatrix * A);

void add(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void sub(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void div(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void mul(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void scalar_mul(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out, float scalar);
void eq(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void lt(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void gt(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void ge(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void le(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void ne(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);
void squared_diff(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *B, FloatMatrix *out);


void vadd(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *v, FloatMatrix *out);
void vsub(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *v, FloatMatrix *out);
void tmatrix(ClusterNet *gpu, FloatMatrix *v, FloatMatrix *out);

void rowMax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout);
void rowSum(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout);
void rowMean(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout);
void colMax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout);
void colSum(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout);
void colMean(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *vout);

float wMax(ClusterNet *gpu, FloatMatrix *A);
float wSum(ClusterNet *gpu, FloatMatrix *A);
float wMean(ClusterNet *gpu, FloatMatrix *A);

FloatMatrix *wto_pinned(ClusterNet *gpu, int rows, int cols, float *cpu);

void wsortbykey(ClusterNet *gpu, FloatMatrix *keys, FloatMatrix *values);

void wprintmat(ClusterNet *gpu, FloatMatrix *A, int rstart, int rend, int cstart, int cend);
FloatMatrix *wget_view(ClusterNet *gpu, FloatMatrix *A, int rstart, int rend);

void wlookup(ClusterNet *gpu, FloatMatrix *embedding, FloatMatrix *idx_batch, FloatMatrix *out);

void wslice(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out, int rstart, int rend, int cstart, int cend);

void wsoftmax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);
void wargmax(ClusterNet *gpu, FloatMatrix *A, FloatMatrix *out);


