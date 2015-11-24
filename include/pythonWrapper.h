#include <basicOps.cuh>
#include <clusterNet2.h>

typedef ClusterNet2<float> ClusterNet;
typedef Matrix<float> FloatMatrix;

ClusterNet *get_clusterNet();

FloatMatrix *fill_matrix(int rows, int cols, float fill_value);
FloatMatrix *empty(int rows, int cols);
void to_host(FloatMatrix *gpu, float *cpu);
void to_gpu(float *cpu, FloatMatrix *gpu);

